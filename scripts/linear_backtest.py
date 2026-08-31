"""
Ridge 线性回归 baseline 完整流水线：
  1. 用 train 集拟合 Ridge
  2. 在 test 集逐日推理，生成 pred_detail.parquet（格式与 backtest.py 一致）
  3. 调用 backtest.py --pred_parquet 跑完整回测（IC、分层、TopKDrop、summary.md）

用法:
  cd /root/workspace/syl/iTransformer
  python scripts/linear_backtest.py
  python scripts/linear_backtest.py --cache_dir data/cache/cache_fea2_ret5do_new \
      --out_dir backtest_results/ridge_fea2
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir",   default="data/cache/cache_fea2_ret5davgo")
parser.add_argument("--label_lib",   default="data/feature_lib/label_lib.parquet")
parser.add_argument("--out_dir",     default="backtest_results/ridge_fea2_ret5davgo")
parser.add_argument("--seq_len",     type=int,   default=1,
                    help="回看窗口天数，特征展平为 seq_len*F 维（建议1，等价于截面线性）")
parser.add_argument("--alpha",       type=float, default=1000.0,
                    help="Ridge 正则系数。linear_baseline.py alpha 搜索结果决定此值")
parser.add_argument("--y_norm",      choices=["zscore", "demean", "none"], default="zscore",
                    help="label 每日截面归一化。zscore 对应 iTransformer rankic loss 口径；"
                         "none 对应 mse loss 口径")
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
parser.add_argument("--nan_thresh",  type=float, default=0.3)
parser.add_argument("--feat_win",    type=int,   default=30,
                    help="有效性判定窗口，必须等于 iTransformer 的 seq_len，否则股票池不一致")
# 透传给 backtest.py 的参数
parser.add_argument("--topk",        type=int,   default=None)
parser.add_argument("--n_drop",      type=int,   default=None)
parser.add_argument("--horizon",     type=int,   default=5)
parser.add_argument("--benchmark",   default="/root/dmd/BaoStock/Index/sh.000001.csv")
parser.add_argument("--slippage_bps", type=float, default=0.0)
parser.add_argument("--weight_mode",  default="drift", choices=["drift", "equal"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── 加载 cache ────────────────────────────────────────────────────────────────
print(f"Loading cache: {args.cache_dir}")
feat_arr  = np.load(os.path.join(args.cache_dir, "feat_arr.npy"),  mmap_mode="r")
label_arr = np.load(os.path.join(args.cache_dir, "label_arr.npy"), mmap_mode="r")
dates     = np.load(os.path.join(args.cache_dir, "dates.npy"),     allow_pickle=True)
stocks    = np.load(os.path.join(args.cache_dir, "stocks.npy"),    allow_pickle=True)
feat_cols = np.load(os.path.join(args.cache_dir, "feature_cols.npy"), allow_pickle=True)

univ_path = os.path.join(args.cache_dir, "universe_mask.npy")
UNIV = np.load(univ_path, mmap_mode="r") if os.path.exists(univ_path) else None

dates_pd = pd.to_datetime(dates)
T, S, F  = feat_arr.shape
seq      = args.seq_len
MIN_DI   = max(seq, args.feat_win) - 1
EMB      = args.horizon + 1
print(f"T={T}, S={S}, F={F}, seq_len={seq}, feat_win={args.feat_win}")
print(f"universe_mask: {'loaded, coverage=%.3f' % UNIV.mean() if UNIV is not None else 'none (全市场)'}")


def date_range_idx(start, end, embargo_to=None):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    idx = np.where((dates_pd >= s) & (dates_pd <= e))[0]
    idx = idx[idx >= MIN_DI]
    if embargo_to is not None:
        lim = pd.Timestamp(embargo_to)
        idx = np.array([i for i in idx if dates_pd[min(i + EMB, T - 1)] < lim], dtype=int)
    return idx


train_idx = date_range_idx(args.train_start, args.train_end, args.val_start)
val_idx   = date_range_idx(args.val_start,   args.val_end,   args.test_start)
test_idx  = date_range_idx(args.test_start,  args.test_end)
print(f"Days after embargo(span={EMB}) — Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")


def get_features(di):
    """返回 (S, seq*F) 矩阵和有效 mask，口径对齐 StockDataset。"""
    if di < MIN_DI:
        return None, None
    # 有效性用 feat_win 判定（对齐 iTransformer 的 seq_len）
    chk = feat_arr[di - args.feat_win + 1: di + 1, :, :]   # (feat_win, S, F)
    nan_ratio = np.isnan(chk).mean(axis=(0, 2))             # (S,)
    valid = nan_ratio <= args.nan_thresh
    if UNIV is not None:
        valid = valid & np.asarray(UNIV[di, :], dtype=bool)
    # 特征仍只取 seq_len 天
    window = feat_arr[di - seq + 1: di + 1, :, :]           # (seq, S, F)
    x = window.transpose(1, 0, 2).reshape(S, -1).copy().astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    return x, valid


def normalize_y(y):
    """每日截面归一化，对齐 iTransformer 的 IC loss 口径。不影响 spearmanr 评估。"""
    if args.y_norm == "none":
        return y
    y = y - y.mean()
    if args.y_norm == "zscore":
        y = y / (y.std() + 1e-8)
    return y


# ── 拟合 Ridge ────────────────────────────────────────────────────────────────
print(f"\nBuilding train matrix (y_norm={args.y_norm}) ...")
X_list, y_list = [], []
for di in train_idx:
    x, valid = get_features(di)
    if x is None:
        continue
    lbl = label_arr[di, :].copy()
    mask = valid & ~np.isnan(lbl)
    if mask.sum() < 50:
        continue
    X_list.append(x[mask])
    y_list.append(normalize_y(lbl[mask]).astype(np.float32))

X_train = np.concatenate(X_list)
y_train = np.concatenate(y_list)
print(f"Train samples: {len(X_train)}, features: {X_train.shape[1]}")

# ── val 集预构建（alpha 搜索用）────────────────────────────────────────────────
val_days = []
for di in val_idx:
    x, valid = get_features(di)
    if x is None:
        continue
    lbl = label_arr[di, :].copy()
    mask = valid & ~np.isnan(lbl)
    if mask.sum() < 50:
        continue
    val_days.append((x[mask], lbl[mask].astype(np.float32)))

# ── Alpha 搜索（按 val RankIC 选择）──────────────────────────────────────────
print("\nAlpha 搜索（按 val RankIC 选择）:")
alpha_results = {}
for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=a, fit_intercept=True).fit(X_train, y_train)
    rics = [spearmanr(m.predict(x), y)[0] for x, y in val_days]
    rics = np.array([r for r in rics if not np.isnan(r)])
    alpha_results[a] = (m, rics.mean())
    print(f"  alpha={a:>8.2f}   val RankIC={rics.mean():+.4f}")

best_alpha = max(alpha_results, key=lambda k: alpha_results[k][1])
model = alpha_results[best_alpha][0]
print(f"最优 alpha = {best_alpha}（命令行 --alpha 仅作初始参考，以搜索结果为准）")
print("Done.")

# 因子权重（seq=1 时直接可读）
if seq == 1:
    coef = model.coef_
    order = np.argsort(np.abs(coef))[::-1]
    print("\n因子权重 Top-20（|coef| 降序）:")
    print(f"{'Rank':>4}  {'Factor':<45}  {'Coef':>10}")
    print("-" * 65)
    for rank, i in enumerate(order[:20]):
        print(f"{rank+1:>4}  {str(feat_cols[i]):<45}  {coef[i]:>+10.5f}")


# ── 全量推理（train+val+test，存 pred_detail 时只取 test）────────────────────
def infer_split(idx, split_name):
    date_col, stock_col, pred_col, label_col = [], [], [], []
    for di in idx:
        x, valid = get_features(di)
        if x is None:
            continue
        lbl  = label_arr[di, :].copy()
        pred = model.predict(x)          # (S,)
        date = dates_pd[di]
        sidx = np.where(valid)[0]
        date_col.extend([date] * len(sidx))
        stock_col.extend(stocks[sidx].tolist())
        pred_col.extend(pred[sidx].tolist())
        label_col.extend(lbl[sidx].tolist())

    df = pd.DataFrame({"date": date_col, "stock": stock_col,
                        "pred": pred_col, "label": label_col})
    if len(df) == 0:
        return df
    rics = (df.groupby("date")
              .apply(lambda g: spearmanr(g["pred"], g["label"])[0]
                     if len(g) > 10 else np.nan, include_groups=False)
              .dropna())
    print(f"{split_name}: {rics.mean():.4f} RankIC  ICIR={rics.mean()/(rics.std()+1e-8):.4f}"
          f"  IC>0={(rics>0).mean():.1%}  ({len(rics)} days)")
    return df

print("\n--- 推理各集 ---")
infer_split(train_idx, "Train")
infer_split(val_idx,   "Val  ")
test_df = infer_split(test_idx, "Test ")

# ── 合并 o2o，保存 pred_detail.parquet ───────────────────────────────────────
print("\nMerging ret_1d_open ...")
lib = pd.read_parquet(args.label_lib, columns=["ret_1d_open"])
lib = lib.reset_index()
lib.columns = ["date", "stock", "o2o"]
lib["date"] = pd.to_datetime(lib["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
test_df = test_df.merge(lib, on=["date", "stock"], how="left")
n_miss = test_df["o2o"].isna().sum()
if n_miss:
    print(f"  [WARN] {n_miss}/{len(test_df)} 条无 o2o（退市/停牌），剔除而非填 0")
test_df = test_df.dropna(subset=["o2o"]).reset_index(drop=True)

pred_path = os.path.join(args.out_dir, "pred_detail.parquet")
test_df.to_parquet(pred_path, index=False)
print(f"pred_detail saved -> {pred_path}  ({len(test_df)} rows)")

# ── 调用 backtest.py 跑完整回测 ───────────────────────────────────────────────
print("\n=== 调用 backtest.py ===")
cmd = [
    sys.executable, "scripts/backtest.py",
    "--pred_parquet",  pred_path,
    "--cache_dir",     args.cache_dir,
    "--label_lib",     args.label_lib,
    "--out_dir",       args.out_dir,
    "--horizon",       str(args.horizon),
    "--test_start",    args.test_start,
    "--test_end",      args.test_end,
    "--train_start",   args.train_start,
    "--train_end",     args.train_end,
    "--val_start",     args.val_start,
    "--val_end",       args.val_end,
    "--benchmark",      args.benchmark,
    "--slippage_bps",   str(args.slippage_bps),
    "--weight_mode",    args.weight_mode,
]
if args.topk is not None:
    cmd += ["--topk", str(args.topk)]
if args.n_drop is not None:
    cmd += ["--n_drop", str(args.n_drop)]

subprocess.run(cmd, check=True)
