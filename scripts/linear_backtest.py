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
parser.add_argument("--cache_dir",   default="data/cache/cache_fea2_ret5do_new")
parser.add_argument("--label_lib",   default="data/feature_lib/label_lib.parquet")
parser.add_argument("--out_dir",     default="backtest_results/ridge_fea2")
parser.add_argument("--seq_len",     type=int,   default=1,
                    help="回看窗口天数，特征展平为 seq_len*F 维（建议1，等价于截面线性）")
parser.add_argument("--alpha",       type=float, default=100.0)
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
parser.add_argument("--nan_thresh",  type=float, default=0.3)
# 透传给 backtest.py 的参数
parser.add_argument("--topk",        type=int,   default=None)
parser.add_argument("--n_drop",      type=int,   default=None)
parser.add_argument("--horizon",     type=int,   default=5)
parser.add_argument("--benchmark",   default="/root/dmd/BaoStock/Index/sh.000001.csv")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── 加载 cache ────────────────────────────────────────────────────────────────
print(f"Loading cache: {args.cache_dir}")
feat_arr  = np.load(os.path.join(args.cache_dir, "feat_arr.npy"),  mmap_mode="r")
label_arr = np.load(os.path.join(args.cache_dir, "label_arr.npy"), mmap_mode="r")
dates     = np.load(os.path.join(args.cache_dir, "dates.npy"),     allow_pickle=True)
stocks    = np.load(os.path.join(args.cache_dir, "stocks.npy"),    allow_pickle=True)
feat_cols = np.load(os.path.join(args.cache_dir, "feature_cols.npy"), allow_pickle=True)

dates_pd = pd.to_datetime(dates)
T, S, F  = feat_arr.shape
seq      = args.seq_len
print(f"T={T}, S={S}, F={F}, seq_len={seq}")

def date_range_idx(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return np.where((dates_pd >= s) & (dates_pd <= e))[0]

train_idx = date_range_idx(args.train_start, args.train_end)
val_idx   = date_range_idx(args.val_start,   args.val_end)
test_idx  = date_range_idx(args.test_start,  args.test_end)
print(f"Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)} days")


def get_features(di):
    """返回 (S, seq*F) 矩阵和有效 mask"""
    if di < seq:
        return None, None
    window = feat_arr[di - seq + 1: di + 1, :, :]   # (seq, S, F)
    x = window.transpose(1, 0, 2).reshape(S, -1).copy().astype(np.float32)
    nan_ratio = np.isnan(x).mean(axis=1)
    valid = nan_ratio <= args.nan_thresh
    x = np.nan_to_num(x, nan=0.0)
    return x, valid


# ── 拟合 Ridge ────────────────────────────────────────────────────────────────
print(f"\nBuilding train matrix ...")
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
    y_list.append(lbl[mask].astype(np.float32))

X_train = np.concatenate(X_list)
y_train = np.concatenate(y_list)
print(f"Train samples: {len(X_train)}, features: {X_train.shape[1]}")

print(f"Fitting Ridge(alpha={args.alpha}) ...")
model = Ridge(alpha=args.alpha, fit_intercept=True)
model.fit(X_train, y_train)
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
    print(f"  [WARN] {n_miss} 条记录无 o2o，填 0")
test_df["o2o"] = test_df["o2o"].fillna(0.0)

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
    "--benchmark",     args.benchmark,
]
if args.topk is not None:
    cmd += ["--topk", str(args.topk)]
if args.n_drop is not None:
    cmd += ["--n_drop", str(args.n_drop)]

subprocess.run(cmd, check=True)
