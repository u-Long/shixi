"""
线性 baseline：与 iTransformer 走完全相同的数据口径，用于判断时序建模是否带来增量。

相对旧版的修正（旧版的数字和模型不可比）:
  [1] NaN 有效性过滤窗口对齐 StockDataset
      StockDataset 用 seq_len 天窗口算 nan_ratio；旧版 baseline 默认只看 1 天。
      一只数据稀疏的股票能过 1 天的筛、过不了 30 天的筛，两边评估的截面
      根本不是同一批股票。现在用 --feat_win（默认=模型 seq_len）做有效性判定，
      特征本身仍只取 --seq_len 天，两者解耦。

  [2] 应用 universe_mask
      StockDataset 里有 `valid &= universe_mask[di]`，旧版 baseline 没有。
      cache 若用 --universe hs300 建，stocks.npy 仍是全市场，旧版会在
      成分股之外的票上一起算 IC。

  [3] 补 embargo
      ret_5d_open 实际用到 open[t+6]，span = horizon+1。旧版训练集最后几天的
      label 落在验证期内 —— 验证期信息漏进训练，等于给 baseline 开后门，
      而 StockDataset 有明确的 embargo。

  [4] label 按日归一化（--y_norm）
      模型用 rankic loss 时，Pearson 定义里自带减均值除标准差，配合
      DayBatchSampler 就是「按当日截面归一化」。Ridge 是 pooled MSE，
      没有这一步：y 里的市场整体涨跌原封不动进了目标函数。
      补上之后两边才在优化同一个东西。
      注意这不影响评估 —— spearmanr 在每日内部计算，对 y 的每日仿射变换不变。
      对照 MSE-loss 训练的模型时可以设 --y_norm none。

  [5] 新增零参数基线（Level-0）
      单因子 RankIC 排行 / 符号对齐等权合成 / IC 加权合成。
      Ridge 打不过等权合成 => 线性组合这条路到头了，问题在因子不在模型。
      权重只用训练集估计，不碰 val/test。

  [6] alpha 搜索复用预构建的验证矩阵，不再每次重读 mmap

用法:
  python scripts/linear_baseline.py
  python scripts/linear_baseline.py --seq_len 5 --train_stride 3
  python scripts/linear_baseline.py --y_norm none          # 对照 MSE-loss 模型
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir",    default="data/cache/cache_fea3_ret5do_new")
parser.add_argument("--seq_len",      type=int, default=1,
                    help="用于特征的回看天数，展平成 seq_len*F 维")
parser.add_argument("--feat_win",     type=int, default=30,
                    help="有效性判定窗口，必须等于模型的 seq_len，否则股票池不一致")
parser.add_argument("--horizon",      type=int, default=5,
                    help="label 的持有天数。embargo span = horizon+1（open 系列）")
parser.add_argument("--train_start",  default="2018-04-24")
parser.add_argument("--train_end",    default="2024-04-23")
parser.add_argument("--val_start",    default="2024-04-24")
parser.add_argument("--val_end",      default="2025-04-23")
parser.add_argument("--test_start",   default="2025-04-24")
parser.add_argument("--test_end",     default="2026-08-14")
parser.add_argument("--alpha",        type=float, default=1.0)
parser.add_argument("--nan_thresh",   type=float, default=0.3)
parser.add_argument("--min_stocks",   type=int, default=50)
parser.add_argument("--y_norm",       choices=["zscore", "demean", "none"], default="zscore",
                    help="label 的每日截面归一化。zscore 对应 rankic loss 里的 Pearson；"
                         "none 对应 MSE loss")
parser.add_argument("--train_stride", type=int, default=1,
                    help="训练集抽样步长。seq_len>1 时展平矩阵极大，用它降内存")
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


# ══════════════════════════════════════════════════════════════════════
# 加载 cache
# ══════════════════════════════════════════════════════════════════════
print(f"Loading cache: {args.cache_dir}")
feat_arr  = np.load(os.path.join(args.cache_dir, "feat_arr.npy"),  mmap_mode="r")   # (T,S,F)
label_arr = np.load(os.path.join(args.cache_dir, "label_arr.npy"), mmap_mode="r")   # (T,S)
dates     = np.load(os.path.join(args.cache_dir, "dates.npy"),        allow_pickle=True)
feat_cols = np.load(os.path.join(args.cache_dir, "feature_cols.npy"), allow_pickle=True)
feat_cols = [str(c) for c in feat_cols]

univ_path = os.path.join(args.cache_dir, "universe_mask.npy")
UNIV = np.load(univ_path, mmap_mode="r") if os.path.exists(univ_path) else None

dates_pd = pd.to_datetime(dates)
T, S, F  = feat_arr.shape
print(f"T={T}, S={S}, F={F}")
print(f"universe_mask: {'loaded, coverage=%.3f' % UNIV.mean() if UNIV is not None else 'none (全市场)'}")

MIN_DI  = max(args.seq_len, args.feat_win) - 1
EMB     = args.horizon + 1          # open 系列 label 的实际 span


def date_range_idx(start, end, embargo_to=None):
    """embargo_to 非 None 时，剔除 label 结束日已进入下一段的样本日。"""
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
print(f"Days after embargo(span={EMB}) — train={len(train_idx)}, "
      f"val={len(val_idx)}, test={len(test_idx)}")


# ══════════════════════════════════════════════════════════════════════
# 单日样本构建 —— 有效性口径与 StockDataset 完全一致
# ══════════════════════════════════════════════════════════════════════
def day_valid(di):
    """返回该日有效股票的布尔掩码，口径对齐 StockDataset。"""
    chk = feat_arr[di - args.feat_win + 1: di + 1, :, :]      # (feat_win, S, F)
    nan_ratio = np.isnan(chk).mean(axis=(0, 2))               # (S,)
    y = label_arr[di, :]
    valid = (~np.isnan(y)) & (nan_ratio <= args.nan_thresh)
    if UNIV is not None:
        valid = valid & np.asarray(UNIV[di, :], dtype=bool)
    return valid


def normalize_y(y):
    """每日截面归一化。不改变 spearman 评估结果，只影响拟合目标。"""
    if args.y_norm == "none":
        return y
    y = y - y.mean()
    if args.y_norm == "zscore":
        y = y / (y.std() + 1e-8)
    return y


def build_day(di, norm_y=True):
    """返回 (X, y)。X: (n, seq_len*F)，y: (n,)。样本不足时返回 (None, None)。"""
    if di < MIN_DI:
        return None, None
    valid = day_valid(di)
    if valid.sum() < args.min_stocks:
        return None, None

    w = feat_arr[di - args.seq_len + 1: di + 1, :, :]         # (seq, S, F)
    x = w.transpose(1, 0, 2).reshape(S, -1)[valid]            # (n, seq*F)
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)          # 0 = 截面均值（特征已 z-score）

    y = np.asarray(label_arr[di, :], dtype=np.float64)[valid]
    if norm_y:
        y = normalize_y(y)
    return x, y.astype(np.float32)


def collect(idx, stride=1, norm_y=True):
    """预构建并驻留内存，避免 alpha 搜索时反复读 mmap。"""
    out = []
    for di in idx[::stride]:
        x, y = build_day(di, norm_y=norm_y)
        if x is not None:
            out.append((di, x, y))
    return out


# ══════════════════════════════════════════════════════════════════════
# Level-0：零参数基线
# 权重只在训练集上估计，val/test 仅用于评估
# ══════════════════════════════════════════════════════════════════════
def day_factor_rank_ic(x_raw, y):
    """一天内，每个因子各自的 RankIC。x_raw: (n,F) 未展平。返回 (F,)"""
    rx = rankdata(x_raw, axis=0).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rx = (rx - rx.mean(0)) / (rx.std(0) + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return rx.T @ ry / len(ry)


print("\n" + "=" * 70)
print("Level-0  零参数基线")
print("=" * 70)
print("Computing per-factor RankIC on train ...")

ic_acc, n_days = np.zeros(F), 0
for di in train_idx[::max(1, len(train_idx) // 400)]:          # 抽样约 400 天即可稳定
    valid = day_valid(di)
    if valid.sum() < args.min_stocks:
        continue
    xr = np.nan_to_num(np.asarray(feat_arr[di, :, :])[valid], nan=0.0)
    yr = np.asarray(label_arr[di, :])[valid]
    ic_acc += day_factor_rank_ic(xr, yr)
    n_days += 1

single_ic = pd.Series(ic_acc / max(n_days, 1), index=feat_cols)
print(f"  ({n_days} sampled train days)")
print("\n单因子 RankIC Top-10 / Bottom-5 (train):")
srt = single_ic.sort_values(ascending=False)
print(pd.concat([srt.head(10), srt.tail(5)]).round(4).to_string())

# 符号对齐等权 & IC 加权，权重只来自训练集
sign_w = np.sign(single_ic.values)
sign_w[sign_w == 0] = 1.0
icw = single_ic.values / (np.abs(single_ic.values).sum() + 1e-12)


def eval_static_weight(idx, w, name):
    rics = []
    for di in idx:
        valid = day_valid(di)
        if valid.sum() < args.min_stocks:
            continue
        xr = np.nan_to_num(np.asarray(feat_arr[di, :, :])[valid], nan=0.0)
        yr = np.asarray(label_arr[di, :])[valid]
        rho, _ = spearmanr(xr @ w, yr)
        if not np.isnan(rho):
            rics.append(rho)
    r = np.array(rics)
    print(f"  {name:<34s} RankIC={r.mean():+.4f}  ICIR={r.mean()/(r.std()+1e-8):+.3f}")
    return r


print("\n静态权重基线：")
print("[val]")
eval_static_weight(val_idx, sign_w, "sign-aligned equal weight")
eval_static_weight(val_idx, icw,    "IC-weighted (train IC)")
print("[test]")
eval_static_weight(test_idx, sign_w, "sign-aligned equal weight")
eval_static_weight(test_idx, icw,    "IC-weighted (train IC)")


# ══════════════════════════════════════════════════════════════════════
# Level-1：Ridge
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Level-1  Ridge   (seq_len={args.seq_len}, y_norm={args.y_norm})")
print("=" * 70)

if args.seq_len > 1 and args.train_stride == 1:
    est_gb = len(train_idx) * S * args.seq_len * F * 4 / 1e9
    if est_gb > 8:
        print(f"[WARN] 训练矩阵估计约 {est_gb:.1f} GB，建议加 --train_stride")

print("Building matrices ...")
train_days = collect(train_idx, stride=args.train_stride, norm_y=True)
val_days   = collect(val_idx,   norm_y=False)      # 评估用原始 y
test_days  = collect(test_idx,  norm_y=False)

X_train = np.concatenate([x for _, x, _ in train_days], axis=0)
y_train = np.concatenate([y for _, _, y in train_days], axis=0)
print(f"  train: {X_train.shape[0]} samples × {X_train.shape[1]} features "
      f"({len(train_days)} days, stride={args.train_stride})")
print(f"  val:   {len(val_days)} days,  test: {len(test_days)} days")

from sklearn.linear_model import Ridge


def eval_days(model, days, name):
    rics = []
    for _, x, y in days:
        rho, _ = spearmanr(model.predict(x), y)
        if not np.isnan(rho):
            rics.append(rho)
    r = np.array(rics)
    print(f"  {name:<10s} RankIC={r.mean():+.4f}  std={r.std():.4f}  "
          f"ICIR={r.mean()/(r.std()+1e-8):+.3f}  IC>0={np.mean(r > 0):.3f}  ({len(r)}d)")
    return r


print("\nAlpha 搜索（按 val RankIC 选择）:")
results = {}
for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=a, fit_intercept=True).fit(X_train, y_train)
    rics = [spearmanr(m.predict(x), y)[0] for _, x, y in val_days]
    rics = np.array([r for r in rics if not np.isnan(r)])
    results[a] = (m, rics.mean())
    print(f"  alpha={a:>8.2f}   val RankIC={rics.mean():+.4f}")

best_alpha = max(results, key=lambda k: results[k][1])
best_model = results[best_alpha][0]
print(f"\n最优 alpha = {best_alpha}")

print("\nRidge (best alpha):")
tr_r = eval_days(best_model, train_days, "train")
va_r = eval_days(best_model, val_days,   "val")
te_r = eval_days(best_model, test_days,  "test")

if args.seq_len == 1:
    coef = pd.Series(best_model.coef_, index=feat_cols)
    print("\nRidge 系数 Top-15（按绝对值）—— 可与 model.factor_weights() 对照:")
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(15)
    print(pd.DataFrame({"coef": top.round(4),
                        "train_rank_ic": single_ic.reindex(top.index).round(4)}).to_string())


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Summary（与 iTransformer 对比时请确认：同一 cache、同一 seq_len 口径、"
      "同一 embargo）")
print("=" * 70)
print(f"{'Model':<34}{'val RankIC':>12}{'test RankIC':>14}{'test ICIR':>12}")
print("-" * 72)
print(f"{'Ridge(alpha=%g)' % best_alpha:<34}{va_r.mean():>+12.4f}"
      f"{te_r.mean():>+14.4f}{te_r.mean()/(te_r.std()+1e-8):>+12.3f}")
print(f"\ntrain RankIC = {tr_r.mean():+.4f}  "
      f"(与 val 差距大说明 Ridge 已过拟合，iTransformer 大概率更严重)")
print(f"\n注意：{EMB} 天重叠 label 使逐日 IC 序列高度自相关，ICIR 高估约 √{EMB} 倍。"
      f"\n      该高估对 baseline 与模型同等作用，横向比较仍然有效。")
