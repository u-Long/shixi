"""
fea3 因子筛选 + 输出 parquet

从 fea3_v5_770.parquet 出发，做两步筛选：
  1. IC 门槛：|train 期 RankIC| < IC_THRESH 的因子剔除
  2. 相关性剪枝：按 |IC| 降序贪心，与已选集合内任意因子的绝对相关 > CORR_THRESH 则跳过

输出：
  data/feature_lib/fea3_selected.parquet   筛选后的因子库
  data/scripts/fea3_selected_meta.csv      每个因子的 IC / 去留原因，方便溯源

用法：
  cd /root/workspace/syl/iTransformer
  python data/scripts/build_fea3_selected.py
  python data/scripts/build_fea3_selected.py --ic_thresh 0.02 --corr_thresh 0.6 --top_k 25
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--fea3_file",   default="data/feature_lib/fea3_v5_770.parquet")
parser.add_argument("--cache_dir",   default="data/cache/cache_fea3_ret5do_new",
                    help="用于计算 IC 的 cache（已有的 fea3 cache）")
parser.add_argument("--out_file",    default="data/feature_lib/fea3_selected.parquet")
parser.add_argument("--meta_file",   default="data/scripts/fea3_selected_meta.csv")
parser.add_argument("--ic_thresh",   type=float, default=0.02,
                    help="|train RankIC| 最低门槛，低于此值直接剔除")
parser.add_argument("--corr_thresh", type=float, default=0.7,
                    help="与已选因子的最大允许绝对相关，超过则跳过")
parser.add_argument("--top_k",       type=int,   default=25,
                    help="最终保留因子数上限")
parser.add_argument("--train_end",   default="2024-04-23",
                    help="IC 计算只用 train 期，避免 forward-looking")
parser.add_argument("--sample_days", type=int,   default=400,
                    help="抽样天数，用于 IC 和相关矩阵计算，越大越准但越慢")
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── 从 cache 读 feat_arr（已是截面 z-score，和 fea3 parquet 完全对应）──────────
print(f"Loading cache: {args.cache_dir}")
feat_arr  = np.load(os.path.join(args.cache_dir, "feat_arr.npy"),  mmap_mode="r")
label_arr = np.load(os.path.join(args.cache_dir, "label_arr.npy"), mmap_mode="r")
dates     = np.load(os.path.join(args.cache_dir, "dates.npy"),     allow_pickle=True)
cols      = [str(c) for c in np.load(os.path.join(args.cache_dir, "feature_cols.npy"), allow_pickle=True)]

dates_pd = pd.to_datetime(dates)
T, S, F  = feat_arr.shape
print(f"T={T}, S={S}, F={F}")

# 只用 train 期
train_mask = dates_pd <= pd.Timestamp(args.train_end)
train_idx  = np.where(train_mask)[0]
print(f"Train days: {len(train_idx)}  (up to {args.train_end})")

# 抽样
step      = max(1, len(train_idx) // args.sample_days)
sample_idx = train_idx[::step]
print(f"Sampling {len(sample_idx)} days for IC / correlation ...")

# ── 计算单因子 RankIC ───────────────────────────────────────────────────────
ic_acc = np.zeros(F)
n_days = 0
for ti in sample_idx:
    lbl   = label_arr[ti]
    valid = np.isfinite(lbl)
    if valid.sum() < 50:
        continue
    for fi in range(F):
        fv   = feat_arr[ti, :, fi]
        mask = valid & np.isfinite(fv)
        if mask.sum() < 50:
            continue
        ic_acc[fi] += spearmanr(fv[mask], lbl[mask])[0]
    n_days += 1

ic_series = pd.Series(ic_acc / max(n_days, 1), index=cols)
print(f"\n单因子 RankIC（{n_days} 天均值）：")
print(ic_series.sort_values(key=abs, ascending=False).round(4).to_string())

# ── 计算因子间截面相关矩阵（按天平均）──────────────────────────────────────
print(f"\n计算因子相关矩阵 ...")
corr_acc = np.zeros((F, F))
n_corr   = 0
for ti in sample_idx:
    lbl   = label_arr[ti]
    valid = np.isfinite(lbl)
    if valid.sum() < 50:
        continue
    X = feat_arr[ti, :, :]           # (S, F)
    # 只取所有因子都有效的股票行
    row_valid = np.all(np.isfinite(X), axis=1) & valid
    if row_valid.sum() < 50:
        continue
    Xv = X[row_valid].astype(np.float64)
    # 列相关（pearson，已 z-score 所以等价于 cov）
    Xv -= Xv.mean(axis=0)
    norms = np.linalg.norm(Xv, axis=0) + 1e-12
    Xv /= norms
    corr_acc += Xv.T @ Xv / Xv.shape[0]
    n_corr += 1

corr_mat = corr_acc / max(n_corr, 1)
np.fill_diagonal(corr_mat, 1.0)
corr_df  = pd.DataFrame(np.abs(corr_mat), index=cols, columns=cols)
print(f"  相关矩阵计算完成（{n_corr} 天均值）")

# ── IC 门槛筛选 ──────────────────────────────────────────────────────────────
passed_ic = ic_series[ic_series.abs() >= args.ic_thresh].index.tolist()
dropped_ic = [c for c in cols if c not in passed_ic]
print(f"\n[IC 筛选] |IC| >= {args.ic_thresh}：{len(passed_ic)}/{F} 通过，剔除 {len(dropped_ic)} 个")
if dropped_ic:
    print(f"  剔除: {dropped_ic}")

# 按 |IC| 降序
passed_ic_sorted = sorted(passed_ic, key=lambda c: abs(ic_series[c]), reverse=True)

# ── 贪心相关性剪枝 ────────────────────────────────────────────────────────────
print(f"\n[相关性剪枝] corr_thresh={args.corr_thresh}, top_k={args.top_k}")
selected, dropped_corr, skip_reason = [], [], {}
for c in passed_ic_sorted:
    if len(selected) >= args.top_k:
        break
    if not selected:
        selected.append(c)
        continue
    max_corr = corr_df.loc[c, selected].max()
    if max_corr > args.corr_thresh:
        most_corr = corr_df.loc[c, selected].idxmax()
        dropped_corr.append(c)
        skip_reason[c] = f"corr={max_corr:.3f} with {most_corr}"
    else:
        selected.append(c)

print(f"  保留 {len(selected)} 个，剔除 {len(dropped_corr)} 个冗余因子")
if dropped_corr:
    for c in dropped_corr:
        print(f"    跳过 {c:<55s}  ({skip_reason[c]})")

# ── 输出 meta csv ─────────────────────────────────────────────────────────────
rows = []
for c in cols:
    if c in selected:
        status = "selected"
        reason = ""
    elif c in dropped_ic:
        status = "dropped_ic"
        reason = f"|IC|={abs(ic_series[c]):.4f} < {args.ic_thresh}"
    else:
        status = "dropped_corr"
        reason = skip_reason.get(c, "over top_k")
    rows.append({"factor": c, "train_ic": round(ic_series[c], 5),
                 "abs_ic": round(abs(ic_series[c]), 5),
                 "status": status, "reason": reason})

meta_df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False)
meta_df.to_csv(args.meta_file, index=False)
print(f"\n最终选中 {len(selected)} 个因子:")
for i, c in enumerate(selected, 1):
    print(f"  {i:2d}. {c:<55s}  IC={ic_series[c]:+.4f}")

# ── 从 fea3_v5_770 读取并过滤列，输出 parquet ──────────────────────────────
print(f"\nLoading {args.fea3_file} ...")
fea3 = pd.read_parquet(args.fea3_file, columns=selected)
print(f"  原始: {F} 列  →  筛选后: {len(selected)} 列")
print(f"  shape: {fea3.shape}")

fea3.to_parquet(args.out_file)
print(f"\nSaved -> {args.out_file}")
print(f"Meta   -> {args.meta_file}")
print(f"\n下一步:")
print(f"  python data/scripts/build_cache.py \\")
print(f"    --fea fea3 --fea3_file fea3_selected.parquet \\")
print(f"    --label ret_5d_open --tag fea3_selected_ret5do")
