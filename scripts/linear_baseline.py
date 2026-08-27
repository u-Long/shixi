"""
线性回归 baseline：用同一份 fea3 cache 的最近 1 天特征（F 维）训练 Ridge 回归，
预测截面收益率。相同日期划分，相同评估指标（daily RankIC）。

如果 iTransformer 打不过这个 baseline，说明时序建模没有带来增量价值。

用法:
  cd /root/workspace/syl/iTransformer
  python scripts/linear_baseline.py
  python scripts/linear_baseline.py --cache_dir data/cache/cache_fea3_ret5do_new
  python scripts/linear_baseline.py --seq_len 5   # 用最近 5 天特征展平
"""

import argparse
import os
import sys
import numpy as np
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir",   default="data/cache/cache_fea3_ret5do_new")
parser.add_argument("--seq_len",     type=int, default=1, help="回看窗口，特征展平为 seq_len*F 维")
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
parser.add_argument("--alpha",       type=float, default=1.0, help="Ridge 正则系数")
parser.add_argument("--nan_thresh",  type=float, default=0.3, help="股票 NaN 率上限，超过则剔除")
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── 加载 cache ────────────────────────────────────────────────────────────────
print(f"Loading cache: {args.cache_dir}")
feat_arr  = np.load(os.path.join(args.cache_dir, "feat_arr.npy"),  mmap_mode="r")  # (T, S, F)
label_arr = np.load(os.path.join(args.cache_dir, "label_arr.npy"), mmap_mode="r")  # (T, S)
dates     = np.load(os.path.join(args.cache_dir, "dates.npy"),     allow_pickle=True)
feat_cols = np.load(os.path.join(args.cache_dir, "feature_cols.npy"), allow_pickle=True)

import pandas as pd
dates_pd = pd.to_datetime(dates)
T, S, F  = feat_arr.shape
print(f"T={T}, S={S}, F={F}")

def date_range_idx(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return np.where((dates_pd >= s) & (dates_pd <= e))[0]

train_idx = date_range_idx(args.train_start, args.train_end)
val_idx   = date_range_idx(args.val_start,   args.val_end)
test_idx  = date_range_idx(args.test_start,  args.test_end)
print(f"Train days={len(train_idx)}, Val days={len(val_idx)}, Test days={len(test_idx)}")

seq = args.seq_len

def build_day_samples(di):
    """返回 (X, y, valid_mask)，X: (valid_S, seq*F)，y: (valid_S,)"""
    if di < seq:
        return None, None, None
    # 特征窗口 (seq, S, F) -> (S, seq*F)
    window = feat_arr[di - seq + 1: di + 1, :, :]   # (seq, S, F)
    x = window.transpose(1, 0, 2).reshape(S, -1).copy()  # (S, seq*F)
    y = label_arr[di, :].copy()

    # valid：label 有效 + NaN 率在阈值内
    nan_ratio = np.isnan(x).mean(axis=1)             # (S,)
    valid = (~np.isnan(y)) & (nan_ratio <= args.nan_thresh)
    if valid.sum() < 50:
        return None, None, None

    x = x[valid]
    y = y[valid]
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)
    return x, y, valid

# ── 收集训练集，拟合 Ridge ────────────────────────────────────────────────────
print("\nBuilding train matrix ...")
X_list, y_list = [], []
for di in train_idx:
    x, y, _ = build_day_samples(di)
    if x is None:
        continue
    X_list.append(x)
    y_list.append(y)

X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)
print(f"Train samples: {len(X_train)}, features: {X_train.shape[1]}")

from sklearn.linear_model import Ridge
print(f"Fitting Ridge(alpha={args.alpha}) ...")
model = Ridge(alpha=args.alpha, fit_intercept=True)
model.fit(X_train, y_train)
print("Done.")

# ── 特征权重（单因子视角，seq=1 时等价于线性因子权重）─────────────────────────
if seq == 1:
    coef = model.coef_  # (F,)
    order = np.argsort(np.abs(coef))[::-1]
    print("\n模型系数 Top-20（绝对值，降序）:")
    print(f"{'Rank':>4}  {'Factor':<45}  {'Coef':>10}")
    print("-" * 65)
    for rank, i in enumerate(order[:20]):
        print(f"{rank+1:>4}  {str(feat_cols[i]):<45}  {coef[i]:>+10.4f}")

# ── 评估函数 ──────────────────────────────────────────────────────────────────
def eval_split(idx, name):
    rics = []
    for di in idx:
        x, y, _ = build_day_samples(di)
        if x is None:
            continue
        pred = model.predict(x)
        rho, _ = spearmanr(pred, y)
        if not np.isnan(rho):
            rics.append(rho)
    rics = np.array(rics)
    print(f"\n{name} ({len(rics)} days):")
    print(f"  RankIC mean : {rics.mean():.4f}")
    print(f"  RankIC std  : {rics.std():.4f}")
    print(f"  ICIR        : {rics.mean() / (rics.std() + 1e-8):.4f}")
    print(f"  IC>0 ratio  : {(rics > 0).mean():.3f}")
    return rics

train_rics = eval_split(train_idx, "Train")
val_rics   = eval_split(val_idx,   "Val")
test_rics  = eval_split(test_idx,  "Test")

# ── 搜索最优 alpha ────────────────────────────────────────────────────────────
print("\n--- Alpha 搜索（val RankIC）---")
best_alpha, best_val_ic = args.alpha, val_rics.mean()
for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=a, fit_intercept=True).fit(X_train, y_train)
    rics = []
    for di in val_idx:
        x, y, _ = build_day_samples(di)
        if x is None: continue
        pred = m.predict(x)
        rho, _ = spearmanr(pred, y)
        if not np.isnan(rho): rics.append(rho)
    mean_ic = np.mean(rics) if rics else np.nan
    marker = " <-- best" if mean_ic > best_val_ic else ""
    print(f"  alpha={a:>8.2f}  val RankIC={mean_ic:.4f}{marker}")
    if mean_ic > best_val_ic:
        best_val_ic, best_alpha = mean_ic, a

print(f"\n最优 alpha={best_alpha}，重新拟合并在 test 上评估 ...")
best_model = Ridge(alpha=best_alpha, fit_intercept=True).fit(X_train, y_train)

test_rics_best = []
for di in test_idx:
    x, y, _ = build_day_samples(di)
    if x is None: continue
    pred = best_model.predict(x)
    rho, _ = spearmanr(pred, y)
    if not np.isnan(rho): test_rics_best.append(rho)
test_rics_best = np.array(test_rics_best)
print(f"Test RankIC  mean={test_rics_best.mean():.4f}  "
      f"std={test_rics_best.std():.4f}  "
      f"ICIR={test_rics_best.mean()/(test_rics_best.std()+1e-8):.4f}")

print("\n=== Summary ===")
print(f"{'Model':<30}  {'Test RankIC':>12}  {'ICIR':>8}")
print("-" * 55)
print(f"{'Ridge(alpha='+str(args.alpha)+')':<30}  {test_rics.mean():>12.4f}  {test_rics.mean()/(test_rics.std()+1e-8):>8.4f}")
print(f"{'Ridge(alpha='+str(best_alpha)+', tuned)':<30}  {test_rics_best.mean():>12.4f}  {test_rics_best.mean()/(test_rics_best.std()+1e-8):>8.4f}")
print("\n（iTransformer 结果从 checkpoints/*/config.json 读取后手动对比）")
