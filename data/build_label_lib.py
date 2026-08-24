"""
Label 库构建脚本
生成多种 label，统一保存为 (datetime×instrument, L) MultiIndex parquet：

  label_lib.parquet  — 所有 label 合并在一张表，每列一种 label

Label 列表:
  ret_10d_log      — log(close_{t+10}/close_t)           当前默认 label
  ret_5d_log       — log(close_{t+5}/close_t)
  ret_1d_log       — log(close_{t+1}/close_t)
  ret_20d_log      — log(close_{t+20}/close_t)
  ret_10d_open     — log(open_{t+11}/open_{t+1})         data_utils 风格（开开）
  ret_5d_open      — log(open_{t+6}/open_{t+1})
  ret_1d_open      — log(open_{t+2}/open_{t+1})
  ret_10d_cs_rank  — ret_10d_log 的截面百分位 rank [0,1]
  excess_10d       — ret_10d_log 减去当日市场平均（超额）
  direction_10d    — ret_10d_log > 0 的二分类标签 {0,1}
  vol_10d          — 未来10日日收益率标准差（波动预测）

用法:
  python data/build_label_lib.py
  python data/build_label_lib.py --start_date 2015-01-01
"""

import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", default="2015-01-01")
parser.add_argument("--end_date",   default="2026-08-20")
parser.add_argument("--out_dir",    default="data/feature_lib")
parser.add_argument("--panel_path", default="/root/dmd/BaoStock/panel.parquet")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
start = pd.Timestamp(args.start_date)
end   = pd.Timestamp(args.end_date)

print("Loading panel ...")
panel = pd.read_parquet(args.panel_path)
panel.index = panel.index.set_levels(
    pd.to_datetime(panel.index.get_level_values("datetime").unique()), level="datetime"
)
# 不过滤停牌/ST：label 需要完整时间轴
mask = (panel.index.get_level_values("datetime") >= start) & \
       (panel.index.get_level_values("datetime") <= end)
panel = panel[mask].sort_index()

print("Computing labels per stock ...")

def calc_labels(g):
    close = g["close"]
    open_ = g["open"]
    l = pd.DataFrame(index=g.index)

    # close-to-close
    l["ret_1d_log"]  = np.log((close.shift(-1)  / close).clip(lower=1e-8))
    l["ret_5d_log"]  = np.log((close.shift(-5)  / close).clip(lower=1e-8))
    l["ret_10d_log"] = np.log((close.shift(-10) / close).clip(lower=1e-8))
    l["ret_20d_log"] = np.log((close.shift(-20) / close).clip(lower=1e-8))

    # open-to-open（data_utils 风格：t+1开盘 买入，t+N+1开盘 卖出）
    l["ret_1d_open"]  = np.log((open_.shift(-2)  / open_.shift(-1)).clip(lower=1e-8))
    l["ret_5d_open"]  = np.log((open_.shift(-6)  / open_.shift(-1)).clip(lower=1e-8))
    l["ret_10d_open"] = np.log((open_.shift(-11) / open_.shift(-1)).clip(lower=1e-8))

    # 未来10日波动率
    fwd_rets = pd.concat(
        [close.shift(-i) / close.shift(-(i-1)) - 1 for i in range(1, 11)], axis=1
    )
    l["vol_10d"] = fwd_rets.std(axis=1)

    # 方向分类
    l["direction_10d"] = (l["ret_10d_log"] > 0).astype(np.float32)

    return l

labels = panel.groupby(level="instrument", group_keys=False).apply(calc_labels)
labels = labels.replace([np.inf, -np.inf], np.nan)

# 截面 rank [0,1]
print("Computing cross-sectional rank label ...")
labels["ret_10d_cs_rank"] = labels["ret_10d_log"].groupby(level="datetime").rank(pct=True)

# 超额收益（减去当日截面均值）
print("Computing excess return label ...")
daily_mean = labels["ret_10d_log"].groupby(level="datetime").transform("mean")
labels["excess_10d"] = labels["ret_10d_log"] - daily_mean

print(f"\nLabel stats:")
print(labels.describe().T[["mean","std","min","max"]].round(4))

out = os.path.join(args.out_dir, "label_lib.parquet")
labels.to_parquet(out)
print(f"\nSaved: {labels.shape} -> {out}")
print(f"Cols: {labels.columns.tolist()}")
print(f"NaN: {labels.isna().mean().mean():.4f}")
