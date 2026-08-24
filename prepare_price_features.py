"""
价格特征处理脚本
输入: /root/dmd/BaoStock/panel.parquet  (index: datetime/instrument, cols: OHLCV等)
输出: /root/workspace/syl/iTransformer/data/price_features.parquet
      同样的 MultiIndex (datetime, instrument)，每列是一个特征

特征列表 (共14个):
  基础收益:  ret_1d, ret_3d, ret_5d, ret_10d
  价格结构:  log_close, body_ratio, upper_shadow, lower_shadow, close_position, gap_open
  量价:      log_volume, log_amount, volume_ratio_20, turnover
"""

import pandas as pd
import numpy as np
import os

SRC = "/root/dmd/BaoStock/panel.parquet"
DST = "/root/workspace/syl/iTransformer/data/price_features.parquet"

os.makedirs(os.path.dirname(DST), exist_ok=True)

print("Loading panel...")
df = pd.read_parquet(SRC)
df = df.sort_index()

# 过滤：只保留正常交易日（trade_status==1, is_st==0）
df = df[(df["trade_status"] == 1) & (df["is_st"] == 0)]

# 按股票分组计算时序特征，避免跨股票计算
def calc_features(g):
    close = g["close"]
    open_ = g["open"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]
    amount = g["amount"]
    turnover = g["turnover"]

    feat = pd.DataFrame(index=g.index)

    # 收益率
    feat["ret_1d"]  = close.pct_change(1)
    feat["ret_3d"]  = close.pct_change(3)
    feat["ret_5d"]  = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)

    # 价格水平（对数化去量纲）
    feat["log_close"] = np.log(close)

    # K线结构
    rng = (high - low).clip(lower=1e-8)
    feat["body_ratio"]     = (close - open_).abs() / rng
    feat["upper_shadow"]   = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
    feat["lower_shadow"]   = (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng
    feat["close_position"] = (close - low) / rng
    feat["gap_open"]       = open_ / close.shift(1) - 1

    # 量价
    feat["log_volume"]     = np.log(volume.clip(lower=1))
    feat["log_amount"]     = np.log(amount.clip(lower=1))
    feat["volume_ratio_20"] = volume / volume.rolling(20).mean().clip(lower=1e-8)
    feat["turnover"]       = turnover

    return feat

print("Computing features per stock (this takes a few minutes)...")
result = df.groupby(level="instrument", group_keys=False).apply(calc_features)

# 截面 z-score：每个交易日对每个特征在所有股票上标准化
print("Cross-sectional z-score normalization...")
result = result.groupby(level="datetime").transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# 去掉 inf / nan
result = result.replace([np.inf, -np.inf], np.nan)

print(f"Final shape: {result.shape}")
print(f"Features: {result.columns.tolist()}")
print(f"NaN ratio: {result.isna().mean().mean():.4f}")

result.to_parquet(DST)
print(f"Saved -> {DST}")
