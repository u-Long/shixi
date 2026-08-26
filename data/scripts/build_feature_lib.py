"""
特征库构建脚本
生成三个特征集，统一保存为 (datetime×instrument, F) MultiIndex parquet：

  fea1_price_basic.parquet      — 基础价格特征（来自 prepare_price_features.py 逻辑，截面z-score归一化）
  fea2_price_new.parquet        — data_utils.py 风格的价格+量价+估值+资金流特征（适配到 BaoStock 数据源）
  fea3_alpha191.parquet         — 20个筛选后的 Alpha191 因子（通过 Qlib DSL 计算）

用法:
  python data/build_feature_lib.py                     # 全部重建
  python data/build_feature_lib.py --fea 1 2           # 只建 fea1 和 fea2
  python data/build_feature_lib.py --start_date 2015-01-01
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--fea", nargs="+", type=int, default=[1, 2, 3], help="构建哪些特征集 1/2/3")
parser.add_argument("--start_date", default="2015-01-01")
parser.add_argument("--end_date",   default="2026-08-20")
parser.add_argument("--out_dir",    default="data/feature_lib")
parser.add_argument("--fea2_name",  default=None, help="fea2 输出文件名（不含路径），默认 fea2_price_new.parquet")
parser.add_argument("--panel_path", default="/root/dmd/BaoStock/panel.parquet")
parser.add_argument("--qlib_uri",   default="/root/dmd/BaoStock/qlib_fullmkt")
parser.add_argument("--factor_list",default="data/selected_factors.txt")
parser.add_argument("--factor_spec",default="/root/gp_factor_qlib/autofactorsetnew/factor_specs/alpha191_factor_spec_v1.yaml")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
start = pd.Timestamp(args.start_date)
end   = pd.Timestamp(args.end_date)


# ══════════════════════════════════════════════════════════════════════════════
# 工具：截面 z-score（按每个交易日在所有股票上标准化）
# ══════════════════════════════════════════════════════════════════════════════
def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="datetime").transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


def load_panel():
    print("Loading panel.parquet ...")
    df = pd.read_parquet(args.panel_path)
    df.index = df.index.set_levels(
        pd.to_datetime(df.index.get_level_values("datetime").unique()), level="datetime"
    )
    mask = (df.index.get_level_values("datetime") >= start) & \
           (df.index.get_level_values("datetime") <= end)
    df = df[mask]
    df = df[(df["trade_status"] == 1) & (df["is_st"] == 0)]
    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# fea1: 基础价格特征（与 prepare_price_features.py 一致）
# ══════════════════════════════════════════════════════════════════════════════
def build_fea1(panel):
    print("\n=== Building fea1_price_basic ===")

    def calc(g):
        close = g["close"]; open_ = g["open"]
        high = g["high"]; low = g["low"]
        volume = g["volume"]; amount = g["amount"]
        rng = (high - low).clip(lower=1e-8)
        f = pd.DataFrame(index=g.index)
        f["ret_1d"]          = close.pct_change(1)
        f["ret_3d"]          = close.pct_change(3)
        f["ret_5d"]          = close.pct_change(5)
        f["ret_10d"]         = close.pct_change(10)
        f["log_close"]       = np.log(close)
        f["body_ratio"]      = (close - open_).abs() / rng
        f["upper_shadow"]    = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
        f["lower_shadow"]    = (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng
        f["close_position"]  = (close - low) / rng
        f["gap_open"]        = open_ / close.shift(1) - 1
        f["log_volume"]      = np.log(volume.clip(lower=1))
        f["log_amount"]      = np.log(amount.clip(lower=1))
        f["volume_ratio_20"] = volume / volume.rolling(20).mean().clip(lower=1e-8)
        f["turnover"]        = g["turnover"]
        return f

    feat = panel.groupby(level="instrument", group_keys=False).apply(calc)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = cs_zscore(feat)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    out = os.path.join(args.out_dir, "fea1_price_basic.parquet")
    feat.to_parquet(out)
    print(f"fea1 saved: {feat.shape} -> {out}")
    print(f"  cols: {feat.columns.tolist()}")
    print(f"  NaN:  {feat.isna().mean().mean():.4f}")
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# fea2: data_utils.py 风格（适配 BaoStock，含量价+估值+资金流）
# ══════════════════════════════════════════════════════════════════════════════
def build_fea2(panel):
    print("\n=== Building fea2_price_new ===")

    # 加载额外字段：moneyflow + valuation
    # 从各股 csv 拼合成宽表
    print("  Loading moneyflow + valuation from per-stock CSVs ...")
    base_dir = "/root/dmd/BaoStock/daily"
    mf_records, val_records = [], []

    for market in ["SH", "SZ"]:
        mkt_dir = os.path.join(base_dir, market)
        if not os.path.isdir(mkt_dir):
            continue
        for code in os.listdir(mkt_dir):
            stock_dir = os.path.join(mkt_dir, code)

            # moneyflow
            mf_path = os.path.join(stock_dir, "moneyflow.csv")
            if os.path.exists(mf_path):
                try:
                    mf = pd.read_csv(mf_path)
                    mf["trade_date"] = pd.to_datetime(mf["trade_date"].astype(str))
                    # 转换 ts_code 为 instrument 格式（600000.SH -> 600000.SH）
                    mf["instrument"] = mf["ts_code"].str.replace(
                        r"(\d+)\.(SH|SZ)", lambda m: m.group(1) + "." + m.group(2), regex=True
                    )
                    mf = mf.set_index(["trade_date", "instrument"])
                    mf.index.names = ["datetime", "instrument"]
                    mf_records.append(mf.drop(columns=["ts_code"], errors="ignore"))
                except Exception:
                    pass

            # valuation
            val_path = os.path.join(stock_dir, "valuation.csv")
            if os.path.exists(val_path):
                try:
                    val = pd.read_csv(val_path)
                    val["date"] = pd.to_datetime(val["date"])
                    # code: sh.600000 -> 600000.SH
                    def conv(c):
                        parts = str(c).split(".")
                        if len(parts) == 2:
                            return parts[1] + "." + parts[0].upper()
                        return c
                    val["instrument"] = val["code"].apply(conv)
                    val = val.set_index(["date", "instrument"])
                    val.index.names = ["datetime", "instrument"]
                    val_records.append(val.drop(columns=["code", "isST"], errors="ignore"))
                except Exception:
                    pass

    mf_df  = pd.concat(mf_records,  axis=0).sort_index() if mf_records  else None
    val_df = pd.concat(val_records, axis=0).sort_index() if val_records else None

    def calc_new(g):
        close  = g["close"];  open_  = g["open"]
        high   = g["high"];   low    = g["low"]
        volume = g["volume"]; amount = g["amount"]
        pre_close = g["pre_close"]

        f = pd.DataFrame(index=g.index)

        # open: 相对5日前 open 的变化（data_utils 原始逻辑）
        f["open"]     = open_ / open_.shift(5).ffill() - 1
        f["pct_chg"]  = (close / pre_close - 1).clip(-0.3, 0.3)

        # 价格特征均相对当日 open 归一化
        for col, series in [("high", high), ("low", low), ("close", close), ("pre_close", pre_close)]:
            f[col] = series / open_ - 1

        # 移动均线相对 open 归一化
        for w in [5, 10, 15, 20, 25]:
            f[f"ma{w}"] = close.rolling(w).mean() / open_ - 1

        # 量和金额：除以10日滚动标准差
        for col, s in [("vol", volume), ("amount", amount)]:
            std = s.rolling(10).std().ffill().clip(lower=1e-8)
            f[col] = s / std

        # 换手率直接保留
        f["turnover_rate"] = g["turnover"]

        # 5/20日收益率 (用于 vwap 代理)
        f["ret_5d"]  = close.pct_change(5)
        f["ret_20d"] = close.pct_change(20)

        # 振幅
        f["amplitude"] = high / low - 1

        # 量比20日
        f["vol_ratio_20"] = volume / volume.rolling(20).mean().clip(lower=1e-8)

        return f

    feat = panel.groupby(level="instrument", group_keys=False).apply(calc_new)

    # 拼接估值字段（pe_ttm, pb, ps_ttm → 1阶差分）
    if val_df is not None:
        val_df = val_df[~val_df.index.duplicated(keep="first")]
        mask_v = (val_df.index.get_level_values("datetime") >= start) & \
                 (val_df.index.get_level_values("datetime") <= end)
        val_df = val_df[mask_v]
        for col in ["peTTM", "pbMRQ", "psTTM"]:
            if col in val_df.columns:
                s = val_df[col]
                diff = s.groupby(level="instrument").diff(1)
                feat[col + "_diff"] = diff

    # 拼接资金流字段（log+10日std归一化）
    if mf_df is not None:
        mf_df = mf_df[~mf_df.index.duplicated(keep="first")]
        mask_m = (mf_df.index.get_level_values("datetime") >= start) & \
                 (mf_df.index.get_level_values("datetime") <= end)
        mf_df = mf_df[mask_m]
        mf_cols = [c for c in [
            "buy_sm_vol","buy_sm_amount","sell_sm_vol","sell_sm_amount",
            "buy_md_vol","buy_md_amount","sell_md_vol","sell_md_amount",
            "buy_lg_vol","buy_lg_amount","sell_lg_vol","sell_lg_amount",
            "buy_elg_vol","buy_elg_amount","sell_elg_vol","sell_elg_amount",
            "net_mf_vol","net_mf_amount",
        ] if c in mf_df.columns]
        for col in mf_cols:
            s = np.log(mf_df[col].clip(lower=0) + 1)
            std = s.groupby(level="instrument").transform(
                lambda x: x.rolling(10).std().ffill().clip(lower=1e-8)
            )
            feat[col] = (s / std).reindex(feat.index)

    # clip 去极值（含估值差分三列，防止 PE/PB 翻转时产生极端值）
    clip_dict = {
        "open": (-0.2, 0.2), "pct_chg": (-0.2, 0.2),
        "high": (-0.2, 0.2), "low": (-0.2, 0.2),
        "close": (-0.2, 0.2), "pre_close": (-0.2, 0.2),
        "ma5": (-0.2, 0.2), "ma10": (-0.2, 0.2), "ma15": (-0.2, 0.2),
        "ma20": (-0.2, 0.2), "ma25": (-0.2, 0.2),
        "vol": (0, 10), "amount": (0, 10), "turnover_rate": (0, 20),
        "ret_5d": (-0.5, 0.5), "ret_20d": (-0.5, 0.5),
        "amplitude": (0, 0.4), "vol_ratio_20": (0, 10),
        "peTTM_diff": (-30, 30), "pbMRQ_diff": (-5, 5), "psTTM_diff": (-10, 10),
        "buy_sm_vol": (0, 100), "buy_sm_amount": (0, 100),
        "sell_sm_vol": (0, 100), "sell_sm_amount": (0, 100),
        "buy_md_vol": (0, 70), "buy_md_amount": (0, 70),
        "sell_md_vol": (0, 80), "sell_md_amount": (0, 80),
        "buy_lg_vol": (0, 50), "buy_lg_amount": (0, 50),
        "sell_lg_vol": (0, 80), "sell_lg_amount": (0, 80),
        "buy_elg_vol": (0, 30), "buy_elg_amount": (0, 30),
        "sell_elg_vol": (0, 40), "sell_elg_amount": (0, 40),
        "net_mf_vol": (-10, 10), "net_mf_amount": (-10, 10),
    }
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col, (lo, hi) in clip_dict.items():
        if col in feat.columns:
            feat[col] = feat[col].clip(lower=lo, upper=hi)
    feat = feat.fillna(0)

    # 截面 z-score：每日在全截面标准化，对齐模型输入期望（与 fea1/fea3 口径一致）
    feat = cs_zscore(feat)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    fname = args.fea2_name if args.fea2_name else "fea2_price_new.parquet"
    out = os.path.join(args.out_dir, fname)
    feat.to_parquet(out)
    print(f"fea2 saved: {feat.shape} -> {out}")
    print(f"  cols ({len(feat.columns)}): {feat.columns.tolist()}")
    print(f"  NaN:  {feat.isna().mean().mean():.4f}")
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# fea3: Alpha191 因子（Qlib DSL 计算）
# ══════════════════════════════════════════════════════════════════════════════
def build_fea3():
    print("\n=== Building fea3_alpha191 ===")
    import sqlite3

    DB_PATH = "/root/gp_factor_qlib/autofactorsetnew/factorlibrary/storage/library.db"

    # 读取选中因子列表
    with open(args.factor_list) as f:
        selected = [l.strip() for l in f if l.strip()]

    # 从 library.db 获取每个因子的 cache_path
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" for _ in selected)
    rows = conn.execute(
        f"SELECT factor_name, cache_path FROM library_factors WHERE factor_name IN ({placeholders})",
        selected,
    ).fetchall()
    conn.close()
    name2path = {r[0]: r[1] for r in rows}
    print(f"  Found {len(name2path)}/{len(selected)} factors in library cache")

    # 逐个读取并拼合
    dfs = []
    for name in selected:
        if name not in name2path:
            print(f"  [SKIP] {name} not in library")
            continue
        path = name2path[name]
        if not os.path.exists(path):
            print(f"  [SKIP] cache file missing: {path}")
            continue
        df = pd.read_parquet(path)
        # 过滤日期范围
        mask = (df.index.get_level_values("datetime") >= pd.Timestamp(args.start_date)) & \
               (df.index.get_level_values("datetime") <= pd.Timestamp(args.end_date))
        dfs.append(df[mask])

    if not dfs:
        raise RuntimeError("No factor cache files found")

    result = dfs[0].join(dfs[1:], how="outer") if len(dfs) > 1 else dfs[0]
    result = result.sort_index()

    # 截面 z-score
    result = result.replace([np.inf, -np.inf], np.nan)
    result = cs_zscore(result)
    result = result.replace([np.inf, -np.inf], np.nan)

    out = os.path.join(args.out_dir, "fea3_alpha191.parquet")
    result.to_parquet(out)
    print(f"fea3 saved: {result.shape} -> {out}")
    print(f"  cols: {result.columns.tolist()}")
    print(f"  NaN:  {result.isna().mean().mean():.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if 1 in args.fea or 2 in args.fea:
        panel = load_panel()
    if 1 in args.fea:
        build_fea1(panel)
    if 2 in args.fea:
        build_fea2(panel)
    if 3 in args.fea:
        build_fea3()
    print("\nAll done.")
