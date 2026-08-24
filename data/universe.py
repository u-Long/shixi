"""
股票池（Universe）管理模块
支持：
  all       — 全市场（非ST，正常交易）
  hs300     — 沪深300（历史成分，按季度快照）
  zz500     — 中证500（市值分层近似：按流通市值排名301-800）
  zz1000    — 中证1000（市值分层近似：按流通市值排名801-1800）
  zz800     — 沪深300 + 中证500
  zz2000    — 中证2000（历史成分，按季度快照，仅2023年后有数据）
  custom    — 自定义股票列表文件（每行一个 code）

用法（独立脚本）:
  python data/universe.py --universe hs300
  python data/universe.py --universe zz500 --date 2022-06-30

作为模块调用:
  from data.universe import get_universe_stocks
  stocks = get_universe_stocks('hs300', panel_df, date='2022-06-30')
  # -> list[str]，当日成分股
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

# 成分股权重文件路径
WEIGHT_FILES = {
    "hs300":  "/root/dmd/BaoStock/daily/market/hs300_weight.csv",
    "zz2000": "/root/dmd/BaoStock/daily/market/zz2000_weight.csv",
}

# 市值近似分层定义：(流通市值降序排名起点, 终点)，含两端
SIZE_UNIVERSE = {
    "zz500":  (301,  800),
    "zz1000": (801, 1800),
    "zz800":  (1,   800),   # hs300 + zz500
}


def _load_weight_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df


def get_universe_stocks(
    universe: str,
    panel: pd.DataFrame,
    date: Optional[str] = None,
    circ_mv_col: str = "circ_mv",    # panel 里流通市值列名（若有）
    custom_path: Optional[str] = None,
) -> list:
    """
    返回指定 universe 在指定 date 的股票列表。
    date=None 时返回全量（所有出现过的股票）。

    panel: MultiIndex (datetime, instrument) 的 DataFrame
    """
    universe = universe.lower()

    # ── 全市场 ────────────────────────────────────────────────────────────────
    if universe == "all":
        return sorted(panel.index.get_level_values("instrument").unique().tolist())

    # ── 自定义列表 ────────────────────────────────────────────────────────────
    if universe == "custom":
        if custom_path is None or not os.path.exists(custom_path):
            raise ValueError(f"custom universe needs --custom_path, got: {custom_path}")
        with open(custom_path) as f:
            return [l.strip() for l in f if l.strip()]

    # ── 历史成分文件（hs300 / zz2000）────────────────────────────────────────
    if universe in WEIGHT_FILES:
        wf = _load_weight_file(WEIGHT_FILES[universe])
        if date is None:
            return sorted(wf["con_code"].unique().tolist())
        query_date = pd.Timestamp(date)
        # 取 <= query_date 的最近一次快照
        available = wf[wf["trade_date"] <= query_date]["trade_date"]
        if available.empty:
            # 早于最早快照，取第一个快照
            snap_date = wf["trade_date"].min()
        else:
            snap_date = available.max()
        stocks = wf[wf["trade_date"] == snap_date]["con_code"].tolist()
        return sorted(stocks)

    # ── 市值分层近似（zz500 / zz1000 / zz800）────────────────────────────────
    if universe in SIZE_UNIVERSE:
        lo, hi = SIZE_UNIVERSE[universe]

        # 先把 hs300 挖掉（zz500/zz1000 不包含 hs300 成分）
        exclude_hs300 = universe in ("zz500", "zz1000")

        if date is None:
            # 没有指定日期，返回全量出现过的近似池（用最新日期的排名）
            all_dates = panel.index.get_level_values("datetime").unique()
            date = all_dates.max()

        query_date = pd.Timestamp(date)

        # 取当天面板数据
        if query_date in panel.index.get_level_values("datetime"):
            day_panel = panel.xs(query_date, level="datetime")
        else:
            # 取最近的一天
            avail = panel.index.get_level_values("datetime")
            closest = avail[avail <= query_date].max()
            day_panel = panel.xs(closest, level="datetime")

        # 需要流通市值字段来排名
        # panel_with_sidecar 有 circ_mv，普通 panel 没有，用 amount 作代理
        if circ_mv_col in day_panel.columns:
            mv = day_panel[circ_mv_col].dropna()
        elif "amount" in day_panel.columns:
            # 用20日均成交额代理流通市值
            mv = day_panel["amount"].dropna()
        else:
            raise ValueError("panel 中缺少流通市值或成交额字段，无法做市值分层")

        # 过滤正常交易（若有 trade_status 字段）
        if "trade_status" in day_panel.columns:
            mv = mv[day_panel["trade_status"] == 1]
        if "is_st" in day_panel.columns:
            mv = mv[day_panel["is_st"] == 0]

        # 降序排名（1=最大）
        ranked = mv.rank(ascending=False, method="first")

        # 排除 hs300
        if exclude_hs300 and os.path.exists(WEIGHT_FILES["hs300"]):
            hs300 = get_universe_stocks("hs300", panel, date=str(query_date.date()))
            ranked = ranked[~ranked.index.isin(hs300)]

        # 取 lo~hi 名
        stocks = ranked[(ranked >= lo) & (ranked <= hi)].index.tolist()
        return sorted(stocks)

    raise ValueError(
        f"Unknown universe: '{universe}'. "
        f"Choose from: all, hs300, zz500, zz1000, zz800, zz2000, custom"
    )


def build_universe_mask(
    universe: str,
    dates: list,
    stocks: list,
    panel: pd.DataFrame,
    custom_path: Optional[str] = None,
) -> np.ndarray:
    """
    构建 (T, S) bool 掩码，True 表示该日期该股票在 universe 内。
    用于 build_cache.py 过滤样本。
    """
    T = len(dates)
    S = len(stocks)
    stock2i = {s: i for i, s in enumerate(stocks)}
    mask = np.zeros((T, S), dtype=bool)

    if universe == "all":
        mask[:] = True
        return mask

    # 季度更新频率足够，按月取一次快照减少计算量
    prev_members = set()
    for ti, date in enumerate(dates):
        date_str = str(pd.Timestamp(date).date())
        # 只在月初重新查询（减少重复计算）
        if ti == 0 or pd.Timestamp(date).month != pd.Timestamp(dates[ti-1]).month:
            members = get_universe_stocks(
                universe, panel, date=date_str, custom_path=custom_path
            )
            prev_members = set(members)
        for s in prev_members:
            if s in stock2i:
                mask[ti, stock2i[s]] = True

    return mask


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="hs300")
    parser.add_argument("--date",     default=None)
    parser.add_argument("--custom_path", default=None)
    args = parser.parse_args()

    panel = pd.read_parquet("/root/dmd/BaoStock/panel.parquet")
    stocks = get_universe_stocks(
        args.universe, panel,
        date=args.date,
        custom_path=args.custom_path,
    )
    print(f"Universe: {args.universe}  Date: {args.date or 'all'}")
    print(f"Stock count: {len(stocks)}")
    print(f"Sample: {stocks[:10]}")
