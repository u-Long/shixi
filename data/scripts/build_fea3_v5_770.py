"""
构建 fea3_v5_770.parquet

相对上一版的关键修正
────────────────────────────────────────────────────────────────────────
1. [严重] ST/停牌过滤移到**因子计算之后**。
   旧版在算因子前就把停牌/ST 行删掉，导致 rolling/pct_change 在被挖洞的
   序列上运算：停牌 10 天的票，pct_change 会把停牌前后当成相邻交易日，
   算出一个 10 日累计涨幅当作单日收益。动量/波动率类因子全部被污染。
   现在：用完整序列算因子 → 再按 valid mask 过滤 → 再做截面标准化。
   （过滤必须在标准化之前，否则 ST 股会参与 mean/std 计算。）

2. [严重] 截面标准化顺序修正：先 winsorize，再 z-score。
   旧版 `z-score → clip(-3,3)`，异常值已经污染了 mean/std 才去截断。
   pe/pb/pcf 分母接近 0 时能到 1e5 量级：某天一只票 10000、其余在 [0,50]，
   会把 std 拉到 ~180，正常股票 z 值全挤在 -0.02 附近，clip 根本够不着，
   整个因子当天被压平。
   现在用 MAD（中位数绝对偏差）截尾，对异常值免疫，再算 mean/std。

3. market_return 广播改用 index.map，并加断言校验。
   旧版 `reindex(...).to_numpy()` 丢掉索引校验，一旦错位是**静默的**——
   每只股票拿到别人那天的市场收益，beta/残差类因子全废且不报错。

4. 新增 std/mad 退化保护。低频财务派生因子某天可能全市场同值，
   旧版除以 1e-8 会产出 1e8 量级的垃圾值。

5. 新增：表达式未来函数扫描、字段覆盖度报告、逐列 NaN 率报告。

6. --norm 可选 zscore(默认) / rank。rank 无异常值风险、分布天然均匀，
   若模型侧出现 attention 塌缩可切换对比。

输出：
  data/feature_lib/fea3_v5_770.parquet — (datetime×instrument, N) MultiIndex

用法:
  cd /root/workspace/syl/iTransformer
  python data/scripts/build_fea3_v5_770.py
  python data/scripts/build_fea3_v5_770.py --norm rank
  python data/scripts/build_fea3_v5_770.py --start_date 2015-01-01 --end_date 2026-08-20
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/root")

from gp_factor_qlib.core.expression_tree import parse_qlib_expr
from gp_factor_qlib.engine.gp_engine import compute_factor_series

parser = argparse.ArgumentParser()
parser.add_argument("--start_date",   default="2015-01-01")
parser.add_argument("--end_date",     default="2026-08-20")
parser.add_argument("--panel_path",   default="/root/dmd/BaoStock/panel.parquet")
parser.add_argument("--baostock_dir", default="/root/dmd/BaoStock/daily")
parser.add_argument("--yaml_path",
                    default="/root/gp_factor_qlib/autofactorsetnew/factor_specs/"
                            "daily_factor_library_full.yaml")
parser.add_argument("--out_file",     default=None,
                    help="输出 parquet 文件名，默认根据 yaml 文件名自动生成")
parser.add_argument("--out_dir",      default="data/feature_lib")
parser.add_argument("--norm", choices=["zscore", "rank"], default="zscore",
                    help="截面归一化方式。zscore=MAD winsorize + z-score；rank=截面分位数映射到[-1,1]")
parser.add_argument("--mad_k", type=float, default=5.0,
                    help="MAD winsorize 的倍数，越小截得越狠")
parser.add_argument("--min_cs_size", type=int, default=20,
                    help="截面有效样本数低于此值则该日该因子置为 NaN")
parser.add_argument("--max_nan_rate", type=float, default=0.5,
                    help="NaN 率超过此值的因子在最终报告中高亮告警")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
if args.out_file:
    OUT_PATH = os.path.join(args.out_dir, args.out_file)
else:
    yaml_stem = os.path.splitext(os.path.basename(args.yaml_path))[0]
    OUT_PATH = os.path.join(args.out_dir, f"fea3_{yaml_stem}.parquet")
start = pd.Timestamp(args.start_date)
end   = pd.Timestamp(args.end_date)
EPS = 1e-12

# 自动跳过依赖季报字段或数据源不可用字段的因子
_QUARTERLY_KW = {
    'profit_growth', 'revenue_growth', 'roe_ttm', 'total_liab',
    'total_assets', 'gross_margin', 'net_margin',
}
_MISSING_FIELDS = {
    'current_ratio', 'dividend_per_share', 'eps_ttm', 'industry_code',
    'lhb_buy_amount', 'lhb_sell_amount', 'north_cum_position',
    'rating_up_flag', 'roa_ttm', 'total_revenue', 'total_shares', 'pledge_shares',
}

REWRITE_EXPR = {}


# ══════════════════════════════════════════════════════════════════════
# 1. 读取 yaml 因子定义 + 未来函数扫描
# ══════════════════════════════════════════════════════════════════════
print(f"Loading factor yaml: {args.yaml_path}")
with open(args.yaml_path) as f:
    spec = yaml.safe_load(f)

_FIELD_PAT = re.compile(r"id\((\w+)\)")
def _should_skip(fac):
    expr = fac.get("expr", "") or ""
    fields = set(_FIELD_PAT.findall(expr))
    if fields & _QUARTERLY_KW:
        return "quarterly"
    if fields & _MISSING_FIELDS:
        return "missing_field"
    return None

skip_reasons = {}
factors = []
for fac in spec["factors"]:
    reason = _should_skip(fac)
    if reason:
        skip_reasons[fac["name"]] = reason
    else:
        factors.append(fac)

print(f"Factors to compute: {len(factors)} "
      f"(skipped {len(skip_reasons)}: "
      f"{sum(1 for r in skip_reasons.values() if r=='quarterly')} quarterly, "
      f"{sum(1 for r in skip_reasons.values() if r=='missing_field')} missing_field)")

# 扫描负偏移 —— Ref(x, -n) / Delay(x, -n) 这类是明确的未来函数
LOOKAHEAD_PAT = re.compile(r"\b(Ref|Delay|Shift)\s*\([^()]*,\s*-\s*\d+", re.IGNORECASE)
suspicious = [(f["name"], f["expr"]) for f in factors
              if f.get("expr") and LOOKAHEAD_PAT.search(f["expr"])]
if suspicious:
    print("\n" + "!" * 70)
    print("!! 检测到疑似未来函数（负偏移），请人工确认后再继续：")
    for n, e in suspicious:
        print(f"!!   {n}: {e}")
    print("!" * 70 + "\n")
    if input("继续? [y/N] ").strip().lower() != "y":
        sys.exit(1)
else:
    print("Lookahead scan: OK (未发现负偏移)")


# ══════════════════════════════════════════════════════════════════════
# 2. 加载 panel —— 注意：这里**不做** ST/停牌过滤
#    因子计算必须在完整时序上进行，否则 rolling / pct_change 会跨洞
# ══════════════════════════════════════════════════════════════════════
print(f"\nLoading panel: {args.panel_path}")
panel = pd.read_parquet(args.panel_path)
panel.index = panel.index.set_levels(
    pd.to_datetime(panel.index.get_level_values("datetime").unique()), level="datetime"
)
date_mask = (panel.index.get_level_values("datetime") >= start) & \
            (panel.index.get_level_values("datetime") <= end)
panel = panel[date_mask].sort_index()

# valid mask 先存下来，等因子算完再用
trade_status = panel.get("trade_status", pd.Series(1, index=panel.index))
is_st        = panel.get("is_st",        pd.Series(0, index=panel.index))
valid_mask   = (trade_status == 1) & (is_st == 0)

all_instruments = panel.index.get_level_values("instrument").unique()
all_dates       = panel.index.get_level_values("datetime").unique().sort_values()
print(f"Panel: {len(all_dates)} dates × {len(all_instruments)} instruments "
      f"(rows={len(panel)}, valid={valid_mask.sum()} / {valid_mask.mean():.2%})")


# ══════════════════════════════════════════════════════════════════════
# 3. 构建基础 data_ctx（量价派生）—— 全序列，不过滤
# ══════════════════════════════════════════════════════════════════════
close     = panel["close"].astype(float).rename("close")
open_     = panel["open"].astype(float).rename("open")
high      = panel["high"].astype(float).rename("high")
low       = panel["low"].astype(float).rename("low")
volume    = panel["volume"].astype(float).rename("volume")
amount    = panel["amount"].astype(float).rename("amount")
pre_close = panel["pre_close"].astype(float).rename("pre_close")

vwap    = (amount / (volume.abs() + EPS)).rename("vwap")
returns = close.groupby(level="instrument").pct_change(fill_method=None).rename("returns")

# ── market_return：全市场等权收益率，广播回 (datetime, instrument) ──
close_wide    = close.unstack("instrument")
mkt_ret_daily = close_wide.pct_change(fill_method=None).mean(axis=1)
mkt_ret_daily.name = "market_return"

# 用 index.map 而不是 reindex().to_numpy()，保留标签对齐语义
market_return = pd.Series(
    panel.index.get_level_values("datetime").map(mkt_ret_daily).to_numpy(),
    index=panel.index, name="market_return",
)

# 校验：同一天内必须唯一，且等于 mkt_ret_daily 对应值
_nuniq = market_return.groupby(level="datetime").nunique(dropna=False)
assert (_nuniq <= 1).all(), "market_return 在同一天内不唯一 —— 广播错位"
_probe = all_dates[min(100, len(all_dates) - 1)]
_lhs = market_return.xs(_probe, level="datetime").iloc[0]
_rhs = mkt_ret_daily.loc[_probe]
assert (pd.isna(_lhs) and pd.isna(_rhs)) or np.isclose(_lhs, _rhs), \
    f"market_return 对齐失败 @ {_probe}: {_lhs} vs {_rhs}"
print(f"market_return alignment check: OK (@{_probe.date()} = {_rhs:.6f})")

# turnover_rate：BaoStock turnover 列是百分比形式
turnover_rate = (pd.to_numeric(panel["turnover"], errors="coerce") / 100.0
                 ).rename("turnover_rate")

data_ctx = {
    "open":          open_,
    "high":          high,
    "low":           low,
    "close":         close,
    "volume":        volume,
    "amount":        amount,
    "pre_close":     pre_close,
    "vwap":          vwap,
    "returns":       returns,
    "market_return": market_return,
    "turnover_rate": turnover_rate,
}


# ══════════════════════════════════════════════════════════════════════
# 4. 加载日频财务/资金数据
# ══════════════════════════════════════════════════════════════════════
def _norm_inst(code, market):
    """sh.600000 / 600000.SH -> 600000.SH"""
    code = str(code).strip()
    if "." in code:
        left, right = code.rsplit(".", 1)
        if right.lower() in ("sh", "sz"):
            return f"{left}.{right.upper()}"
        if left.lower() in ("sh", "sz"):
            return f"{right}.{left.upper()}"
    return f"{code}.{market.upper()}"


def load_daily_to_daily(csv_name, date_col, col_map, baostock_dir):
    """逐股读取日频 CSV（moneyflow / valuation），对齐到 (datetime, instrument)。"""
    frames = []
    for market in ["SH", "SZ"]:
        mkt_dir = os.path.join(baostock_dir, market)
        if not os.path.isdir(mkt_dir):
            continue
        for code in os.listdir(mkt_dir):
            fpath = os.path.join(mkt_dir, code, csv_name)
            if not os.path.exists(fpath):
                continue
            try:
                df_ = pd.read_csv(fpath)
            except Exception:
                continue
            if df_.empty or date_col not in df_.columns:
                continue
            df_[date_col] = pd.to_datetime(df_[date_col].astype(str), errors="coerce")
            df_ = df_.dropna(subset=[date_col])

            code_col = "ts_code" if "ts_code" in df_.columns else (
                       "code" if "code" in df_.columns else None)
            raw = df_[code_col].iloc[0] if (code_col and len(df_)) else code
            inst = _norm_inst(raw, market)

            keep = [c for c in col_map if c in df_.columns]
            if not keep:
                continue
            df_ = df_[[date_col] + keep].rename(columns={date_col: "datetime", **col_map})
            df_["instrument"] = inst
            frames.append(df_)

    if not frames:
        return {}

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.set_index(["datetime", "instrument"]).sort_index()
    m = (all_df.index.get_level_values("datetime") >= start) & \
        (all_df.index.get_level_values("datetime") <= end)
    all_df = all_df[m]
    all_df = all_df[~all_df.index.duplicated(keep="first")]

    out_cols = list(col_map.values())
    return {c: all_df[c].astype(float) for c in out_cols if c in all_df.columns}


print("\nLoading moneyflow ...")
mf_fields = load_daily_to_daily(
    "moneyflow.csv", "trade_date",
    {"buy_elg_amount":  "buy_elg_amount",
     "buy_lg_amount":   "buy_lg_amount",
     "sell_elg_amount": "sell_elg_amount",
     "sell_lg_amount":  "sell_lg_amount"},
    args.baostock_dir,
)

print("Loading valuation ...")
val_fields = load_daily_to_daily(
    "valuation.csv", "date",
    {"peTTM":     "pe_ttm",
     "pbMRQ":     "pb_ttm",
     "psTTM":     "ps_ttm",
     "pcfNcfTTM": "pcf_ttm"},
    args.baostock_dir,
)

print("Loading margin_detail ...")
margin_fields = load_daily_to_daily(
    "margin_detail.csv", "trade_date",
    {"rzye": "margin_balance",   # 融资余额
     "rqye": "short_balance"},   # 融券余额
    args.baostock_dir,
)

print("Loading pledge_stat ...")
pledge_fields = load_daily_to_daily(
    "pledge_stat.csv", "end_date",
    {"pledge_ratio": "pledge_ratio"},
    args.baostock_dir,
)

# ── 覆盖度报告：覆盖率过低的字段，依赖它的因子基本是噪声 ──
all_ext = {**mf_fields, **val_fields, **margin_fields, **pledge_fields}
print("\nExternal field coverage (vs panel rows):")
for k, v in all_ext.items():
    aligned = v.reindex(panel.index)
    cov = aligned.notna().mean()
    flag = "  <-- LOW" if cov < 0.5 else ""
    print(f"  {k:<20s} coverage={cov:6.2%}  raw_rows={len(v)}{flag}")

for d in (mf_fields, val_fields, margin_fields, pledge_fields):
    data_ctx.update(d)

print(f"\ndata_ctx keys: {sorted(data_ctx.keys())}")


# ══════════════════════════════════════════════════════════════════════
# 5. 逐因子计算（全序列）
# ══════════════════════════════════════════════════════════════════════
results = {}
for fac in factors:
    name = fac["name"]
    expr_str = REWRITE_EXPR.get(name, fac.get("expr", ""))
    if not expr_str:
        print(f"[SKIP] {name}: no expr")
        continue

    print(f"Computing {name} ...", flush=True)
    try:
        node = parse_qlib_expr(expr_str)
        series = compute_factor_series(node, data_ctx, eval_backend="python")
        series = series.astype(float).rename(name)

        # 显式判断层级顺序，避免 names 为 [None, None] 时被误 swap
        if list(series.index.names) == ["instrument", "datetime"]:
            series = series.swaplevel().sort_index()
        elif list(series.index.names) != ["datetime", "instrument"]:
            series.index.names = ["datetime", "instrument"]
            series = series.sort_index()

        series = series.replace([np.inf, -np.inf], np.nan)
        print(f"  rows={len(series)}  NaN={series.isna().mean():.3f}")
        results[name] = series
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

print(f"\nSuccessfully computed: {len(results)}/{len(factors)} factors")
if not results:
    raise RuntimeError("All factors failed. Check errors above.")


# ══════════════════════════════════════════════════════════════════════
# 6. 合并 → 过滤 ST/停牌 → 截面归一化
#    顺序不能变：过滤必须在归一化之前，否则 ST 股污染 mean/std
# ══════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.index.names = ["datetime", "instrument"]
df = df.sort_index().replace([np.inf, -np.inf], np.nan)
print(f"\nRaw factor frame: {df.shape}")

_vm = valid_mask.reindex(df.index, fill_value=False)
df = df[_vm]
print(f"After ST/suspension filter: {df.shape} (dropped {(~_vm).sum()} rows)")


def cs_zscore(x: pd.Series) -> pd.Series:
    """MAD winsorize 后再 z-score。对每个 (交易日, 因子) 独立调用。"""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    v = x.dropna()
    if len(v) < args.min_cs_size:
        return out

    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    if mad < 1e-12:
        # 全截面近似同值 —— 无排序信息，置 0（中性）而非除以 eps 产出巨值
        out.loc[v.index] = 0.0
        return out

    v = v.clip(med - args.mad_k * mad, med + args.mad_k * mad)
    sd = v.std()
    if sd < 1e-12:
        out.loc[v.index] = 0.0
        return out

    out.loc[v.index] = (v - v.mean()) / sd
    return out


def cs_rank(x: pd.Series) -> pd.Series:
    """截面分位数映射到 [-1, 1]。无异常值风险，分布天然均匀。"""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    v = x.dropna()
    if len(v) < args.min_cs_size:
        return out
    out.loc[v.index] = v.rank(pct=True) * 2.0 - 1.0
    return out


norm_fn = cs_zscore if args.norm == "zscore" else cs_rank
print(f"\nCross-sectional normalization: {args.norm} "
      f"(mad_k={args.mad_k}, min_cs_size={args.min_cs_size}) ...")
df = df.groupby(level="datetime").transform(norm_fn)
df = df.replace([np.inf, -np.inf], np.nan)

# NaN 保留，不在此处 fillna。训练侧应在**标准化之后**填 0（此时 0 = 截面中性）。


# ══════════════════════════════════════════════════════════════════════
# 7. 保存 + 质检报告
# ══════════════════════════════════════════════════════════════════════
df.to_parquet(OUT_PATH)
print(f"\nSaved -> {OUT_PATH}")
print(f"  shape  : {df.shape}")
print(f"  dates  : {df.index.get_level_values('datetime').min()} ~ "
      f"{df.index.get_level_values('datetime').max()}")
print(f"  stocks : {df.index.get_level_values('instrument').nunique()}")

# 逐列质检：NaN 率 / 截面 std（应≈1）/ 时序自相关（过高说明因子近乎常数）
print("\nPer-factor diagnostics (sorted by NaN rate):")
nan_rate = df.isna().mean()
cs_std   = df.groupby(level="datetime").std().median()
sample_inst = df.index.get_level_values("instrument").unique()[:300]
autoc = df[df.index.get_level_values("instrument").isin(sample_inst)] \
          .groupby(level="instrument").apply(lambda g: g.apply(lambda s: s.autocorr(1))) \
          .median()

diag = pd.DataFrame({"nan_rate": nan_rate, "cs_std": cs_std, "autocorr_1d": autoc})
diag = diag.sort_values("nan_rate", ascending=False)
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(diag.round(4))

bad = diag.index[diag["nan_rate"] > args.max_nan_rate].tolist()
if bad:
    print(f"\n[WARN] NaN 率 > {args.max_nan_rate:.0%} 的因子（建议加入 SKIP_FACTORS）：")
    for b in bad:
        print(f"  {b}: {diag.loc[b, 'nan_rate']:.2%}")

if args.norm == "zscore":
    off = diag.index[(diag["cs_std"] < 0.8) | (diag["cs_std"] > 1.2)].tolist()
    if off:
        print("\n[WARN] 截面 std 明显偏离 1（可能仍受极值影响，考虑调小 --mad_k 或改用 --norm rank）：")
        for o in off:
            print(f"  {o}: cs_std={diag.loc[o, 'cs_std']:.3f}")

hi = diag.index[diag["autocorr_1d"] > 0.98].tolist()
if hi:
    print("\n[NOTE] 日间自相关 > 0.98 的因子（近乎常数，embedding 后 token 范数会显著偏大，"
          "是 attention 塌缩的常见来源）：")
    for h in hi:
        print(f"  {h}: autocorr={diag.loc[h, 'autocorr_1d']:.4f}")

print("\nDone.")
