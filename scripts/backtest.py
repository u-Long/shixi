"""
回测框架 —— TopKDrop 每日调仓，双 drop 参数对比

时点验证（已确认无未来函数）:
  特征窗口最后一根 bar = 决策日 T（T 日收盘后已知）
  信号在 T 日收盘后计算，T+1 日开盘买入执行
  label  = ret_5d_open[T] = log(open_{T+6}/open_{T+1})，T+1 开盘买入 ✓
  结算    = ret_1d_open[T] = log(open_{T+2}/open_{T+1})，同一买入时点 ✓

结算逻辑（对齐 run_backtest.py）:
  T 日：用 T 日 pred → topkdrop → new
        gross = mean(o2o[T] of new)   ← T+1 开盘买入 new，T+2 开盘卖出
        cost  = 换仓成本（new vs current）
        current = new

两组对比：
  baseline — 固定 topk=30 drop=3（每次都跑）
  custom   — 由 --topk / --n_drop 指定（默认同 baseline，自动跳过重复运行）

可买过滤（buyable）: 涨停 / 停牌 / ST 当日不可新买入
成本：买入 12bps，卖出 17bps，名义资金 1000 万，单笔最低 5 元

用法:
  nohup python scripts/backtest.py \
    --ckpt checkpoints/my_run/best.pt \
    --cache_dir data/cache/cache_fea2_hs300_ret5do \
    --topk 30 --n_drop 6 \
    --out_dir backtest_results/my_run \
    > logs/backtest_my_run.log 2>&1 &
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

from data_provider.stock_dataset import StockDataset, DayBatchSampler
from model.iTransformer_stock import Model

# ── 参数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        default="checkpoints/stock/best.pt")
parser.add_argument("--cache_dir",   default="data/cache/cache_fea2_hs300_ret5do")
parser.add_argument("--label_lib",   default="data/feature_lib/label_lib.parquet")
parser.add_argument("--seq_len",     type=int,   default=30)
parser.add_argument("--horizon",     type=int,   default=5)
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio",   type=float, default=0.15)
# 模型结构
parser.add_argument("--d_model",     type=int,   default=256)
parser.add_argument("--n_heads",     type=int,   default=4)
parser.add_argument("--e_layers",    type=int,   default=2)
parser.add_argument("--d_ff",        type=int,   default=512)
parser.add_argument("--dropout",     type=float, default=0.0)
parser.add_argument("--mlp_hidden",  type=int,   default=64)
parser.add_argument("--class_strategy", default="mean")
parser.add_argument("--embed",       default="fixed")
parser.add_argument("--freq",        default="b")
parser.add_argument("--factor",      type=int,   default=1)
parser.add_argument("--activation",  default="gelu")
parser.add_argument("--output_attention", action="store_true")
# 执行参数
parser.add_argument("--topk",        type=int,   default=None,
                    help="custom 组 topk，不传则只跑 baseline（k30d3）")
parser.add_argument("--n_drop",      type=int,   default=None,
                    help="custom 组 drop 数，不传则只跑 baseline（k30d3）")
parser.add_argument("--capital",     type=float, default=10_000_000.0)
parser.add_argument("--buy_cost",    type=float, default=0.0012)
parser.add_argument("--sell_cost",   type=float, default=0.0017)
parser.add_argument("--min_cost",    type=float, default=5.0)
parser.add_argument("--slippage_bps", type=float, default=0.0,
                    help="单边滑点（bps），叠加在 buy_cost/sell_cost 之上。"
                         "默认 0 只为保持与历史结果可比；以 T+1 开盘价全额成交本身就很乐观，"
                         "小盘股实盘单边 20~50bps 是常态，评估真实收益时务必设非 0 值")
parser.add_argument("--weight_mode", default="drift", choices=["drift", "equal"],
                    help="drift（默认，正确口径）：持仓权重随收益自然漂移，只对实际换仓收费；"
                         "equal（旧口径）：每日强制拉回等权，相当于免费日频再平衡，会高估收益")
parser.add_argument("--n_groups",    type=int,   default=5)
parser.add_argument("--benchmark",   default="/root/dmd/BaoStock/Index/sh.000001.csv")
parser.add_argument("--out_dir",     default="backtest_results")
parser.add_argument("--num_workers", type=int,   default=4)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 从 checkpoint 恢复训练参数 ────────────────────────────────────────────
raw_ckpt = torch.load(args.ckpt, map_location="cpu")
if isinstance(raw_ckpt, dict) and "args" in raw_ckpt:
    saved = raw_ckpt["args"]
    for k in ["seq_len", "horizon",
              "train_start", "train_end", "val_start", "val_end",
              "test_start", "test_end", "train_ratio", "val_ratio",
              "d_model", "n_heads", "e_layers", "d_ff", "mlp_hidden",
              "class_strategy", "embed", "freq", "factor", "activation"]:
        if k in saved:
            setattr(args, k, saved[k])
    print(f"[backtest] Restored from ckpt: seq_len={args.seq_len}  "
          f"horizon={args.horizon}  test={args.test_start}~{args.test_end}")

HORIZON = args.horizon
CAP     = args.capital
SLIP    = args.slippage_bps / 1e4
BUY_C   = args.buy_cost  + SLIP
SELL_C  = args.sell_cost + SLIP
MIN_C   = args.min_cost
WMODE   = args.weight_mode

print(f"成本：买 {BUY_C*1e4:.1f}bps / 卖 {SELL_C*1e4:.1f}bps"
      f"（含滑点 {args.slippage_bps:.1f}bps 单边）  权重口径：{WMODE}")
if args.slippage_bps == 0:
    print("  [WARN] 滑点为 0，且假设 T+1 开盘价全额成交，净收益偏乐观。"
          "评估真实可实现收益请加 --slippage_bps 20 之类的设定")
if WMODE == "equal":
    print("  [WARN] weight_mode=equal 等于每日免费再平衡回等权，"
          "对波动大的小盘股会白拿再平衡收益，仅用于和旧结果对比")

# baseline 永远是 topk=30 drop=3；custom 仅在 --topk/--n_drop 任一传入时启用
BASELINE_TOPK = 30
BASELINE_DROP = 3
run_custom    = args.topk is not None or args.n_drop is not None
CUSTOM_TOPK   = args.topk  if args.topk   is not None else BASELINE_TOPK
CUSTOM_DROP   = args.n_drop if args.n_drop is not None else BASELINE_DROP
same_as_baseline = (CUSTOM_TOPK == BASELINE_TOPK and CUSTOM_DROP == BASELINE_DROP)
print(f"Baseline: TopK={BASELINE_TOPK}  Drop={BASELINE_DROP}")
if run_custom:
    print(f"Custom:   TopK={CUSTOM_TOPK}  Drop={CUSTOM_DROP}")
else:
    print("Custom: 未指定，只跑 baseline")

# ── 加载数据集 ────────────────────────────────────────────────────────────
def raw_collate(batch):
    """回测需要 (di, si) 来把预测值对回「日期, 股票」，所以一并返回。"""
    xs, ys, dis, sis = zip(*batch)
    return (torch.stack(xs), torch.stack(ys).unsqueeze(-1),
            torch.stack(dis), torch.stack(sis))

test_ds = StockDataset(
    cache_dir   = args.cache_dir,
    seq_len     = args.seq_len,
    horizon     = args.horizon,
    flag        = "test",
    train_start = args.train_start, train_end = args.train_end,
    val_start   = args.val_start,   val_end   = args.val_end,
    test_start  = args.test_start,  test_end  = args.test_end,
    train_ratio = args.train_ratio, val_ratio = args.val_ratio,
)
test_loader = DataLoader(
    test_ds, batch_sampler=DayBatchSampler(test_ds, shuffle=False),
    collate_fn=raw_collate, num_workers=args.num_workers,
)

# ── 加载模型 & 推理 ───────────────────────────────────────────────────────
args.enc_in = test_ds.n_features
model = Model(args).to(device)
model.load_state_dict(raw_ckpt["state_dict"] if "state_dict" in raw_ckpt else raw_ckpt)
model.eval()
print(f"Loaded: {args.ckpt}  features={args.enc_in}")

print("Running inference...")
# di/si 直接从 DataLoader 一路带出来，预测值与「日期, 股票」严格同源。
# 旧写法是按位置 zip(all_pred, test_ds.samples)，只有在
# DayBatchSampler(shuffle=False) 且日内顺序恰好等于 samples 顺序时才对，
# 一旦有人改了 sampler 就会静默错配，而长度断言抓不到。
all_pred, all_label, all_di, all_si = [], [], [], []
with torch.no_grad():
    for x, y, di, si in test_loader:
        pred = model(x.to(device)).cpu().numpy().squeeze()
        y_np = y.numpy().squeeze()
        all_pred.extend(np.atleast_1d(pred).tolist())
        all_label.extend(np.atleast_1d(y_np).tolist())
        all_di.extend(di.numpy().tolist())
        all_si.extend(si.numpy().tolist())

samples = test_ds.samples
dates   = test_ds.dates
stocks  = test_ds.stocks
assert len(all_pred) == len(samples), \
    f"推理条数 {len(all_pred)} != 样本数 {len(samples)}"
assert set(zip(all_di, all_si)) == set((d, s) for d, s in samples), \
    "推理得到的 (di, si) 集合与 dataset.samples 不一致，DataLoader 可能漏样本或重复"

# 注意：samples 由 StockDataset 构建，过滤条件包含"label 非 NaN"。
# 这对训练/评估是合理的，但回测候选池若只含 label 有效的股票，
# 会因为停牌/退市前的股票被静默剔除而产生前视选择偏差（见 review_new.md L1）。
df = pd.DataFrame({
    "date":  [dates[di]  for di in all_di],
    "stock": [stocks[si] for si in all_si],
    "pred":  all_pred,
    "label": all_label,
})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "stock"]).reset_index(drop=True)
print(f"Inference done: {len(df)} records, {df.date.nunique()} dates")

# ── 加载 ret_1d_open（T+1 开盘买入、T+2 开盘卖出，日度结算）────────────
print("Loading ret_1d_open from label_lib...")
lib = pd.read_parquet(args.label_lib, columns=["ret_1d_open"])
lib = lib.reset_index()
lib.columns = ["date", "stock", "o2o"]
lib["date"] = pd.to_datetime(lib["date"])
df = df.merge(lib, on=["date", "stock"], how="left")
n_miss = df["o2o"].isna().sum()
if n_miss:
    print(f"  [WARN] {n_miss}/{len(df)} 条记录无 o2o，填 0")
df["o2o"] = df["o2o"].fillna(0.0)
df.to_parquet(os.path.join(args.out_dir, "pred_detail.parquet"))

print(f"  o2o: mean={df['o2o'].mean():.5f}  std={df['o2o'].std():.4f}  "
      f"NaN={n_miss/len(df):.2%}")

# ── 构建 buyable / sellable ───────────────────────────────────────────────
# buyable[T]  = T+1 开盘能否买入（涨停次日仍可能封板，保守处理：当日涨停则不买）
# sellable[T] = T+1 开盘能否卖出（跌停/停牌卖不掉，只过滤买入不过滤卖出是不对称的
#               乐观假设——A股跌停无法出货很常见，尤其小市值）
print("Building buyable / sellable flags from panel...")
panel = pd.read_parquet(
    "/root/dmd/BaoStock/panel.parquet",
    columns=["open", "high", "low", "pre_close", "trade_status", "is_st"],
)
panel = panel.reset_index()
panel.columns = ["date", "stock", "open", "high", "low",
                 "pre_close", "trade_status", "is_st"]
panel["date"] = pd.to_datetime(panel["date"])
_pc = panel["pre_close"].clip(lower=1e-8)
panel["up_limit"]   = (panel["high"] / _pc - 1) >=  0.099
panel["down_limit"] = (panel["low"]  / _pc - 1) <= -0.099
panel["buyable"]  = (
    (panel["trade_status"] == 1) &
    (~panel["up_limit"]) &
    (panel["is_st"] == 0)
).astype(bool)
panel["sellable"] = (
    (panel["trade_status"] == 1) &
    (~panel["down_limit"])
).astype(bool)
buyable_map, sellable_map = {}, {}
for row in panel[["date", "stock", "buyable", "sellable"]].itertuples(index=False):
    buyable_map[(row.date, row.stock)]  = row.buyable
    sellable_map[(row.date, row.stock)] = row.sellable
print(f"  buyable/sellable loaded: {len(buyable_map)} records  "
      f"（可买 {panel['buyable'].mean():.1%}，可卖 {panel['sellable'].mean():.1%}）")

# ── 基准 ──────────────────────────────────────────────────────────────────
bench_map: dict = {}
if args.benchmark and Path(args.benchmark).exists():
    idx = pd.read_csv(args.benchmark)
    idx.columns = [c.strip().lower() for c in idx.columns]
    idx["date"] = pd.to_datetime(idx["date"])
    idx["open"] = pd.to_numeric(idx["open"], errors="coerce")
    idx = idx.sort_values("date").reset_index(drop=True)
    # 基准 o2o：同样是 T+1 开盘买入、T+2 开盘卖出
    idx["bench_o2o"] = idx["open"].shift(-2) / idx["open"].shift(-1) - 1.0
    bench_map = dict(zip(idx["date"], idx["bench_o2o"]))
    print(f"Benchmark loaded: {args.benchmark}")

all_dates_sorted = sorted(df["date"].unique())
df_by_date = {d: g.set_index("stock") for d, g in df.groupby("date")}


# ══════════════════════════════════════════════════════════════════════════
# IC / RankIC（模型 label，因子质量评估）
# ══════════════════════════════════════════════════════════════════════════
ic_rows = []
for date, grp in df.groupby("date"):
    if len(grp) < 10:
        continue
    ic_v,  _ = pearsonr(grp["pred"],  grp["label"])
    ric_v, _ = spearmanr(grp["pred"], grp["label"])
    if not np.isnan(ic_v + ric_v):
        ic_rows.append({"date": date, "ic": float(ic_v), "rankic": float(ric_v)})

ic_df = pd.DataFrame(ic_rows).set_index("date")
mean_ic     = ic_df["ic"].mean()
mean_rankic = ic_df["rankic"].mean()
icir        = mean_ic     / (ic_df["ic"].std()     + 1e-8)
rankicir    = mean_rankic / (ic_df["rankic"].std() + 1e-8)
ic_pos      = (ic_df["rankic"] > 0).mean()

# ── 重叠修正 ─────────────────────────────────────────────────────────────
# ret_Xd_open[T] = log(open[T+horizon+1]/open[T+1])，实际跨 horizon+1 个交易日，
# 相邻两天的 label 共享 horizon/(horizon+1) 的行情区间 → 日度 IC 序列高度自相关。
# 直接用 mean/std 当显著性判据会高估：方差膨胀因子约等于 span，
# 即有效独立样本 N_eff ≈ N/span，t 统计量要按 √N_eff 而非 √N 折算。
LABEL_SPAN = HORIZON + 1


def icir_nooverlap(s: pd.Series, span: int) -> float:
    """不重叠采样的 ICIR。对 span 个相位分别采样再取均值——
    只取单一相位（s.iloc[::span]）的结果对起点很敏感，换个起点就变。"""
    vals = []
    for phase in range(span):
        sub = s.iloc[phase::span].dropna()
        if len(sub) > 2 and sub.std(ddof=1) > 1e-12:
            vals.append(sub.mean() / sub.std(ddof=1))
    return float(np.mean(vals)) if vals else float("nan")


def newey_west_t(s: pd.Series, lag: int) -> float:
    """IC 均值的 Newey-West t 统计量（Bartlett 核），直接处理重叠导致的自相关。
    比不重叠采样更充分利用样本，是判断信号是否显著的首选口径。"""
    x = s.dropna().values
    n = len(x)
    if n < lag + 3:
        return float("nan")
    e = x - x.mean()
    var = float(e @ e) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(e[k:] @ e[:-k]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


icir_nl     = icir_nooverlap(ic_df["ic"],     LABEL_SPAN)
rankicir_nl = icir_nooverlap(ic_df["rankic"], LABEL_SPAN)
t_ic        = newey_west_t(ic_df["ic"],     LABEL_SPAN - 1)
t_rankic    = newey_west_t(ic_df["rankic"], LABEL_SPAN - 1)
n_eff       = len(ic_df) / LABEL_SPAN
ic_df.to_csv(os.path.join(args.out_dir, "ic_series.csv"))

print(f"\n== IC 统计（因子层，基于模型 label）==")
print(f"  IC    ={mean_ic:.4f}   RankIC={mean_rankic:.4f}   RankIC>0={ic_pos:.2%}")
print(f"  ICIR    ={icir:.4f}（含重叠，偏乐观）  不重叠={icir_nl:.4f}      t_NW={t_ic:.2f}")
print(f"  RankICIR={rankicir:.4f}（含重叠，偏乐观）  不重叠={rankicir_nl:.4f}  t_NW={t_rankic:.2f}")
print(f"  N={len(ic_df)} 天，label 跨 {LABEL_SPAN} 天重叠 → 有效独立样本≈{n_eff:.0f}")
print(f"  显著性请看 t_NW（|t|>2 才算有信号），不要用含重叠的 ICIR")


# ══════════════════════════════════════════════════════════════════════════
# 核心执行引擎
# ══════════════════════════════════════════════════════════════════════════
def cost_from_weights(w_pre: dict, w_new: dict) -> float:
    """按实际成交的权重计费，返回占组合净值的比例。

    旧实现按「换仓只数 × 单档名义金额」计费，隐含每档权重恒等于 1/topk，
    与 drift 口径下的真实权重不符；改成按 |w_new - w_pre| 逐只计费后，
    equal 模式下被强制拉回等权的那部分再平衡交易也会如实收费。
    """
    total = 0.0
    for s in set(w_pre) | set(w_new):
        d = w_new.get(s, 0.0) - w_pre.get(s, 0.0)
        notional = abs(d) * CAP
        if notional < 1.0:        # 忽略权重浮尘，否则每只都要吃一次 min_cost
            continue
        rate = BUY_C if d > 0 else SELL_C
        total += max(notional * rate, MIN_C)
    return total / CAP


def topkdrop(score: pd.Series, current: list, topk: int, n_drop: int,
             buyable: pd.Series | None = None,
             sellable: pd.Series | None = None) -> list:
    """
    参照参考包 choose_topkdrop（keep 模式）。
    buyable:  index=stock，False 表示涨停/停牌/ST，不可新买入（已持仓可继续持有）。
    sellable: index=stock，False 表示跌停/停牌，卖不掉，被迫继续持有。
    """
    pred = score.dropna().sort_values(ascending=False)
    if pred.empty:
        return list(current)
    if not current:
        # 首日：只从可买股中选
        if buyable is not None:
            pred = pred[pred.index.map(lambda c: bool(buyable.get(c, True)))]
        return list(pred.head(topk).index)
    cur_set = set(current)
    in_pred = [c for c in current if c in pred.index]
    not_in  = [c for c in current if c not in pred.index]   # 行情缺失保留
    last    = pred.reindex(in_pred).sort_values(ascending=False).index.tolist()
    n_new   = max(0, n_drop + topk - len(not_in) - len(last))
    # 候选新股：只选可买股
    cand_pool = pred.loc[~pred.index.isin(cur_set)]
    if buyable is not None:
        cand_pool = cand_pool[cand_pool.index.map(lambda c: bool(buyable.get(c, True)))]
    cands  = cand_pool.head(n_new).index.tolist()
    pool   = pred.reindex(last + cands).sort_values(ascending=False).index.tolist()
    sells  = set(pool[-n_drop:]) if n_drop > 0 else set()
    # 跌停/停牌的持仓卖不掉，从卖出名单里剔除，被迫留在组合里
    if sellable is not None:
        sells = {c for c in sells if bool(sellable.get(c, True))}
    kept   = not_in + [c for c in last if c not in sells]
    buys   = [c for c in cand_pool.index if c not in set(kept) and c not in cur_set]
    result = (kept + buys)[:topk]
    if len(result) < topk:
        for c in pred.index:
            if c not in set(result):
                if c in cur_set or buyable is None or bool(buyable.get(c, True)):
                    result.append(c)
            if len(result) >= topk:
                break
    return result[:topk]


def simulate(topk: int, n_drop: int) -> pd.DataFrame:
    """
    每日调仓 TopKDrop：
      T 日：pred → topkdrop → new（buyable 挡涨停/停牌/ST，sellable 挡跌停/停牌）
            gross = Σ w_new[i] × o2o[T][i]   ← new 在 T+1 开盘买入，T+2 开盘卖出
            cost  = 按 |w_new − w_pre| 逐只计费
            收盘后权重随收益漂移，作为下一日的 w_pre

    权重口径（--weight_mode）：
      drift — 保留仓位的权重随收益自然漂移，卖出释放的权重等分给新买入；
              若当日没有新买入，释放的权重按比例回到保留仓位（等比缩放，
              不改变相对权重，无需交易，不计费）。
      equal — 每日把组合强制拉回等权。这是旧口径，隐含每天免费再平衡一次，
              对波动大的小盘股会白拿一份再平衡收益；现在这部分交易会如实计费。
    """
    w: dict = {}          # stock -> 权重，有持仓时 sum(w) == 1
    nav = 1.0
    rows = []
    for date in all_dates_sorted:
        day_df   = df_by_date[date]
        score    = day_df["pred"]
        date_ts  = pd.Timestamp(date)
        o2o      = day_df["o2o"]

        buyable_day = pd.Series({
            stock: buyable_map.get((date_ts, stock), True)
            for stock in day_df.index
        })
        # 卖出约束只对当前持仓有意义，持仓可能已不在当日截面里，单独取
        current = list(w.keys())
        sellable_day = pd.Series({
            stock: sellable_map.get((date_ts, stock), True)
            for stock in set(day_df.index) | set(current)
        })

        new = topkdrop(score, current, topk, n_drop,
                       buyable=buyable_day, sellable=sellable_day)

        # ── 目标权重 ──────────────────────────────────────────────────────
        w_pre = dict(w)
        if not new:
            w_new = {}
        elif WMODE == "equal":
            w_new = {s: 1.0 / len(new) for s in new}
        else:
            kept = [s for s in new if s in w_pre]
            buys = [s for s in new if s not in w_pre]
            freed = 1.0 - sum(w_pre[s] for s in kept) if w_pre else 1.0
            w_new = {s: w_pre[s] for s in kept}
            if buys:
                per = max(freed, 0.0) / len(buys)
                for s in buys:
                    w_new[s] = per
            tot = sum(w_new.values())
            if tot > 1e-12:
                w_new = {s: v / tot for s, v in w_new.items()}

        gross = sum(wt * float(o2o.get(s, 0.0)) for s, wt in w_new.items())
        cost  = cost_from_weights(w_pre, w_new)
        net   = gross - cost
        nav  *= 1.0 + net

        cur_set, new_set = set(w_pre), set(new)
        turnover_w = sum(abs(w_new.get(s, 0.0) - w_pre.get(s, 0.0))
                         for s in cur_set | new_set)

        bench = float(bench_map.get(date_ts, float("nan")))
        rows.append({
            "date":        date,
            "gross":       gross,
            "net":         net,
            "nav":         nav,
            "n_buy":       len(new_set - cur_set),
            "n_sell":      len(cur_set - new_set),
            "turnover":    turnover_w,          # 权重口径的双边换手
            "n_hold":      len(new_set),
            "bench_o2o":   bench,
        })

        # ── 权重漂移到下一日 ──────────────────────────────────────────────
        w = {s: wt * (1.0 + float(o2o.get(s, 0.0))) for s, wt in w_new.items()}
        tot = sum(w.values())
        if tot > 1e-12:
            w = {s: v / tot for s, v in w.items()}
        else:
            w = {}

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 统计函数（对齐参考包 summary_metrics）
# ══════════════════════════════════════════════════════════════════════════
def summary_stats(sim_df: pd.DataFrame, label: str = "", topk: int = BASELINE_TOPK) -> dict:
    r  = sim_df["net"].fillna(0.0)
    g  = sim_df["gross"].fillna(0.0)
    b  = sim_df["bench_o2o"].fillna(0.0)
    ex = r - b
    n  = len(r)
    nav = (1.0 + r).cumprod()

    cum_ret = float(nav.iloc[-1] - 1)
    ann_ret = float((1 + cum_ret) ** (252.0 / n) - 1) if n and (1 + cum_ret) > 0 else float("nan")
    std_r   = float(r.std(ddof=1))           if n > 1        else float("nan")
    std_ds  = float(r[r < 0].std(ddof=1))    if (r < 0).any() else float("nan")
    std_ex  = float(ex.std(ddof=1))          if n > 1        else float("nan")
    mdd     = float((nav / nav.cummax() - 1).min())
    sharpe  = float(r.mean() / std_r  * math.sqrt(252)) if std_r  and std_r  > 1e-15 else float("nan")
    sortino = float(r.mean() / std_ds * math.sqrt(252)) if std_ds and std_ds > 1e-15 else float("nan")
    info_r  = float(ex.mean() / std_ex * math.sqrt(252)) if std_ex and std_ex > 1e-15 else float("nan")

    monthly = r.copy()
    monthly.index = sim_df["date"]
    mret    = monthly.groupby(monthly.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    win_rate = float((mret > 0).mean())

    avg_n_sell = float(sim_df["n_sell"].mean())
    hold_days  = topk / avg_n_sell if avg_n_sell > 1e-9 else float("nan")
    ann_to     = float(sim_df["turnover"].sum() / n * 252)
    ann_cost   = float((g - r).mean() * 252 * 100)

    # 去最差月
    worst_m  = mret.idxmin()
    r_ex     = r.copy(); r_ex.index = sim_df["date"]
    r_normal = r_ex[r_ex.index.to_period("M") != worst_m]
    nav_n    = (1.0 + r_normal).cumprod()
    cum_n    = float(nav_n.iloc[-1] - 1) if len(nav_n) else float("nan")
    ann_n    = float((1 + cum_n) ** (252.0 / len(r_normal)) - 1) if len(r_normal) and (1 + cum_n) > 0 else float("nan")
    std_n    = float(r_normal.std(ddof=1)) if len(r_normal) > 1 else float("nan")
    shr_n    = float(r_normal.mean() / std_n * math.sqrt(252)) if std_n and std_n > 1e-15 else float("nan")
    mdd_n    = float((nav_n / nav_n.cummax() - 1).min()) if len(nav_n) else float("nan")

    return {
        "label":      label,
        "n_days":     n,
        "cum_ret":    cum_ret,
        "ann_ret":    ann_ret,
        "ann_vol":    std_r * math.sqrt(252) if std_r and not math.isnan(std_r) else float("nan"),
        "sharpe":     sharpe,
        "sortino":    sortino,
        "info_ratio": info_r,
        "mdd":        mdd,
        "win_rate":   win_rate,
        "hit":        float((r > 0).mean()),
        "ann_to":       ann_to,
        "hold_days":    hold_days,
        "cost_ppt":     ann_cost,
        "monthly":      mret,
        "worst_month":  str(worst_m),
        "ex_worst_ann": ann_n,
        "ex_worst_sh":  shr_n,
        "ex_worst_mdd": mdd_n,
    }


def print_stats(s: dict):
    print(f"  {s['label']:<24s}  "
          f"年化={s['ann_ret']:>+7.2%}  Sharpe={s['sharpe']:>5.2f}  "
          f"Sortino={s['sortino']:>5.2f}  InfoR={s['info_ratio']:>5.2f}  "
          f"MDD={s['mdd']:>+7.2%}  累计={s['cum_ret']:>+7.2%}  "
          f"月胜率={s['win_rate']:>5.1%}  "
          f"持仓≈{s['hold_days']:>4.1f}日  换手={s['ann_to']:.1f}x  "
          f"成本={s['cost_ppt']:.2f}ppt  "
          f"[去最差月({s['worst_month']}) 年化={s['ex_worst_ann']:>+7.2%}  "
          f"Sharpe={s['ex_worst_sh']:>5.2f}  MDD={s['ex_worst_mdd']:>+7.2%}]")


# ══════════════════════════════════════════════════════════════════════════
# 运行两组：baseline(30/3) 永远跑；custom 由 --topk/--n_drop 指定
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRunning baseline: TopK={BASELINE_TOPK}  Drop={BASELINE_DROP} ...")
df_b = simulate(BASELINE_TOPK, BASELINE_DROP)
df_b.to_csv(os.path.join(args.out_dir, f"exec_baseline_k{BASELINE_TOPK}_d{BASELINE_DROP}.csv"), index=False)

if run_custom and not same_as_baseline:
    print(f"Running custom:   TopK={CUSTOM_TOPK}  Drop={CUSTOM_DROP} ...")
    df_h = simulate(CUSTOM_TOPK, CUSTOM_DROP)
    df_h.to_csv(os.path.join(args.out_dir, f"exec_custom_k{CUSTOM_TOPK}_d{CUSTOM_DROP}.csv"), index=False)
else:
    df_h = None

LABEL_B = f"baseline k{BASELINE_TOPK}d{BASELINE_DROP}"
LABEL_H = f"custom   k{CUSTOM_TOPK}d{CUSTOM_DROP}"

# ── 窗口统计 ─────────────────────────────────────────────────────────────
windows = {
    "full":  (args.test_start, args.test_end),
    "2025":  ("2025-01-01",    "2025-12-31"),
    "2026":  ("2026-01-01",    args.test_end),
}

all_stats = []
print(f"\n== 执行层回测（买{BUY_C*1e4:.0f}bps/卖{SELL_C*1e4:.0f}bps）==")
for wname, (ws, we) in windows.items():
    b_sub = df_b[df_b["date"].between(ws, we)].copy()
    if b_sub.empty:
        continue
    print(f"\n  [{wname}]  {ws} ~ {we}")
    s = summary_stats(b_sub, label=f"{LABEL_B} [{wname}]", topk=BASELINE_TOPK)
    print_stats(s)
    all_stats.append({k: v for k, v in s.items() if k != "monthly"})
    if df_h is not None:
        h_sub = df_h[df_h["date"].between(ws, we)].copy()
        if not h_sub.empty:
            s = summary_stats(h_sub, label=f"{LABEL_H} [{wname}]", topk=CUSTOM_TOPK)
            print_stats(s)
            all_stats.append({k: v for k, v in s.items() if k != "monthly"})

pd.DataFrame(all_stats).to_csv(os.path.join(args.out_dir, "exec_metrics.csv"), index=False)

# ── 月度收益对比 ──────────────────────────────────────────────────────────
sb = summary_stats(df_b, LABEL_B, topk=BASELINE_TOPK)
sh = summary_stats(df_h, LABEL_H, topk=CUSTOM_TOPK) if df_h is not None else None
if sh is not None:
    mr = pd.concat([sb["monthly"].rename(LABEL_B),
                    sh["monthly"].rename(LABEL_H)], axis=1)
    print(f"\n月度净收益对比:")
    print(f"  {'月份':<10}  {LABEL_B:>20}  {LABEL_H:>20}")
    for m, row in mr.iterrows():
        print(f"  {str(m):<10}  {row[LABEL_B]:>+20.2%}  {row[LABEL_H]:>+20.2%}")
else:
    print(f"\n月度净收益（baseline）:")
    print(f"  {'月份':<10}  {LABEL_B:>20}")
    for m, v in sb["monthly"].items():
        print(f"  {str(m):<10}  {v:>+20.2%}")


# ══════════════════════════════════════════════════════════════════════════
# 分层回测（因子层，o2o 结算，无成本）
# ══════════════════════════════════════════════════════════════════════════
group_rets = {g: [] for g in range(args.n_groups)}
ls_rows = []
for date, grp in df.groupby("date"):
    if len(grp) < args.n_groups * 2:
        continue
    grp = grp.sort_values("pred", ascending=False).reset_index(drop=True)
    n   = len(grp)
    for g in range(args.n_groups):
        lo = int(g * n / args.n_groups)
        hi = int((g + 1) * n / args.n_groups)
        group_rets[g].append(grp.iloc[lo:hi]["o2o"].mean())
    nl = min(BASELINE_TOPK, n // 4)
    ls_rows.append({
        "date":      date,
        "long_ret":  grp.iloc[:nl]["o2o"].mean(),
        "short_ret": grp.iloc[-nl:]["o2o"].mean(),
    })

ls_df = pd.DataFrame(ls_rows).set_index("date")
ls_df["ls_ret"] = ls_df["long_ret"] - ls_df["short_ret"]
ls_df.to_csv(os.path.join(args.out_dir, "ls_returns.csv"))

print(f"\n== 分层收益（o2o 年化，{args.n_groups} 组）==")
for g in range(args.n_groups):
    ann = np.mean(group_rets[g]) * 252
    tag = "多头" if g == 0 else ("空头" if g == args.n_groups - 1 else f"G{g+1}")
    print(f"  Group {g+1} ({tag}): 年化={ann:.2%}  日均 o2o={np.mean(group_rets[g]):.5f}")


# ══════════════════════════════════════════════════════════════════════════
# 画图（6 子图）
# ══════════════════════════════════════════════════════════════════════════
for _fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

bench_label = Path(args.benchmark).stem.upper() if args.benchmark else "基准"

fig, axes = plt.subplots(5, 1, figsize=(14, 20))
title = f"iTransformer Backtest  {LABEL_B}" + (f" vs {LABEL_H}" if df_h is not None else "")
fig.suptitle(title, fontsize=13)

# 基准净值
bench_valid = pd.Series(bench_map).sort_index().dropna()
bench_valid = bench_valid[bench_valid.index >= pd.Timestamp(args.test_start)]
cum_bench   = (1 + bench_valid).cumprod() if not bench_valid.empty else None

# 1. 净值对比
ax = axes[0]
ax.plot(df_b["date"], df_b["nav"],
        label=f"{LABEL_B}  AnnRet={sb['ann_ret']:+.1%}  Sharpe={sb['sharpe']:.2f}",
        color="steelblue", lw=1.8)
if df_h is not None:
    ax.plot(df_h["date"], df_h["nav"],
            label=f"{LABEL_H}  AnnRet={sh['ann_ret']:+.1%}  Sharpe={sh['sharpe']:.2f}",
            color="darkorange", lw=1.8, ls="--")
if cum_bench is not None:
    ax.plot(cum_bench.index, cum_bench.values, label=bench_label,
            color="gray", lw=1.0, ls=":")
ax.axhline(1, color="gray", lw=0.8, ls=":")
ax.set_title("净值（净，含成本）")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylabel("NAV")

# 2. 超额净值
ax = axes[1]
if cum_bench is not None:
    cb_b = cum_bench.reindex(df_b["date"].values).ffill().values
    exc_b = df_b["nav"].values / np.where(np.isnan(cb_b) | (cb_b == 0), 1.0, cb_b)
    ax.plot(df_b["date"], exc_b,
            label=f"{LABEL_B} 超额  InfoR={sb['info_ratio']:.2f}",
            color="steelblue", lw=1.5)
    if df_h is not None:
        cb_h = cum_bench.reindex(df_h["date"].values).ffill().values
        exc_h = df_h["nav"].values / np.where(np.isnan(cb_h) | (cb_h == 0), 1.0, cb_h)
        ax.plot(df_h["date"], exc_h,
                label=f"{LABEL_H} 超额  InfoR={sh['info_ratio']:.2f}",
                color="darkorange", lw=1.5, ls="--")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_title(f"超额净值（net / {bench_label}）")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
else:
    ax.set_visible(False)

# 3. 因子层多空净值
ax = axes[2]
cum_ls   = (1 + ls_df["ls_ret"]).cumprod()
cum_long = (1 + ls_df["long_ret"]).cumprod()
ax.plot(cum_ls.index,   cum_ls.values,   label="Long-Short", color="darkred", lw=1.5)
ax.plot(cum_long.index, cum_long.values, label="Long Only",  color="olive",   lw=1.0, alpha=0.7)
ax.axhline(1, color="gray", lw=0.8, ls=":")
ax.set_title("因子层多空净值（o2o 结算，无成本）")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 4. 日度 RankIC
ax = axes[3]
ax.bar(ic_df.index, ic_df["rankic"],
       color=["green" if v > 0 else "red" for v in ic_df["rankic"]],
       alpha=0.5, width=1.0)
ax.axhline(0, color="black", lw=0.8)
ax.plot(ic_df.index, ic_df["rankic"].rolling(20).mean(),
        color="navy", lw=1.5, label="20d MA")
ax.set_title(f"日度 RankIC  mean={mean_rankic:.4f}  RankICIR={rankicir:.2f}  "
             f"IC={mean_ic:.4f}  ICIR={icir:.2f}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5. 分层年化收益
ax = axes[4]
group_anns = [np.mean(group_rets[g]) * 252 for g in range(args.n_groups)]
ax.bar([f"G{g+1}" for g in range(args.n_groups)], group_anns,
       color=["green" if v > 0 else "red" for v in group_anns], alpha=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"分层年化收益（{args.n_groups} 组，o2o 结算）")
ax.set_ylabel("Ann. Return"); ax.grid(True, alpha=0.3, axis="y")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

plt.tight_layout()
fig_path = os.path.join(args.out_dir, "backtest.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nPlot -> {fig_path}")

# ══════════════════════════════════════════════════════════════════════════
# 生成 summary.md
# ══════════════════════════════════════════════════════════════════════════
def _fmt(v, pct=True, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:+.{decimals}%}" if pct else f"{v:.{decimals}f}"

def _stats_row(s, window):
    return (f"| {window} | {_fmt(s['ann_ret'])} | {_fmt(s['sharpe'], pct=False)} | "
            f"{_fmt(s['sortino'], pct=False)} | {_fmt(s['info_ratio'], pct=False)} | "
            f"{_fmt(s['mdd'])} | {_fmt(s['cum_ret'])} | {_fmt(s['win_rate'])} |")

runs = [(sb, LABEL_B, BASELINE_TOPK, BASELINE_DROP)]
if df_h is not None:
    runs.append((sh, LABEL_H, CUSTOM_TOPK, CUSTOM_DROP))

md_lines = [
    f"# Backtest Summary",
    f"",
    f"**Checkpoint**: `{args.ckpt}`  ",
    f"**Cache**: `{args.cache_dir}`  ",
    f"**Test period**: {args.test_start} ~ {args.test_end}  ",
    f"**Benchmark**: `{Path(args.benchmark).name if args.benchmark else 'None'}`  ",
    f"**Cost**: buy {BUY_C*1e4:.0f}bps / sell {SELL_C*1e4:.0f}bps  ",
    f"",
    f"## Factor Quality",
    f"",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| IC | {mean_ic:.4f} |",
    f"| ICIR | {icir:.4f} |",
    f"| RankIC | {mean_rankic:.4f} |",
    f"| RankICIR | {rankicir:.4f} |",
    f"| RankIC > 0 | {ic_pos:.2%} |",
    f"",
]

for stat, label, topk, ndrop in runs:
    win_stats = {}
    df_src = df_b if label == LABEL_B else df_h
    for wname, (ws, we) in windows.items():
        sub = df_src[df_src["date"].between(ws, we)]
        if not sub.empty:
            win_stats[wname] = summary_stats(sub, topk=topk)

    md_lines += [
        f"## Strategy: {label}  (TopK={topk}, Drop={ndrop})",
        f"",
        f"年化换手 {stat['ann_to']:.1f}x  |  成本 {stat['cost_ppt']:.2f}ppt/年  |  "
        f"持仓≈{stat['hold_days']:.1f}日  |  去最差月({stat['worst_month']}) 年化{_fmt(stat['ex_worst_ann'])} Sharpe{stat['ex_worst_sh']:.2f}",
        f"",
        f"| 窗口 | 年化收益 | Sharpe | Sortino | InfoR | MDD | 累计 | 月胜率 |",
        f"|------|---------|--------|---------|-------|-----|------|--------|",
    ]
    for wname in ["2025", "2026", "full"]:
        if wname in win_stats:
            label_w = {"full": "全测试期", "2025": "2025", "2026": "2026"}[wname]
            md_lines.append(_stats_row(win_stats[wname], label_w))
    md_lines.append("")

md_lines += [
    f"## Factor Layer (no cost)",
    f"",
    f"| Group | Ann. Return |",
    f"|-------|-------------|",
]
for g in range(args.n_groups):
    ann = np.mean(group_rets[g]) * 252
    tag = "多头" if g == 0 else ("空头" if g == args.n_groups - 1 else f"G{g+1}")
    md_lines.append(f"| G{g+1} ({tag}) | {ann:+.2%} |")

md_lines += [
    f"",
    f"## Monthly Returns",
    f"",
]
if sh is not None:
    md_lines.append(f"| 月份 | {LABEL_B} | {LABEL_H} |")
    md_lines.append(f"|------|{'|'.join(['-'*len(LABEL_B), '-'*len(LABEL_H)])}|")
    for m, v in sb["monthly"].items():
        vh = sh["monthly"].get(m, float("nan"))
        md_lines.append(f"| {m} | {v:+.2%} | {vh:+.2%} |")
else:
    md_lines.append(f"| 月份 | {LABEL_B} |")
    md_lines.append(f"|------|------|")
    for m, v in sb["monthly"].items():
        md_lines.append(f"| {m} | {v:+.2%} |")

md_path = os.path.join(args.out_dir, "summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Summary -> {md_path}")
print(f"Results -> {args.out_dir}/")
