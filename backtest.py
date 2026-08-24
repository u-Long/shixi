"""
回测框架（执行层版）
参照 exec_topkdrop_backtest_package_20260820 的成本口径和 TopKDrop 执行逻辑。

收益口径:
  label = open_{T+2} / open_{T+1} - 1   (T+1 开盘买入, T+2 开盘卖出)
  成本: 买入 12bps, 卖出 17bps, 名义资金 1000 万, 单笔最低 5 元

用法:
  python backtest.py --ckpt checkpoints/stock/best.pt --cache_dir data/cache
  python backtest.py --topk 30 --n_drop 3 --out_dir backtest_results
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from data_provider.stock_dataset import StockDataset, DayBatchSampler
from model.iTransformer_stock import Model

# ── 参数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       default="checkpoints/stock/best.pt")
parser.add_argument("--cache_dir",  default="data/cache")
parser.add_argument("--seq_len",    type=int,   default=30)
parser.add_argument("--horizon",    type=int,   default=10)
# 数据集日期划分（默认与训练保持一致，优先从 checkpoint 恢复）
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
# 兼容旧比例划分
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio",   type=float, default=0.15)
# 模型结构
parser.add_argument("--d_model",    type=int,   default=256)
parser.add_argument("--n_heads",    type=int,   default=4)
parser.add_argument("--e_layers",   type=int,   default=2)
parser.add_argument("--d_ff",       type=int,   default=512)
parser.add_argument("--dropout",    type=float, default=0.0)
parser.add_argument("--mlp_hidden", type=int,   default=64)
parser.add_argument("--class_strategy", default="mean")
parser.add_argument("--embed",      default="fixed")
parser.add_argument("--freq",       default="b")
parser.add_argument("--factor",     type=int,   default=1)
parser.add_argument("--activation", default="gelu")
parser.add_argument("--output_attention", action="store_true")
# 执行参数
parser.add_argument("--topk",       type=int,   default=30,  help="持仓股数")
parser.add_argument("--n_drop",     type=int,   default=3,   help="TopKDrop 每日最多换出")
parser.add_argument("--capital",    type=float, default=10_000_000.0, help="名义资金（元）")
parser.add_argument("--buy_cost",   type=float, default=0.0012, help="买入费率（含冲击）")
parser.add_argument("--sell_cost",  type=float, default=0.0017, help="卖出费率（含冲击）")
parser.add_argument("--min_cost",   type=float, default=5.0,    help="单笔最低成本（元）")
parser.add_argument("--n_groups",   type=int,   default=5,    help="分层回测分组数")
parser.add_argument("--out_dir",    default="backtest_results")
parser.add_argument("--batch_size", type=int,   default=512)
parser.add_argument("--num_workers",type=int,   default=4)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 从 checkpoint 恢复训练参数 ────────────────────────────────────────────────
raw_ckpt = torch.load(args.ckpt, map_location="cpu")
if isinstance(raw_ckpt, dict) and "args" in raw_ckpt:
    saved = raw_ckpt["args"]
    for k in ["seq_len", "horizon",
              "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
              "train_ratio", "val_ratio",
              "d_model", "n_heads", "e_layers", "d_ff", "mlp_hidden",
              "class_strategy", "embed", "freq", "factor", "activation"]:
        if k in saved:
            setattr(args, k, saved[k])
    print(f"[backtest] Restored from ckpt: seq_len={args.seq_len} "
          f"test={args.test_start}~{args.test_end}")


# ── 加载 test 数据集（不做 rank label 归一化）────────────────────────────────
def raw_collate(batch):
    """推理时不做 rank 归一化，保留原始 label。shape 与训练 collate 对齐：(B, 1)。"""
    xs, ys, _ = zip(*batch)
    return torch.stack(xs), torch.stack(ys).unsqueeze(-1)  # ys: (B,) -> (B, 1)

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
    collate_fn=raw_collate, num_workers=args.num_workers
)

# ── 加载模型 ─────────────────────────────────────────────────────────────────
args.enc_in = test_ds.n_features
model = Model(args).to(device)
model.load_state_dict(raw_ckpt["state_dict"] if "state_dict" in raw_ckpt else raw_ckpt)
model.eval()
print(f"Loaded: {args.ckpt}  features={args.enc_in}")

# ── 推理 ─────────────────────────────────────────────────────────────────────
print("Running inference...")
all_pred, all_ret = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().squeeze()
        y_np = y.numpy().squeeze()
        # 单样本 batch 时 squeeze 降为 0-d，需保持 1-d
        all_pred.extend(np.atleast_1d(pred).tolist())
        all_ret.extend(np.atleast_1d(y_np).tolist())

samples = test_ds.samples
dates   = test_ds.dates
stocks  = test_ds.stocks

assert len(all_pred) == len(samples), (
    f"pred 数量 {len(all_pred)} 与 samples 数量 {len(samples)} 不匹配"
)
records = [
    {"date": dates[di], "stock": stocks[si], "pred": all_pred[i], "ret": all_ret[i]}
    for i, (di, si) in enumerate(samples)
]
df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "stock"]).reset_index(drop=True)
df.to_parquet(os.path.join(args.out_dir, "pred_detail.parquet"))
print(f"Inference done: {len(df)} records, {df.date.nunique()} dates")


# ══════════════════════════════════════════════════════════════════════════════
# IC / ICIR
# ══════════════════════════════════════════════════════════════════════════════
def daily_rankic(df):
    ics = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp["pred"], grp["ret"])
        ics.append({"date": date, "ic": float(ic)})
    return pd.DataFrame(ics).set_index("date")

ic_df  = daily_rankic(df)
mean_ic = ic_df["ic"].mean()
icir    = mean_ic / (ic_df["ic"].std() + 1e-8)
ic_pos  = (ic_df["ic"] > 0).mean()
ic_df.to_csv(os.path.join(args.out_dir, "ic_series.csv"))

print(f"\n== IC 统计 ==")
print(f"  Mean RankIC : {mean_ic:.4f}")
print(f"  ICIR        : {icir:.4f}")
print(f"  IC>0 胜率   : {ic_pos:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# 执行层回测（TopKDrop + 非对称手续费）
# 收益口径: model label = 已是 open_{T+2}/open_{T+1}-1，若用 close-to-close label
#           则此处 ret 为 close label，仅用于 IC 统计。
#           执行层净值单独用 df["ret"] 作为 o2o 的代理（若有 open_arr 则更精确）。
# ══════════════════════════════════════════════════════════════════════════════
TOPK    = args.topk
N_DROP  = args.n_drop
CAP     = args.capital
BUY_C   = args.buy_cost
SELL_C  = args.sell_cost
MIN_C   = args.min_cost


def cost_count(n_buy: int, n_sell: int) -> float:
    slot = CAP / TOPK
    return (n_buy * max(slot * BUY_C, MIN_C) + n_sell * max(slot * SELL_C, MIN_C)) / CAP


def topkdrop(score: pd.Series, current: list, topk: int, n_drop: int) -> list:
    pred = score.dropna().sort_values(ascending=False)
    if pred.empty:
        return list(current)
    if not current:
        return list(pred.head(topk).index)
    cur_set   = set(current)
    in_pred   = [c for c in current if c in pred.index]
    not_in    = [c for c in current if c not in pred.index]   # 行情缺失保留
    last      = pred.reindex(in_pred).sort_values(ascending=False).index.tolist()
    n_new     = max(0, n_drop + topk - len(not_in) - len(last))
    new_cands = pred.loc[~pred.index.isin(cur_set)].head(n_new).index.tolist()
    pool      = pred.reindex(last + new_cands).sort_values(ascending=False).index.tolist()
    sells     = set(pool[-n_drop:]) if n_drop > 0 else set()
    kept      = not_in + [c for c in last if c not in sells]
    buys      = [c for c in pred.index if c not in set(kept) and c not in cur_set]
    result    = (kept + buys)[:topk]
    if len(result) < topk:
        for c in pred.index:
            if c not in set(result):
                result.append(c)
            if len(result) >= topk:
                break
    return result[:topk]


def simulate_exec(daily_groups) -> pd.DataFrame:
    current: list = []
    rows = []
    for date, grp in daily_groups:
        score  = grp.set_index("stock")["pred"]
        ret_s  = grp.set_index("stock")["ret"]
        new    = topkdrop(score, current, TOPK, N_DROP)
        gross  = float(ret_s.reindex(new).fillna(0.0).mean()) if new else 0.0
        cur_set, new_set = set(current), set(new)
        n_sell = len(cur_set - new_set)
        n_buy  = len(new_set - cur_set) if current else len(new_set)
        net    = gross - cost_count(n_buy, n_sell)
        rows.append({
            "date": date, "gross": gross, "net": net,
            "n_buy": n_buy, "n_sell": n_sell,
            "turnover": (n_buy + n_sell) / TOPK,
        })
        current = new
    return pd.DataFrame(rows)


daily_groups = [(d, g) for d, g in df.groupby("date")]
exec_df = simulate_exec(daily_groups)
exec_df.to_csv(os.path.join(args.out_dir, "exec_daily.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 分层回测（纯 IC 视角，不含执行成本）
# ══════════════════════════════════════════════════════════════════════════════
group_rets = {g: [] for g in range(args.n_groups)}
ls_rets = []

for date, grp in df.groupby("date"):
    if len(grp) < args.n_groups * 2:
        continue
    grp = grp.sort_values("pred", ascending=False).reset_index(drop=True)
    n = len(grp)
    for g in range(args.n_groups):
        lo = int(g * n / args.n_groups)
        hi = int((g + 1) * n / args.n_groups)
        group_rets[g].append(grp.iloc[lo:hi]["ret"].mean())
    n_l = min(50, n // 4)
    n_s = min(50, n // 4)
    ls_rets.append({
        "date": date,
        "long_ret":  grp.iloc[:n_l]["ret"].mean(),
        "short_ret": grp.iloc[-n_s:]["ret"].mean(),
    })

ls_df = pd.DataFrame(ls_rets).set_index("date")
ls_df["ls_ret"] = ls_df["long_ret"] - ls_df["short_ret"]
ls_df.to_csv(os.path.join(args.out_dir, "ls_returns.csv"))


# ══════════════════════════════════════════════════════════════════════════════
# 统计函数（参照参考包）
# ══════════════════════════════════════════════════════════════════════════════
def stats(ret: pd.Series, label: str = "") -> dict:
    r = ret.fillna(0.0)
    n = len(r)
    nav = (1 + r).cumprod()
    prod  = float(nav.iloc[-1]) if n else np.nan
    std   = float(r.std(ddof=1)) if n > 1 else np.nan
    vol   = std * np.sqrt(252) if std == std else np.nan
    arith = r.mean() * 252 if n else np.nan
    geo   = prod ** (252 / n) - 1 if n and prod > 0 else np.nan
    mdd   = float((nav / nav.cummax() - 1).min()) if n else np.nan
    sharpe = arith / vol if vol and vol > 0 else np.nan
    calmar = arith / abs(mdd) if mdd and mdd < 0 else np.nan
    return {
        "label": label, "n_days": n,
        "ann_arith": arith, "ann_geo": geo, "ann_vol": vol,
        "sharpe": sharpe, "calmar": calmar,
        "max_dd": mdd, "total_ret": prod - 1 if prod == prod else np.nan,
        "hit": float((r > 0).mean()),
    }


def print_stats(s: dict):
    print(f"  {s['label']:<20s}  "
          f"年化={s['ann_arith']:.2%}  Sharpe={s['sharpe']:.2f}  "
          f"MDD={s['max_dd']:.2%}  总收益={s['total_ret']:.2%}  "
          f"胜率={s['hit']:.2%}")


# ── 多窗口统计 ────────────────────────────────────────────────────────────────
# exec_df 是按决策日记录的 o2o 收益，视为日频（每个决策日≈1日）
exec_df["date"] = pd.to_datetime(exec_df["date"])

windows = {
    "test_full": (args.test_start, args.test_end),
    "2025":      ("2025-01-01",    "2025-12-31"),
    "2026":      ("2026-01-01",    args.test_end),
}

print(f"\n== 执行层回测 (TopK={TOPK}, Drop={N_DROP}, 买{BUY_C*1e4:.0f}bps/卖{SELL_C*1e4:.0f}bps) ==")
all_stats = []
for wname, (ws, we) in windows.items():
    sub = exec_df[exec_df["date"].between(ws, we)]
    if sub.empty:
        continue
    sg = stats(sub["net"],   label=f"net  [{wname}]")
    gg = stats(sub["gross"], label=f"gross[{wname}]")
    print_stats(sg)
    print_stats(gg)
    avg_to = sub["turnover"].mean()
    ann_cost = (sub["gross"] - sub["net"]).mean() * 252 * 100
    print(f"    日均双边换手={avg_to:.2f}  年化成本={ann_cost:.2f}ppt")
    all_stats.append({**sg, "window": wname, "side": "net",
                      "avg_turnover": avg_to, "ann_cost_ppt": ann_cost})
    all_stats.append({**gg, "window": wname, "side": "gross"})

pd.DataFrame(all_stats).to_csv(os.path.join(args.out_dir, "exec_metrics.csv"), index=False)

print(f"\n== 多空分层（纯 IC，无手续费）==")
ls_s = stats(ls_df["ls_ret"] / args.horizon, label="L/S")
print_stats(ls_s)

print(f"\n== IC 总结 ==")
print(f"  Mean RankIC={mean_ic:.4f}  ICIR={icir:.4f}  IC>0={ic_pos:.2%}")

# ── 各组年化收益 ──────────────────────────────────────────────────────────────
print(f"\n== 分层收益 (年化近似，{args.n_groups} 组) ==")
for g in range(args.n_groups):
    mean_r = np.mean(group_rets[g])
    ann    = mean_r * 252 / args.horizon
    tag    = "多头" if g == 0 else ("空头" if g == args.n_groups - 1 else f"G{g+1}")
    print(f"  Group {g+1} ({tag:4s}): 年化={ann:.2%}  平均{args.horizon}日收益={mean_r:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 画图（4 子图）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(13, 16))
fig.suptitle(f"iTransformer Backtest  TopK={TOPK} Drop={N_DROP}", fontsize=13)

# 1. 执行层净值曲线（net vs gross）
exec_df_sorted = exec_df.sort_values("date")
cum_net   = (1 + exec_df_sorted["net"]).cumprod()
cum_gross = (1 + exec_df_sorted["gross"]).cumprod()
axes[0].plot(exec_df_sorted["date"], cum_net,   label="Net (after cost)", color="steelblue")
axes[0].plot(exec_df_sorted["date"], cum_gross, label="Gross",            color="green", alpha=0.7)
axes[0].axhline(1, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_title("Execution Layer: Cumulative Return (TopKDrop)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2. 多空净值（IC 视角）
cum_ls_   = (1 + ls_df["ls_ret"] / args.horizon).cumprod()
cum_long_ = (1 + ls_df["long_ret"] / args.horizon).cumprod()
axes[1].plot(cum_ls_.index,   cum_ls_.values,   label="Long-Short", color="darkred")
axes[1].plot(cum_long_.index, cum_long_.values, label="Long Only",  color="olive", alpha=0.7)
axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_title("Factor Layer: L/S Cumulative Return (no cost)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 3. 日度 RankIC
axes[2].bar(ic_df.index, ic_df["ic"],
            color=["green" if v > 0 else "red" for v in ic_df["ic"]], alpha=0.6)
axes[2].axhline(0, color="black", linewidth=0.8)
ic_ma = ic_df["ic"].rolling(20).mean()
axes[2].plot(ic_ma.index, ic_ma.values, color="navy", linewidth=1.5, label="20d MA")
axes[2].set_title(f"Daily RankIC  mean={mean_ic:.4f}  ICIR={icir:.2f}")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

# 4. 分层收益柱状图
group_means = [np.mean(group_rets[g]) * 252 / args.horizon for g in range(args.n_groups)]
colors = ["green" if v > 0 else "red" for v in group_means]
axes[3].bar([f"G{g+1}" for g in range(args.n_groups)], group_means, color=colors, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8)
axes[3].set_title("Annualized Return by Quintile")
axes[3].set_ylabel("Ann. Return"); axes[3].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig_path = os.path.join(args.out_dir, "backtest.png")
plt.savefig(fig_path, dpi=150)
print(f"\nPlot -> {fig_path}")
print(f"All results -> {args.out_dir}/")
