"""
归因分析：判断策略收益里有多少是真 alpha，有多少是市值/行业 beta。

背景 —— 为什么需要这个脚本
────────────────────────────────────────────────────────────────────────
backtest.py 只用上证指数(SH000001)做基准。上证是大盘加权指数，如果策略实际
持仓集中在小微盘，那么整个小盘/大盘的风格价差都会被记成「超额」。

2026 年 5-7 月中证 2000 自高点回撤逾 26%（Wind），而 fea2_ret5davgo 那次回测
的最大回撤 -21.63% 恰好落在同一时间窗。这个巧合必须用回归证伪或证实，
不能靠看图。

本脚本做四件事：
  1. 多基准净值对比 —— 除外部指数外，直接从 panel 现算「市值分层等权组合」，
     不依赖任何额外指数数据就能得到小微盘代理
  2. 收益归因回归 —— r_port = alpha + beta_mkt·MKT + beta_size·SMB + eps
     报告年化 alpha、t 值，以及 alpha 占总收益的比例
  3. 中性化 IC —— 预测值先对 log(流通市值)+行业做截面回归取残差，再算 IC。
     中性化后 IC 若大幅衰减，说明信号本质是市值因子
  4. 诊断 —— 每日有效股票数、持仓市值分布、IC 的 Newey-West t 值
     （修正重叠 label 造成的自相关，普通 t 值会高估约 √span 倍）

输入（均由 backtest.py 产出）:
  {out_dir}/pred_detail.parquet          date, stock, pred, o2o
  {out_dir}/exec_baseline_k30_d3.csv     date, gross, net, bench_o2o

用法:
  python scripts/attribution.py --bt_dir backtest_results/826_fea2_ret5davgo
  python scripts/attribution.py --bt_dir ... --index_dir /root/dmd/BaoStock/Index
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--bt_dir",     required=True, help="backtest.py 的 out_dir")
parser.add_argument("--exec_csv",   default=None,
                    help="执行层日收益 csv，默认自动找 exec_baseline_*.csv")
parser.add_argument("--panel_path", default="/root/dmd/BaoStock/panel.parquet")
parser.add_argument("--index_dir",  default="/root/dmd/BaoStock/Index",
                    help="外部指数 csv 目录。缺失的指数会自动跳过，不影响其余分析")
parser.add_argument("--industry_csv", default=None,
                    help="可选，两列 instrument,industry。提供后 IC 中性化会同时剔除行业")
parser.add_argument("--mv_col",     default=None,
                    help="panel 中流通市值列名。不指定则用 close*volume/(turnover/100) 推算")
parser.add_argument("--n_size_groups", type=int, default=5)
parser.add_argument("--out_dir",    default=None, help="默认写回 bt_dir")
args = parser.parse_args()

OUT = args.out_dir or args.bt_dir
os.makedirs(OUT, exist_ok=True)

# 外部指数候选。文件不存在就跳过，不报错。
INDEX_FILES = {
    "SH000001 上证":     "sh.000001.csv",
    "CSI300 沪深300":    "sh.000300.csv",
    "CSI500 中证500":    "sh.000905.csv",
    "CSI1000 中证1000":  "sh.000852.csv",
    "CSI2000 中证2000":  "sh.932000.csv",
    "SZ399006 创业板":   "sz.399006.csv",
}


# ══════════════════════════════════════════════════════════════════════
# 载入
# ══════════════════════════════════════════════════════════════════════
pred_path = os.path.join(args.bt_dir, "pred_detail.parquet")
print(f"Loading {pred_path}")
pred = pd.read_parquet(pred_path)
pred["date"] = pd.to_datetime(pred["date"])
print(f"  {len(pred)} rows, {pred['date'].nunique()} days, {pred['stock'].nunique()} stocks")

if args.exec_csv:
    exec_path = args.exec_csv
else:
    cands = [f for f in os.listdir(args.bt_dir) if f.startswith("exec_baseline")]
    if not cands:
        raise FileNotFoundError("找不到 exec_baseline_*.csv，请用 --exec_csv 指定")
    exec_path = os.path.join(args.bt_dir, cands[0])
print(f"Loading {exec_path}")
ex = pd.read_csv(exec_path)
ex["date"] = pd.to_datetime(ex["date"])
ex = ex.set_index("date").sort_index()
port_ret = ex["net"].fillna(0.0)
D0, D1 = port_ret.index.min(), port_ret.index.max()
n_days  = len(port_ret)
years   = n_days / 252.0
print(f"  {n_days} 交易日  {D0.date()} ~ {D1.date()}  ({years:.2f} 年)")

print(f"\nLoading panel: {args.panel_path}")
panel = pd.read_parquet(args.panel_path)
if isinstance(panel.index, pd.MultiIndex):
    panel = panel.reset_index()
panel.columns = [str(c).strip().lower() for c in panel.columns]
panel["datetime"] = pd.to_datetime(panel["datetime"])
panel = panel[(panel["datetime"] >= D0) & (panel["datetime"] <= D1)]

# 流通市值：优先用现成列，否则由 turnover 反推流通股本
if args.mv_col and args.mv_col in panel.columns:
    panel["mv"] = panel[args.mv_col].astype(float)
    print(f"  市值来源: 列 '{args.mv_col}'")
else:
    cand = [c for c in ("circ_mv", "float_mv", "negotiable_mv", "market_cap") if c in panel.columns]
    if cand:
        panel["mv"] = panel[cand[0]].astype(float)
        print(f"  市值来源: 列 '{cand[0]}'")
    else:
        # turnover 为百分比: 流通股本 = volume / (turnover/100)
        to = pd.to_numeric(panel["turnover"], errors="coerce") / 100.0
        shares = panel["volume"].astype(float) / to.replace(0, np.nan)
        panel["mv"] = panel["close"].astype(float) * shares
        print("  市值来源: close*volume/(turnover/100) 推算")
print(f"  mv 有效率: {panel['mv'].notna().mean():.2%}")


# ══════════════════════════════════════════════════════════════════════
# 1. 构造市值分层等权基准（不依赖外部指数数据）
#    用 o2o 结算，与策略口径一致：T+1 开盘买入、T+2 开盘卖出
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("1. 市值分层等权基准（o2o 结算，与策略同口径）")
print("=" * 74)

p = panel[["datetime", "instrument", "open", "mv"]].rename(
    columns={"datetime": "date", "instrument": "stock"}).sort_values(["stock", "date"])
p["o2o_bench"] = p.groupby("stock")["open"].shift(-2) / p.groupby("stock")["open"].shift(-1) - 1.0
p = p.dropna(subset=["o2o_bench", "mv"])
# 极端值来自停复牌跨期，剔除而非填 0
p = p[p["o2o_bench"].abs() < 0.21]

G = args.n_size_groups
p["size_grp"] = p.groupby("date")["mv"].transform(
    lambda s: pd.qcut(s.rank(method="first"), G, labels=False))

size_bench = p.pivot_table(index="date", columns="size_grp",
                           values="o2o_bench", aggfunc="mean")
size_bench.columns = [f"S{int(c)+1}{'(最小)' if c == 0 else '(最大)' if c == G-1 else ''}"
                      for c in size_bench.columns]
eqw_all = p.groupby("date")["o2o_bench"].mean().rename("全A等权")

bench_df = size_bench.join(eqw_all, how="outer").reindex(port_ret.index).fillna(0.0)

# SMB（small minus big）：最小市值组 − 最大市值组
smb = (bench_df.iloc[:, 0] - bench_df.iloc[:, G - 1]).rename("SMB")

print(f"\n{'组':<14}{'年化':>10}{'年化波动':>10}{'累计':>10}")
print("-" * 46)
for c in bench_df.columns:
    r = bench_df[c]
    ann = (1 + r).prod() ** (252 / n_days) - 1
    print(f"{c:<14}{ann:>+10.2%}{r.std()*np.sqrt(252):>10.2%}{(1+r).prod()-1:>+10.2%}")
smb_ann = (1 + smb).prod() ** (252 / n_days) - 1
print(f"{'SMB(最小-最大)':<14}{smb_ann:>+10.2%}{smb.std()*np.sqrt(252):>10.2%}"
      f"{(1+smb).prod()-1:>+10.2%}")

# 外部指数
ext = {}
for name, fn in INDEX_FILES.items():
    fp = os.path.join(args.index_dir, fn)
    if not os.path.exists(fp):
        continue
    idx = pd.read_csv(fp)
    idx.columns = [c.strip().lower() for c in idx.columns]
    dcol = "date" if "date" in idx.columns else idx.columns[0]
    idx[dcol] = pd.to_datetime(idx[dcol])
    idx = idx.sort_values(dcol)
    idx["r"] = idx["open"].shift(-2) / idx["open"].shift(-1) - 1.0
    s = idx.set_index(dcol)["r"].reindex(port_ret.index).fillna(0.0)
    ext[name] = s
if ext:
    print(f"\n外部指数（{len(ext)} 个，o2o 口径）:")
    for name, s in ext.items():
        ann = (1 + s).prod() ** (252 / n_days) - 1
        print(f"  {name:<20}{ann:>+10.2%}")
else:
    print(f"\n[NOTE] {args.index_dir} 下未找到指数文件，仅用分层等权基准")


# ══════════════════════════════════════════════════════════════════════
# 2. 归因回归
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("2. 收益归因回归")
print("=" * 74)


def ols(y, X, names):
    """返回 (coef, t_NW)。Newey-West lag=10，修正重叠 label 的自相关。"""
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    L = 10
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        u = resid[:, None] * X
        G_ = u[l:].T @ u[:-l]
        S += w * (G_ + G_.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return dict(zip(["alpha"] + names, b)), dict(zip(["alpha"] + names, b / (se + 1e-12)))


y = port_ret.values
mkt = bench_df["全A等权"].values

models = [
    ("单因子: 全A等权",              [mkt],                    ["MKT"]),
    ("双因子: 全A等权 + SMB",        [mkt, smb.values],        ["MKT", "SMB"]),
]
if "CSI2000 中证2000" in ext:
    models.append(("双因子: 全A等权 + 中证2000",
                   [mkt, ext["CSI2000 中证2000"].values], ["MKT", "CSI2000"]))

total_ann = (1 + port_ret).prod() ** (252 / n_days) - 1
print(f"\n策略年化（净值口径）: {total_ann:+.2%}\n")

for label, cols, names in models:
    coef, tval = ols(y, cols, names)
    ann_alpha = coef["alpha"] * 252
    print(f"{label}")
    print(f"  年化 alpha = {ann_alpha:>+8.2%}   t_NW = {tval['alpha']:>+6.2f}"
          f"   占总收益 {ann_alpha/total_ann*100 if total_ann else float('nan'):>5.1f}%")
    for nm in names:
        print(f"  beta_{nm:<10} = {coef[nm]:>+7.3f}   t_NW = {tval[nm]:>+6.2f}")
    print()

print("读法：t_NW 绝对值 < 2 的 alpha 不能认为显著。")
print("      beta_SMB 显著为正 => 策略在赚小市值风格的钱，不是选股。")


# ══════════════════════════════════════════════════════════════════════
# 3. 中性化 IC
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("3. IC 中性化：预测值剔除 log(市值) 与行业后重算")
print("=" * 74)

mv_map = panel[["datetime", "instrument", "mv"]].rename(
    columns={"datetime": "date", "instrument": "stock"})
d = pred.merge(mv_map, on=["date", "stock"], how="left")
d = d.dropna(subset=["pred", "o2o", "mv"])
d["logmv"] = np.log(d["mv"].clip(lower=1))

ind_map = None
if args.industry_csv and os.path.exists(args.industry_csv):
    ind = pd.read_csv(args.industry_csv)
    ind.columns = ["stock", "industry"]
    d = d.merge(ind, on="stock", how="left")
    d["industry"] = d["industry"].fillna("UNKNOWN")
    ind_map = True
    print(f"  行业数据: {d['industry'].nunique()} 个行业")
else:
    print("  行业数据: 未提供（只做市值中性化）")


def neutralize(g):
    """对 log(市值)[+行业哑变量] 做截面 OLS，返回残差。"""
    X = [np.ones(len(g)), g["logmv"].values]
    if ind_map:
        dm = pd.get_dummies(g["industry"], drop_first=True).values.astype(float)
        if dm.shape[1] > 0:
            X.append(dm)
    X = np.column_stack(X)
    yv = g["pred"].values
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return pd.Series(yv - X @ b, index=g.index)


print("  computing ...")
d["pred_neu"] = d.groupby("date", group_keys=False).apply(neutralize)


def daily_ic(frame, pcol):
    out = frame.groupby("date").apply(
        lambda g: pd.Series({
            "ic":     g[pcol].corr(g["o2o"]),
            "rankic": g[pcol].corr(g["o2o"], method="spearman"),
            "n":      len(g),
        }))
    return out.dropna(subset=["rankic"])


ic_raw = daily_ic(d, "pred")
ic_neu = daily_ic(d, "pred_neu")


def nw_t(x, lag=10):
    x = np.asarray(x, dtype=float)
    n, m = len(x), x.mean()
    e = x - m
    s = (e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return m / np.sqrt(max(s, 1e-16) / n)


print(f"\n{'':<16}{'RankIC':>10}{'RankICIR':>11}{'t_NW':>9}{'IC>0':>9}")
print("-" * 55)
for nm, s in [("原始", ic_raw), ("市值中性后", ic_neu)]:
    r = s["rankic"]
    print(f"{nm:<16}{r.mean():>+10.4f}{r.mean()/(r.std()+1e-9):>11.3f}"
          f"{nw_t(r.values):>9.2f}{(r > 0).mean():>9.1%}")

decay = 1 - ic_neu["rankic"].mean() / (ic_raw["rankic"].mean() + 1e-12)
print(f"\n中性化后 RankIC 衰减 {decay:.1%}")
if decay > 0.5:
    print("  => 一半以上的 IC 来自市值暴露，信号主体是市值因子。")
elif decay > 0.25:
    print("  => 市值贡献可观，建议在排序前做中性化再回测。")
else:
    print("  => 市值贡献有限，IC 主要来自其他信息。")
print("\n注意：t_NW 已用 Newey-West(lag=10) 修正重叠 label 的自相关；")
print("      未修正的 ICIR×√n 会高估约 √(horizon+1) 倍。")


# ══════════════════════════════════════════════════════════════════════
# 4. 诊断
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("4. 诊断")
print("=" * 74)

print("\n每日有效股票数分位:")
print(ic_raw["n"].describe(percentiles=[.01, .05, .5, .95]).round(0).to_string())
few = ic_raw[ic_raw["n"] < 200]
if len(few):
    print(f"\n[WARN] {len(few)} 天有效股票数 < 200，这些天的 IC 噪声极大：")
    print(few.sort_values("n").head(10)[["n", "rankic"]].round(4).to_string())

print("\nRankIC 绝对值最大的 10 天（|IC|>3/√n 属异常，需核对当日 universe）:")
tmp = ic_raw.copy()
tmp["se_null"] = 1 / np.sqrt(tmp["n"])
tmp["z"] = tmp["rankic"] / tmp["se_null"]
print(tmp.reindex(tmp["rankic"].abs().sort_values(ascending=False).index)
        .head(10)[["n", "rankic", "z"]].round(3).to_string())

# 持仓市值分布：用每日 pred 前 30 名近似
top = d.sort_values(["date", "pred"], ascending=[True, False]).groupby("date").head(30)
mv_stat = top.groupby("date")["mv"].median() / 1e8
print(f"\nTop30 持仓流通市值中位数（亿元）:")
print(mv_stat.describe(percentiles=[.05, .5, .95]).round(2).to_string())
print(f"  全市场同期中位数: {(d.groupby('date')['mv'].median()/1e8).mean():.2f} 亿")
if mv_stat.mean() < (d.groupby("date")["mv"].median() / 1e8).mean() * 0.6:
    print("  => 持仓显著偏小市值，12/17bps 的成本假设过于乐观，")
    print("     小微盘单边冲击成本通常 50~100bps，建议用 --slippage_bps 重跑。")

# 分年
print("\n分年表现:")
yr = port_ret.groupby(port_ret.index.year)
print(f"{'年':<8}{'年化':>10}{'Sharpe':>9}{'天数':>7}")
print("-" * 34)
for y_, r in yr:
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    sh = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    print(f"{y_:<8}{ann:>+10.2%}{sh:>9.2f}{len(r):>7}")
se_sharpe = np.sqrt(1.0 / years)
sh_all = port_ret.mean() / (port_ret.std() + 1e-12) * np.sqrt(252)
print(f"\n全期 Sharpe = {sh_all:.2f}，标准误 ≈ {se_sharpe:.2f}"
      f"（95%CI 约 [{sh_all-2*se_sharpe:.2f}, {sh_all+2*se_sharpe:.2f}]）")


# ══════════════════════════════════════════════════════════════════════
# 5. 多基准净值图
# ══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for f in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]:
    matplotlib.rcParams["font.sans-serif"] = [f]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(14, 15))

ax = axes[0]
ax.plot((1 + port_ret).cumprod(), lw=2.2, color="crimson", label="策略(net)", zorder=5)
for c in bench_df.columns:
    ax.plot((1 + bench_df[c]).cumprod(), lw=1.0, alpha=0.75, label=c)
for name, s in ext.items():
    ax.plot((1 + s).cumprod(), lw=1.0, ls="--", alpha=0.7, label=name)
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_title("净值对比：策略 vs 市值分层等权 vs 外部指数（o2o 同口径）")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
for c in [bench_df.columns[0], bench_df.columns[-1], "全A等权"]:
    ax.plot((1 + port_ret).cumprod() / (1 + bench_df[c]).cumprod(),
            lw=1.4, label=f"策略 / {c}")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_title("相对净值：分母换成小盘基准后，超额还剩多少")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(ic_raw["rankic"].rolling(20).mean(), lw=1.4, label="原始 RankIC (20d MA)")
ax.plot(ic_neu["rankic"].rolling(20).mean(), lw=1.4, label="市值中性 RankIC (20d MA)")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("中性化前后 RankIC 对比 —— 两条线的落差就是市值因子的贡献")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT, "attribution.png")
plt.savefig(fig_path, dpi=110)
print(f"\n图 -> {fig_path}")

pd.concat([ic_raw.add_prefix("raw_"), ic_neu.add_prefix("neu_")], axis=1) \
  .to_csv(os.path.join(OUT, "ic_neutralized.csv"))
bench_df.join(smb).to_csv(os.path.join(OUT, "benchmarks.csv"))
print(f"表 -> {OUT}/ic_neutralized.csv, {OUT}/benchmarks.csv")
print("\nDone.")
