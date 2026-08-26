"""
缓存构建脚本（特征库版）
从 feature_lib/ 中选择特征集合 + label，构建 (T, S, F) numpy 缓存

用法:
  python build_cache.py                              # 默认: fea1, label=ret_10d_log, universe=all
  python build_cache.py --fea fea1 fea3              # 拼合 fea1+fea3
  python build_cache.py --fea fea1 fea2 fea3         # 全特征
  python build_cache.py --label ret_5d_log           # 换 label
  python build_cache.py --universe hs300             # 只保留沪深300成分股
  python build_cache.py --universe zz500             # 中证500（市值分层近似）
  python build_cache.py --universe zz1000            # 中证1000
  python build_cache.py --fea fea1 --tag v2          # 自定义缓存目录后缀

支持的 universe:
  all     全市场（默认）
  hs300   沪深300（历史成分，按季度快照）
  zz500   中证500（市值近似：流通市值301-800名，排除hs300）
  zz1000  中证1000（市值近似：流通市值801-1800名）
  zz800   沪深300+中证500（流通市值前800）
  zz2000  中证2000（历史成分，仅2023年后有完整数据）
  custom  自定义列表文件（配合 --custom_path）

缓存目录: data/cache/cache_{tag}/
  feat_arr.npy          (T, S, F)  float32
  close_arr.npy         (T, S)     float32  原始 close
  label_arr.npy         (T, S)     float32  指定 label
  universe_mask.npy     (T, S)     bool     点时间 universe 成员掩码（非 all 时保存）
  dates.npy             (T,)       datetime64
  stocks.npy            (S,)       str
  feature_cols.npy      (F,)       str
  meta.json                        构建参数记录
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

# 确保能 import data/universe.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument("--fea",        nargs="+", default=["fea1"],
                    help="特征集列表，如 fea1 fea2 fea3")
parser.add_argument("--label",      default="ret_10d_log",
                    help="label 列名（label_lib.parquet 中的列）")
parser.add_argument("--universe",   default="all",
                    help="股票池: all/hs300/zz500/zz1000/zz800/zz2000/custom")
parser.add_argument("--custom_path",default=None,
                    help="universe=custom 时，每行一个股票代码的文件路径")
parser.add_argument("--start_date", default="2015-01-01")
parser.add_argument("--end_date",   default="2026-08-20")
parser.add_argument("--fea_dir",    default="data/feature_lib")
parser.add_argument("--panel_path", default="/root/dmd/BaoStock/panel.parquet")
parser.add_argument("--tag",        default=None,
                    help="缓存目录后缀，默认自动生成 fea1_fea2_hs300_ret10d")
parser.add_argument("--fea2_file",  default=None,
                    help="覆盖 fea2 的 parquet 文件名（如 fea2_price_new_0826.parquet），默认 fea2_price_new.parquet")
args = parser.parse_args()

# 自动生成 tag（含 universe，全市场时省略）
if args.tag is None:
    fea_str   = "_".join(args.fea)
    label_str = args.label.replace("_log","").replace("_open","o").replace("_","")
    univ_str  = f"_{args.universe}" if args.universe != "all" else ""
    args.tag  = f"{fea_str}{univ_str}_{label_str}_new" # 区别在于fea2是否做截面归一化

cache_dir = f"data/cache/cache_{args.tag}"
os.makedirs(cache_dir, exist_ok=True)
print(f"Cache dir: {cache_dir}")

start = pd.Timestamp(args.start_date)
end   = pd.Timestamp(args.end_date)


# ── 加载并拼合特征 ────────────────────────────────────────────────────────────
fea_name_map = {
    "fea1": "fea1_price_basic.parquet",
    "fea2": args.fea2_file if args.fea2_file else "fea2_price_new.parquet",
    "fea3": "fea3_alpha191.parquet",
}

feat_dfs, loaded_names = [], []
for fea in args.fea:
    fname = fea_name_map.get(fea, fea + ".parquet")
    fpath = os.path.join(args.fea_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] {fpath} not found, skipping {fea}")
        continue
    print(f"Loading {fea} from {fpath} ...")
    df = pd.read_parquet(fpath)
    df.index = df.index.set_levels(
        pd.to_datetime(df.index.get_level_values("datetime").unique()), level="datetime"
    )
    feat_dfs.append(df)
    loaded_names.append(fea)   # 与 feat_dfs 严格对齐，跳过的特征集不会让前缀错位

if not feat_dfs:
    raise RuntimeError("No feature files loaded. Run build_feature_lib.py first.")

# 多特征集拼合：只给「与已收录列重名」的列加特征集前缀，避免 join 报错
# （如 fea1/fea2 共有 ret_5d）。seen 是累积集合，所以第一个特征集永远不改名，
# 单独加载某个特征集时列名与多集拼合时保持一致，缓存之间可互换。
seen, labeled = set(), []
for fea_name, df in zip(loaded_names, feat_dfs):
    dup = set(df.columns) & seen
    if dup:
        print(f"  [rename] {fea_name}: {sorted(dup)} -> 加前缀 {fea_name}_")
        df = df.rename(columns={c: f"{fea_name}_{c}" for c in dup})
    seen |= set(df.columns)
    labeled.append(df)

feat_df = labeled[0] if len(labeled) == 1 else labeled[0].join(labeled[1:], how="outer")
# 过滤日期
mask = (feat_df.index.get_level_values("datetime") >= start) & \
       (feat_df.index.get_level_values("datetime") <= end)
feat_df = feat_df[mask]
print(f"Feature shape after date filter: {feat_df.shape}")
print(f"Feature cols ({len(feat_df.columns)}): {feat_df.columns.tolist()}")


# ── 加载 label ────────────────────────────────────────────────────────────────
label_path = os.path.join(args.fea_dir, "label_lib.parquet")
if os.path.exists(label_path):
    print(f"Loading label '{args.label}' from {label_path} ...")
    label_df = pd.read_parquet(label_path, columns=[args.label])
    label_df.index = label_df.index.set_levels(
        pd.to_datetime(label_df.index.get_level_values("datetime").unique()), level="datetime"
    )
    mask_l = (label_df.index.get_level_values("datetime") >= start) & \
             (label_df.index.get_level_values("datetime") <= end)
    label_df = label_df[mask_l]
    has_label = True
else:
    print("[WARN] label_lib.parquet not found, label_arr will be empty")
    has_label = False


# ── 加载原始 close（用于数据集内部计算 label 时的 fallback）────────────────────
print("Loading raw close ...")
raw = pd.read_parquet(args.panel_path, columns=["close"])
raw.index = raw.index.set_levels(
    pd.to_datetime(raw.index.get_level_values("datetime").unique()), level="datetime"
)
mask_r = (raw.index.get_level_values("datetime") >= start) & \
         (raw.index.get_level_values("datetime") <= end)
raw = raw[mask_r]
raw = raw[raw["close"] > 0]


# ── 确定 dates / stocks ────────────────────────────────────────────────────────
all_dates = sorted(feat_df.index.get_level_values("datetime").unique())

# 初步候选股票：特征库与 close 数据的交集
candidate_stocks = sorted(
    set(feat_df.index.get_level_values("instrument")) &
    set(raw.index.get_level_values("instrument"))
)

# Universe 过滤：保留曾经进入过该 universe 的股票
if args.universe != "all":
    print(f"\nApplying universe filter: {args.universe} ...")
    from data.universe import get_universe_stocks, build_universe_mask

    # 读取 panel 用于市值排名（仅 zz500/zz1000 需要）
    if args.universe in ("zz500", "zz1000", "zz800"):
        print("  Loading panel for size-rank universe ...")
        panel_for_univ = pd.read_parquet(args.panel_path,
            columns=["amount", "trade_status", "is_st"])
        panel_for_univ.index = panel_for_univ.index.set_levels(
            pd.to_datetime(panel_for_univ.index.get_level_values("datetime").unique()),
            level="datetime",
        )
        mask_p = (panel_for_univ.index.get_level_values("datetime") >= start) & \
                 (panel_for_univ.index.get_level_values("datetime") <= end)
        panel_for_univ = panel_for_univ[mask_p]
    else:
        panel_for_univ = None  # hs300/zz2000 只需要 weight CSV，不用 panel

    # 取全量 universe 中曾经出现过的股票，用于缩减 S 维度。
    # zz500/zz1000/zz800：需要逐月快照求并集（而非只取最后一天），避免生存偏差。
    # hs300/zz2000：走 weight CSV 并集，不需要 panel。
    if panel_for_univ is not None:
        # 按月遍历所有快照，取并集
        all_months = sorted({d.to_period("M") for d in all_dates})
        ever_in_univ = set()
        for m in all_months:
            snap = str(m.to_timestamp(how="E").date())
            try:
                members = get_universe_stocks(
                    args.universe, panel_for_univ, date=snap, custom_path=args.custom_path
                )
                ever_in_univ.update(members)
            except Exception:
                pass
    else:
        ever_in_univ = set(
            get_universe_stocks(args.universe, pd.DataFrame(),
                                date=None, custom_path=args.custom_path)
        )
    all_stocks = sorted(set(candidate_stocks) & ever_in_univ)
    print(f"  Candidate -> {len(candidate_stocks)}, after universe filter -> {len(all_stocks)}")

    # 构建逐日掩码
    T_tmp = len(all_dates)
    S_tmp = len(all_stocks)
    print(f"  Building point-in-time universe mask ({T_tmp} dates × {S_tmp} stocks) ...")
    if panel_for_univ is not None:
        univ_mask = build_universe_mask(
            args.universe, all_dates, all_stocks,
            panel_for_univ, custom_path=args.custom_path
        )
    else:
        # hs300 / zz2000：build_universe_mask 不需要 panel 内容，传空 DF 即可
        univ_mask = build_universe_mask(
            args.universe, all_dates, all_stocks,
            pd.DataFrame(), custom_path=args.custom_path
        )
    print(f"  Mask coverage: {univ_mask.mean():.3f} (avg daily fraction in universe)")
else:
    all_stocks = candidate_stocks
    univ_mask  = None

T = len(all_dates); S = len(all_stocks); F = len(feat_df.columns)
feature_cols = feat_df.columns.tolist()
date2i  = {d: i for i, d in enumerate(all_dates)}
stock2i = {s: i for i, s in enumerate(all_stocks)}
print(f"T={T}, S={S}, F={F}")


# ── 填充 feat_arr (T, S, F) ───────────────────────────────────────────────────
print("Building feat_arr ...")
feat_arr  = np.full((T, S, F), np.nan, dtype=np.float32)
close_arr = np.full((T, S),    np.nan, dtype=np.float32)
label_arr = np.full((T, S),    np.nan, dtype=np.float32)

def fill_arr(df, arr, cols=None):
    r = df.reset_index()
    r["di"] = r["datetime"].map(date2i)
    r["si"] = r["instrument"].map(stock2i)
    valid = r.dropna(subset=["di", "si"])
    di = valid["di"].values.astype(int)
    si = valid["si"].values.astype(int)
    if cols:
        arr[di, si, :] = valid[cols].values.astype(np.float32)
    else:
        arr[di, si] = valid.iloc[:, -1].values.astype(np.float32)

fill_arr(feat_df, feat_arr, feature_cols)

raw_r = raw.reset_index()
raw_r["di"] = raw_r["datetime"].map(date2i)
raw_r["si"] = raw_r["instrument"].map(stock2i)
vr = raw_r.dropna(subset=["di","si"])
close_arr[vr["di"].values.astype(int), vr["si"].values.astype(int)] = vr["close"].values.astype(np.float32)

if has_label:
    label_df_r = label_df.reset_index()
    label_df_r["di"] = label_df_r["datetime"].map(date2i)
    label_df_r["si"] = label_df_r["instrument"].map(stock2i)
    vl = label_df_r.dropna(subset=["di","si"])
    label_arr[vl["di"].values.astype(int), vl["si"].values.astype(int)] = \
        vl[args.label].values.astype(np.float32)


# ── 保存 ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(cache_dir, "feat_arr.npy"),     feat_arr)
np.save(os.path.join(cache_dir, "close_arr.npy"),    close_arr)
np.save(os.path.join(cache_dir, "label_arr.npy"),    label_arr)
np.save(os.path.join(cache_dir, "dates.npy"),        np.array(all_dates,  dtype="datetime64[ns]"))
np.save(os.path.join(cache_dir, "stocks.npy"),       np.array(all_stocks, dtype=str))
np.save(os.path.join(cache_dir, "feature_cols.npy"), np.array(feature_cols, dtype=str))

if univ_mask is not None:
    np.save(os.path.join(cache_dir, "universe_mask.npy"), univ_mask)
    print(f"  universe_mask : {univ_mask.shape}")

meta = {
    "fea": args.fea, "label": args.label,
    "universe": args.universe,
    "start_date": args.start_date, "end_date": args.end_date,
    "T": T, "S": S, "F": F,
    "feat_nan": float(np.isnan(feat_arr).mean()),
    "label_nan": float(np.isnan(label_arr).mean()),
}
with open(os.path.join(cache_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nCache saved -> {cache_dir}/")
print(f"  feat_arr : {feat_arr.shape}  {feat_arr.nbytes/1e9:.2f} GB")
print(f"  feat NaN : {meta['feat_nan']:.4f}")
print(f"  label    : {args.label}  NaN={meta['label_nan']:.4f}")
print(f"  universe : {args.universe}")
