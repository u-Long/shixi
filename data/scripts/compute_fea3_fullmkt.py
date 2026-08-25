"""
用 autofactorsetnew 评估框架计算全量 fea3 因子（2015-2026全市场）
直接调用 compute_factor_series，不走评估服务，只算因子值

输出: data/feature_lib/fea3_alpha191.parquet
      (datetime×instrument MultiIndex, 20列因子)
"""

import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, '/root')

from gp_factor_qlib.evaluation.single_factor_eval.factor_eval_runner import (
    compute_factor_series,
    prepare_factor_evaluation_inputs,
)

FACTOR_LIST_PATH = "data/selected_factors.txt"
DB_PATH          = "/root/gp_factor_qlib/autofactorsetnew/factorlibrary/storage/library.db"
DATA_DIR         = "/root/dmd/BaoStock/panel.parquet"
START_DATE       = "2015-01-01"
END_DATE         = "2026-08-20"
OUT_PATH         = "data/feature_lib/fea3_alpha191.parquet"
OUT_DIR          = "data/feature_lib"

os.makedirs(OUT_DIR, exist_ok=True)

# 读取因子列表
with open(FACTOR_LIST_PATH) as f:
    selected = [l.strip() for l in f if l.strip()]
print(f"Selected factors: {selected}")

# 从 library.db 获取 expr_json
conn = sqlite3.connect(DB_PATH)
placeholders = ",".join("?" for _ in selected)
rows = conn.execute(
    f"SELECT factor_name, expr_json FROM library_factors WHERE factor_name IN ({placeholders})",
    selected,
).fetchall()
conn.close()
name2expr_json = {r[0]: r[1] for r in rows}
print(f"Found expr_json for {len(name2expr_json)}/{len(selected)} factors")

# 逐个计算因子全量序列
# prepare_factor_evaluation_inputs 已经把因子值算好放在 inputs.factor 里
results = {}
for name in selected:
    if name not in name2expr_json:
        print(f"[SKIP] {name}: no expr_json in library")
        continue

    print(f"Computing {name} ...")
    try:
        inputs = prepare_factor_evaluation_inputs(
            expr_json    = name2expr_json[name],
            factor_name  = name,
            data_source  = "baostock_parquet",
            data_dir     = DATA_DIR,
            data_begin   = START_DATE,
            data_end     = END_DATE,
            n_sh_stocks  = 0,   # 0 = 全部
            n_sz_stocks  = 0,
            eval_backend = "python",
            use_input_cache = True,
        )
        series = inputs.factor  # pd.Series，index=(datetime, instrument)
        if series.index.names == ["instrument", "datetime"]:
            series = series.swaplevel().sort_index()
        results[name] = series
        print(f"  -> {len(series)} rows, NaN={series.isna().mean():.3f}")
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

# 合并成宽表
print(f"\nMerging {len(results)} factors ...")
if not results:
    raise RuntimeError("All factors failed to compute. Check errors above.")
df = pd.DataFrame(results)
if df.index.names != ["datetime", "instrument"]:
    df.index.names = ["datetime", "instrument"]
df = df.sort_index()

# 截面 z-score
print("Cross-sectional z-score ...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.groupby(level="datetime").transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
df = df.replace([np.inf, -np.inf], np.nan)

print(f"\nfea3 shape: {df.shape}")
print(f"date range: {df.index.get_level_values('datetime').min()} -> {df.index.get_level_values('datetime').max()}")
print(f"stocks: {df.index.get_level_values('instrument').nunique()}")
print(f"NaN: {df.isna().mean().mean():.4f}")

df.to_parquet(OUT_PATH)
print(f"Saved -> {OUT_PATH}")
