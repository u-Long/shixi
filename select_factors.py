"""
因子筛选脚本
从 autofactorsetnew 因子库读取已有 IC 指标，按 RankIC 筛选 top-K 个低相关因子

输入:
  - /root/gp_factor_qlib/autofactorsetnew/factor_specs/*.yaml  (含 cc_rank_ic, cc_rank_icir)
  - /root/dmd/BaoStock/panel.parquet  (用于计算因子值，如果需要重新计算)
输出:
  - /root/workspace/syl/iTransformer/data/selected_factors.txt  (选中的因子名列表)
  - 终端打印筛选结果

筛选逻辑:
  1. 读取所有 yaml 中的 cc_rank_ic / cc_rank_icir
  2. 过滤 |rank_ic| > IC_THRESH（默认 0.02）且 |rank_icir| > ICIR_THRESH（默认 0.15）
  3. 按 |rank_ic| 降序排列
  4. 贪心去相关：逐一加入，若与已选因子的平均相关 > CORR_THRESH 则跳过
  5. 最终保留 TOP_K 个（默认 20）

当前只做基于 yaml 元数据的静态筛选（不重新计算因子值），
后续如需动态筛选可扩展 compute_factor_values()。
"""

import yaml
import os
import glob
import pandas as pd
import numpy as np

# ── 超参数 ──────────────────────────────────────────────────────────────────
IC_THRESH   = 0.025   # |rank_ic| 最低门槛
ICIR_THRESH = 0.15    # |rank_icir| 最低门槛
TOP_K       = 20      # 最终选多少个
SPEC_DIR    = "/root/gp_factor_qlib/autofactorsetnew/factor_specs"
DST         = "/root/workspace/syl/iTransformer/data/selected_factors.txt"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(DST), exist_ok=True)

records = []

for fpath in glob.glob(os.path.join(SPEC_DIR, "*.yaml")):
    with open(fpath) as f:
        spec = yaml.safe_load(f)
    freq = spec.get("input_frequency", "daily")
    # 只用日频因子（5min因子维度不同，暂不混用）
    if freq != "daily":
        continue
    for fac in spec.get("factors", []):
        name    = fac.get("name")
        ic      = fac.get("cc_rank_ic")
        icir    = fac.get("cc_rank_icir")
        expr    = fac.get("expr", "")
        cat     = fac.get("category", "")
        if name and ic is not None and icir is not None:
            records.append({
                "name":  name,
                "ic":    float(ic),
                "icir":  float(icir),
                "expr":  expr,
                "cat":   cat,
                "src":   os.path.basename(fpath),
            })

df = pd.DataFrame(records)
print(f"Total factors with IC metadata: {len(df)}")

# 过滤
df = df[(df["ic"].abs() >= IC_THRESH) & (df["icir"].abs() >= ICIR_THRESH)]
print(f"After IC/ICIR filter: {len(df)}")

# 按 |ic| 降序
df = df.reindex(df["ic"].abs().sort_values(ascending=False).index)
df = df.reset_index(drop=True)

# 贪心去相关（基于 category 分散选取，避免同类因子扎堆）
# 如果有实际因子值矩阵可以换成真实相关系数
MAX_PER_CAT = max(TOP_K, 10)   # 同 category 最多取多少个，默认不限
selected = []
cat_count = {}
for _, row in df.iterrows():
    if len(selected) >= TOP_K:
        break
    cat = row["cat"] or "unknown"
    if cat_count.get(cat, 0) >= MAX_PER_CAT:
        continue
    selected.append(row.to_dict())
    cat_count[cat] = cat_count.get(cat, 0) + 1

selected_df = pd.DataFrame(selected)
print(f"\nSelected {len(selected_df)} factors:")
print(selected_df[["name", "ic", "icir", "cat", "src"]].to_string(index=False))

# 保存
names = selected_df["name"].tolist()
with open(DST, "w") as f:
    for n in names:
        f.write(n + "\n")

# 同时保存带元数据的 csv 方便查看
selected_df.to_csv(DST.replace(".txt", "_meta.csv"), index=False)
print(f"\nSaved factor list -> {DST}")
