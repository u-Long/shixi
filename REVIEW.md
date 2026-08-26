# 代码审查：泄漏与问题清单

审查范围：仓库全部代码（`data/`、`data_provider/`、`model/`、`layers/`、`scripts/`）。

---

## 已修复

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| P0.1 | `data/scripts/build_feature_lib.py` | fea2 缺截面 z-score，39列量纲差异2~3个数量级淹没模型 | 恢复 `cs_zscore` + `fillna(0)` |
| P0.2 | `data/scripts/build_feature_lib.py` | `peTTM_diff`/`pbMRQ_diff`/`psTTM_diff` 漏 clip，PE翻转时值上万 | 补 clip 区间 |
| P0.3 | `scripts/run_stock.py` | docstring 示例 `--ic_weight 10` 与原始 label 量纲错配 | 改为 `0.05`，docstring 删除 `_cs_rank_sym` 引用 |
| P1.6 | `data/scripts/build_feature_lib.py` | 3处 `bfill()` 用未来值回填序列开头 | 改为 `ffill()` |
| P1.7 | `data/scripts/build_cache.py` | zz500/zz1000/zz800 只取最后一天市值排名，严重生存偏差 | 改为逐月快照求并集 |
| P2.11 | `scripts/backtest.py`、`scripts/run_stock.py` | 5日重叠 label 逐日算 IC，ICIR 被高估约 √5 倍 | backtest 额外打印不重叠采样的 ICIR；run_stock 加注释说明 |
| P2.13 | `data_provider/stock_dataset.py` | `ret_X_open` label 实际 span 为 horizon+1，embargo 差一天 | 改用 `min(i + horizon + 1, T-1)` 做 embargo 判断 |
| P3.14 | `data/scripts/build_cache.py` | `panel_for_univ or pd.DataFrame()` 对非空 DF 求布尔值必崩 | 拆开三元表达式，分支显式处理 |
| P3.15 | `data/scripts/build_cache.py` | `--fea fea1 fea2` 重名列 join 必崩 | join 前检测重名列并加特征集前缀 |
| P3.16 | `scripts/run_stock.py` | docstring 引用 `ret_10d_cs_rank_sym` 不存在的列 | 删除，改为现有列口径说明 |
| P3.17 | `data/data_utils.py` | `ts_pro = ts.pro_api("")` 在 import 时执行，空 token 触发网络请求 | 改为懒加载代理对象 `_LazyTsPro` |

---

## 遗留（已知、低优先级）

### L1 — 回测候选池前视选择偏差（原 P1.4）

- **位置**：`scripts/backtest.py`（inference 后构建 df）
- **机制**：`StockDataset` 过滤条件含「label 非 NaN」，等价于要求未来 horizon 天股票仍正常交易。停牌/退市前的股票被静默剔除，恰好剔掉最容易出事的那批。
- **现状**：已在代码中加注释标记。现有的 `buyable` 过滤（涨停/停牌/ST）已部分缓解，对当前 hs300/全市场测试影响有限。
- **正确修法**：回测候选池独立构造，只用 T 日及以前的 `trade_status`/`is_st`/上市天数过滤，不看 label 是否存在；label/收益缺失的股票按保守值结算。

### L2 — 停牌缺失 o2o 填 0（原 P1.5）

- **位置**：`scripts/backtest.py:190`、`:333`（`.fillna(0.0)`）
- **机制**：停牌当天记为"不赚不赔"，偏乐观。L1 修复后会承载更多样本，偏差进一步放大。
- **正确修法**：停牌延续持仓、次日结算；退市按 -100% 或跌幅上限计；至少打印缺失比例。

### L3 — 特征与 label 时间轴不一致（原 P2.9）

- **位置**：`data/scripts/build_feature_lib.py`（fea2 load_panel 预过滤停牌/ST）
- **机制**：特征的 rolling/pct_change 在「可交易日」序列上计算，「5天前」= 5个正常交易日前；label 在完整日历上计算，「5天后」含停牌日。停牌期间两者时间轴错位。
- **现状**：`nan_thresh=0.3` 已过滤掉大多数长期停牌股，影响局部。hs300 等流动性好的 universe 影响更小。
- **正确修法**：`build_fea2` 在完整时间轴上计算 rolling 特征，计算完再打 valid 掩码，与 label 对齐。

### L4 — 停牌填 0 与平盘 0 混淆（原 P2.12）

- **位置**：`data_provider/stock_dataset.py:178`（`np.nan_to_num(x, nan=0.0)`）
- **机制**：停牌日特征全零，但 `close/open-1 = 0` 本来就是平盘的合法值，模型无法区分。
- **正确修法**：加一列 validity_mask 特征（参考 `data_utils.py` 的 `validity_label`），显式告知模型哪些行是停牌填充。

### L5 — fea3 因子选择用全样本 IC（原 P2.8）

- **位置**：`data/scripts/select_factors.py:68`
- **机制**：直接读 yaml 里的 `cc_rank_ic` 做阈值过滤，这些指标是在包含测试期的全历史上算出来的，因子选择存在未来信息。
- **现状**：当前训练均用 fea2，不影响现有结果。
- **正确修法**：只用训练期数据重算 IC 做筛选，或改成滚动窗口选因子。
