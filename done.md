# 已修复清单

汇总两轮代码审查中已经修掉的问题。未修的遗留问题见 `review_new.md`。

- 第一轮：`95b72c3` → `eb75001`（P 系列）
- 第二轮：验证第一轮修复的正确性 + 可信度审查（N 系列）

---

## 第一轮：P0 —— 直接影响模型表现

### P0.1 fea2 缺截面 z-score，特征量纲相差 2~3 个数量级

`data/scripts/build_feature_lib.py`

`build_fea2` 沿用 `data_utils.py` 口径「clip + fillna(0)，不做截面 z-score」，
但模型 docstring 明确写着自己去掉了 RevIN、依赖**外部截面 z-score**，
且 `DataEmbedding_inverted` 只有一个共享的 `nn.Linear(seq_len, d_model)`，
39 个特征 token 共用同一套权重、没有 per-feature scaling。

39 列的量纲分布：

| 组 | 列数 | 量级 |
|----|-----|------|
| 价格/收益类 | 14 | ~1e-2（clip ±0.2 / ±0.5） |
| 量能类 | 4 | ~1–20 |
| 资金流 | 18 | ~20–60（`log(x+1)/rolling10_std`） |
| 估值差分 | 3 | 无界（见 P0.2） |

25/39 列比价格类大 2~3 个数量级 → 资金流 token 的 embedding 范数淹没价格 token，
attention logits 被其支配、softmax 饱和，mean-pool over F 也是资金流说话。

**修复**：`build_fea2` 末尾恢复 `cs_zscore`，与 fea1/fea3 口径一致。

### P0.2 `peTTM_diff` / `pbMRQ_diff` / `psTTM_diff` 漏 clip

`clip_dict` 沿用了 `data_utils.py` 的旧键名（`pe_ttm`/`pb`/`ps_ttm`），
而 `build_fea2` 生成的列名是 `peTTM_diff`/`pbMRQ_diff`/`psTTM_diff`，
改名后 clip 没跟上，三列全部漏过循环，只经过 `fillna(0)`。
PE TTM 在盈亏切换时一阶差分能到几千上万，之前靠 z-score 兜着。

**修复**：补 clip 区间 `peTTM_diff (-30,30)` / `pbMRQ_diff (-5,5)` / `psTTM_diff (-10,10)`。

### P0.3 collate 的 rank 归一化被删除后，`ic_weight` 未重新校准

`data_provider/stock_dataset.py`、`scripts/run_stock.py`

`stock_collate_fn` 原本把当天 label 转成 `[-1,1]` 的截面 rank，`95b72c3` 改成只做 `stack`。
但 README 的训练命令仍是 `--label ret_5d_open`（raw，std≈0.04）配 `--ic_weight 10`，
正好是代码自己 help 文本里标注的错配组合：MSE≈1.6e-3 而 IC 项≈0.5，
**IC 项是 MSE 的约 300 倍**，`combined` 实际退化成纯 `rankic`；
且 `neg_pearson` 尺度不变，MSE 唯一作用变成把 pred 往 0 压 → 方差塌缩 → 分母趋零 → 梯度不稳。

**修复**：`--ic_weight` 默认改 0.05、`--loss` 默认改 `combined`、`--horizon` 默认 10 → 5；
docstring 删除对不存在列 `ret_10d_cs_rank_sym` 的引用（见 P3.16），改为现有列的口径说明。

---

## 第一轮：P1 —— 真实的未来信息泄漏

### P1.6 三处 `bfill()` 用未来值回填

`data/scripts/build_feature_lib.py`

- `open_.shift(5).bfill()`
- `vol`/`amount` 的 `rolling(10).std().bfill()`
- 18 列资金流的 `rolling(10).std().bfill()`

前两处只影响每只股票序列开头 5/9 行；第三处更麻烦，rolling 是在 `mf_df` 自己的
日期轴上做的（`.reindex(feat.index)` 在之后才执行），中间有空洞时 NaN 会被未来的 std 填上。

**修复**：全部改为 `ffill()`。

### P1.7 zz500 / zz1000 / zz800 股票池生存偏差

`data/universe.py` 在 `date=None` 时用 `all_dates.max()`（最后一天）的市值排名，
而 `build_cache.py` 正是用 `date=None` 取 `ever_in_univ` 来裁剪 S 维度 →
只保留「在最后一天处于该市值分档」的股票，2019 年在 zz500、2026 年掉出去的股票被整体删除。

**修复**：`ever_in_univ` 改为逐月快照求并集。
（hs300/zz2000 走权重文件并集 + 逐日点时间掩码，原实现是对的。）

---

## 第一轮：P2 —— 统计口径

### P2.11 重叠 label 导致 ICIR 被系统性高估

5 日重叠 label 逐日算 IC，IC 序列高度自相关，`mean/std` 会高估约 √span 倍。

**修复**：`backtest.py` 额外打印不重叠采样的 ICIR；`run_stock.py` 加注释说明。
（第二轮 N3 进一步完善为多相位平均 + Newey-West。）

### P2.13 `ret_X_open` label 的 embargo 差一天

`data_provider/stock_dataset.py`

`ret_5d_open[i]` 实际用到 `open[i+6]`，而 embargo 只检查 `dates[i+horizon] < 下段起点`，
train 末尾最后一个样本的 label 窗口会伸进 val 第一天。

**修复**：引入 `_emb_span = horizon + 1`，embargo 判断改用
`dates_pd[min(i + _emb_span, T-1)] < emb`。

---

## 第一轮：P3 —— 会崩溃的 bug 与文档脱节

### P3.14 `build_cache.py --universe zz500/zz1000/zz800` 必崩

`panel_for_univ or pd.DataFrame()` —— 对非空 DataFrame 求布尔值会抛
`ValueError: The truth value of a DataFrame is ambiguous`，
而这个表达式恰好只在 `panel_for_univ is not None` 时求值。

**修复**：拆开三元表达式，两个分支显式处理。

### P3.15 `build_cache.py --fea fea1 fea2` 必崩

fea1 和 fea2 都有 `ret_5d` 列，`join(how="outer")` 遇到重名列且未指定 suffix 直接报错。

**修复**：join 前检测重名列并加特征集前缀。（该修复本身有 bug，见 N2。）

### P3.16 loss docstring 引用不存在的 label 列

`run_stock.py` 多处提到 `ret_10d_cs_rank_sym` / `rank_sym`，
但 `build_label_lib.py` 只生成 `{col}_cs_rank`（范围 `[0,1]`），没有任何 `_cs_rank_sym` 列。
按文档传参会直接 KeyError。

**修复**：删除引用，改为现有列的口径说明，并标注 `_cs_rank` 是 `[0,1]`、
用它做 target 时 `ic_weight` 需重新校准至约 1。

### P3.17 `data/data_utils.py` 在 import 时发起网络请求

`ts_pro = ts.pro_api("")` 在模块加载时就执行（空 token），
而 `data/` 是包目录，任何人误 import 就会触发。

**修复**：改为懒加载代理对象 `_LazyTsPro`，`import tushare` 也挪进函数体。

---

## 第二轮：N 系列 —— 修复的修复 + 回测口径

### N1 `cs_zscore` 被放在 `fillna(0)` 之后（P0.1 的修复引入了新问题）

`data/scripts/build_feature_lib.py`

原修复写成 `clip → fillna(0) → cs_zscore`。18 列资金流是 `.reindex(feat.index)` 贴上来的、
3 列估值差分来自另一张表，缺失量都很大。这些 NaN 先被填成 0，
**然后 0 参与当日截面的 mean/std**。资金流原始值是 `log(x+1)/rolling_std ≈ 20~60` 的正数，
0 属于极端离群点，会把当日均值拉低、标准差抬高；
且缺失率随年份变化（早年 moneyflow CSV 更少），等于给特征注入了虚假的时间趋势。

**修复**：改为 `clip → cs_zscore → fillna(0)`。pandas 的 `mean()`/`std()` 自动跳过 NaN，
标准化后再填 0，此时 0 恰好等于「当日截面均值」，是中性填充。

### N2 `build_cache.py` 重名列前缀逻辑写反（P3.15 的修复引入了新问题）

处理第一个特征集时 `labeled` 为空，内层循环一次不跑 → `dup = set(df.columns)`（全部列）
→ `if dup` 为真 → **fea1 的每一列都被加上 `fea1_` 前缀**。
副作用：`--fea fea1` 单独调用走单集分支不加前缀，`--fea fea1 fea2` 加前缀，
同一个 fea1 在两种调用下列名不同，缓存不可互换、ckpt 的 `enc_in` 也对不上。
另外 `zip(args.fea, feat_dfs)` 在某个特征文件缺失被 skip 时会错位，前缀贴到错误的特征集上。

**修复**：改为累积 `seen` 集合，只给「与已收录列重名」的列加前缀；
新增 `loaded_names` 只记实际加载成功的特征集名，保证 zip 不错位。
第一个特征集永远不改名，单集与多集拼合的列名保持一致。

### N3 不重叠 ICIR 只取了一个相位

`ic_df.iloc[::HORIZON]` 从第 0 天起隔 5 天采样，315 天只剩 63 点，换个起点结果就变；
且 `ret_5d_open` 的实际跨度是 6 天（`open[t+1]`→`open[t+6]`）而非 5 天。

**修复**：`LABEL_SPAN = horizon + 1`；对 span 个相位分别做不重叠采样再取均值；
补上 **Newey-West t 统计量**（Bartlett 核，lag = span−1）作为首选显著性判据；
输出里打印有效独立样本数 `N/span` 并提示「看 t_NW，不要用含重叠的 ICIR」。

> 「有效独立样本」的来历：315 个 label 的个数没错，但相邻两天共享 5/6 的行情区间，
> 相关系数约 0.83。对 h 天重叠的 MA(h−1) 结构，ρ_k = (h−k)/h，
> 方差膨胀因子 `1 + 2Σρ_k = h`，故均值标准误为 `std/√(N/h)`，
> 等效独立样本 `N_eff = N/h = 315/6 ≈ 52`。t = ICIR × √52 ≈ 3.5，
> 按 315 天算会得到 8.5。h 倍膨胀是保守上限（预测值本身每天也在变，会削弱自相关），
> 真值在 52 和 315 之间，精确判断用 Newey-West。

### N4 预测值与 (日期, 股票) 的对齐由隐式改为显式

`scripts/backtest.py`、`data_provider/stock_dataset.py`

原来是 `assert len(all_pred) == len(samples)` 然后按位置 zip。当前配置顺序确实对得上，
但依赖「`DayBatchSampler(shuffle=False)` 且日内顺序恰好等于 `samples` 顺序」这个隐式契约，
谁改了 sampler 就会静默错配（预测值配到错误的股票上），而长度断言抓不到。

**修复**：`StockDataset.__getitem__` 一并返回 `(di, si)`，
`raw_collate` 带出来直接构造 DataFrame，并加集合一致性断言。
`stock_collate_fn` 同步改为 `xs, ys, _, _ = zip(*batch)`（训练不需要 di/si）。

### N5 回测每天免费把组合拉回等权

原 `gross = mean(o2o of new)` 等于每天强制等权，但计费只覆盖集合差异（约 3 只），
27 只保留仓位的再平衡交易是免费的。

**修复**：重写 `simulate`，维护权重字典：保留仓位权重随收益自然漂移，
卖出释放的权重等分给新买入；当日无新买入时释放的权重按比例回到保留仓位
（等比缩放不改变相对权重，无需交易、不计费）。
计费改为按 `|w_new − w_pre|` 逐只计算（`cost_from_weights`），
`equal` 模式下被强制拉回等权的那部分交易现在也如实收费。
保留 `--weight_mode equal` 用于和旧结果对比。

**已用合成数据校验**：无换仓时 drift 口径精确等于「初始等权买入持有」（误差 1e-15）；
在 30 只、日波动 3%、两两相关 0.4 的设定下，equal 口径虚增 **+0.81% 年化**。
量级不到 1%，比最初估计的「几个百分点」小得多，远小于 N7 滑点的影响。

### N6 只限制买入、不限制卖出

`buyable` 只作用于新买入候选，`sells` 完全无约束。A 股跌停卖不出去很常见，尤其小市值。

**修复**：新增 `sellable`（`trade_status == 1` 且非跌停），
`topkdrop` 把不可卖的持仓从 `sells` 名单剔除、被迫继续持有；
panel 读取增加 `low` 列用于判跌停。

### N7 成本模型缺滑点

**修复**：新增 `--slippage_bps`（单边，叠加在 buy_cost/sell_cost 上）。
默认 0 以保持与历史结果可比，为 0 时打印警告。
以 T+1 开盘价（集合竞价单一价格）全额成交 30 只小盘股本身很乐观，
实盘单边 20~50bps 是常态，年化换手 51 倍意味着每年 10~25 个百分点的成本差异。

### N8 README 与代码脱节的部分清理

- 删除整节「实验结果」：那些 IC / 年化 / Sharpe 是 fea2 未做截面 z-score 时跑的，与当前代码不对应
- 参数表修正过时默认值：`--horizon` 10 → 5、`--ic_weight` 10 → 0.05
- 补充回测参数表（`--weight_mode` / `--slippage_bps` / `--benchmark`）
  及两个口径提醒（滑点默认 0 偏乐观；默认基准是上证综指，对小市值倾斜的组合会低估 beta 暴露）
- IC 统计部分注明「显著性看 `t_NW`，含重叠的 ICIR 会高估约 √span 倍」
- `run_stock.py` 测试集评估后补一行提示，说明 ICIR 的重叠高估问题
