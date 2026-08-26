# 遗留问题清单

已修复的问题见 `done.md`。本文件只保留**尚未修复**的问题，按重要程度排序。

---

## U1 — 单种子、单时间切分，无法区分 alpha 与运气

`run_stock.py` 的 `SEED = 42` 从未变过，没有多种子重复实验。
`TODO.md` 里也写了「nn 有随机性，重训不稳定性」——
那么任何单次的 RankIC 都是 n=1 的观测，无法区分真实 alpha 和种子彩票。
时间上也只有一组固定切分（train 2018-04 ~ 2024-04 / val 一年 / test 一年多），
测试期正好压在特定行情上。

**建议**：至少 5 个种子跑同一配置，报 RankIC 的均值和标准差；
时间上做 walk-forward（滚动重训）而不是单一切分。
**这一步的结果决定现有结论到底站不站得住，优先级高于继续修代码。**

---

## U2 — 收益归因缺失，分不清 alpha 与 size beta

全市场 universe + topk=30 等权，必然重仓小市值，但回测里没有任何风格归因。
`--benchmark` 默认是上证综指，对一个向小市值倾斜的组合来说，
用上证综指或沪深300 做基准都会低估 beta 暴露，超额收益里混了 size 因子。

**建议**：
- 加同风格对照基准（全市场等权、中证2000），看超额还剩多少
- 对 pred 做市值/行业中性化后再回测
- 输出持仓的市值分布，直接看组合到底压在哪个市值档

---

## U3 — 回测候选池按「label 非 NaN」筛选（前视选择偏差）

`data_provider/stock_dataset.py` 的过滤条件含 `~np.isnan(label_arr[di])`，
而 `ret_5d_open[T]` 需要 `open[T+6]`，等价于**要求股票未来 6 天还在正常交易**。
未来几天停牌/退市的股票被静默剔除 —— 恰好剔掉最容易出事的那批。
`backtest.py` 直接复用了这个 samples 当候选池。

训练侧这样过滤是合理的（label 缺失确实学不了），**回测侧不该这样**。
代码里已加注释标记位置。

**正确修法**：回测独立构造候选池，只用 T 日及以前的 `trade_status`/`is_st`/上市天数过滤，
不看 label 是否存在；label/收益缺失的股票保留在池中按保守值结算。
这需要重构 `backtest.py` 的候选池构建，改动较大，暂未做。

---

## U4 — 缺次新股与流动性过滤

`nan_thresh=0.3` + `seq_len=30` 意味着只要有 21 个交易日历史就能入池。
次新股在 A 股收益极端且实际买不到（开板后连续涨停），
而 `buyable` 是用 T 日涨停代理 T+1，对开板行情捕捉不足。

**建议**：加「上市 ≥ 60 或 120 个交易日」和「20 日均成交额下限」两个过滤。

---

## U5 — 停牌缺失 o2o 填 0

`scripts/backtest.py` 的 `df["o2o"].fillna(0.0)`，停牌当天记为不赚不赔，偏乐观。
U3 修好后候选池会承载更多缺失样本，这个偏差会放大。

**正确修法**：停牌延续持仓、次日结算；退市按跌幅上限或 -100% 计；
至少要把缺失比例和它对净值的贡献单独打印出来。

---

## U6 — `summary_stats` 自带「去最差月」指标

`ex_worst_ann` / `ex_worst_sh` / `ex_worst_mdd` 会算出来并写进 `summary.md`。
事后剔除最差月份的收益不能作为决策依据，报告框架默认带这个指标容易自我误导。
保留无妨，但不要拿它做判断。

---

## U7 — 模型选择在很小的验证集上做了 20 次

`run_stock.py` 按 val RankIC 选 best epoch，`epochs=20`，
而 val 只有约 240 天（重叠 label 下有效独立样本约 40 个）。
选出来的 val RankIC 是 20 次里的最大值，是有偏的上界，选出的 epoch 很可能是噪声。

**建议**：验证集指标改用多个 epoch 的平均或平滑值来选，
或者把 val 拉长（相应缩短 train 也无妨，train 有 6 年）。

---

## U8 — 特征与 label 的时间轴不一致

`data/scripts/build_feature_lib.py` 的 `load_panel` 先过滤
`trade_status == 1 & is_st == 0`，之后才 groupby 算 `pct_change(5)` / `rolling(20)`，
所以「5 天前」= 5 个正常交易日前；而 `build_label_lib.py` 明确不过滤、用完整日历，
「5 天后」含停牌日。停牌期间两者错位。

**正确修法**：在完整时间轴上算滚动特征，算完再打 valid 掩码，与 label 对齐。

---

## U9 — 停牌填 0 与平盘 0 混淆

`data_provider/stock_dataset.py` 的 `np.nan_to_num(x, nan=0.0)` 让停牌日成为全零向量，
但 `close/open-1 = 0` 本来就是平盘的合法值，模型区分不了两者。
`nan_thresh=0.3` 允许窗口里最多 30% 是这种全零行。

**正确修法**：加一列 validity_mask 特征（参考 `data_utils.py` 的 `validity_label`），
显式告知模型哪些行是停牌填充。

---

## U10 — fea3 因子选择用全样本 IC

`data/scripts/select_factors.py` 直接读 yaml 里的 `cc_rank_ic` / `cc_rank_icir` 做阈值过滤，
这些指标几乎肯定是在包含测试期的全历史上算出来的 → 因子选择泄漏。
当前训练都用 fea2，暂不影响；一旦上 fea3 就会生效。

**正确修法**：只用训练期数据重算 IC 做筛选，或改成滚动窗口选因子。

---

## U11 — 两处遗留小问题

- `data_provider/stock_dataset.py` 的样本有效性判据是 `i + _horizon < T`，
  对 `_open` 系列 label 应为 `i + _horizon + 1 < T`。
  不是泄漏（超界时 label 本身是 NaN，会被 `valid_label` 过滤掉），只是判据不精确。
- 比例划分分支（`train_ratio`/`val_ratio`）的 embargo 仍用 `_horizon` 而非 `_emb_span`。
  该分支实际不用（都走绝对日期划分）。

---

## 建议的下一步

1. 用修好的 `build_feature_lib.py` 重建 fea2 → 重建 cache → 重训
   （`done.md` N1 会实质改变输入分布，必须重跑）
2. 回测跑两组对比：`--slippage_bps 0` 与 `--slippage_bps 20`，看净收益对滑点的敏感度
3. **5 个种子重训同一配置，看 RankIC 的分散度（U1）** —— 这一步决定现有结论是否成立
4. 加同风格基准做归因（U2），确认超额不是纯 size beta
