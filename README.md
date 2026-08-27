# iTransformer-Stock

将 [iTransformer](https://arxiv.org/abs/2310.06625)（ICLR 2024 Spotlight）适配为 A 股截面收益率预测模型。

原版 iTransformer 在 time-step 维度做 attention（inverted Transformer），天然适合捕捉多因子间的相互关系。本项目将其从多步时序预测改造为单日截面因子模型，输出每只股票的预期收益率排序信号，结合 TopKDrop 策略完成组合回测。

---

## 项目结构

```
iTransformer/
├── scripts/
│   ├── run_stock.py          # 训练入口
│   └── backtest.py           # 回测（TopKDrop，baseline + custom 对比）
│
├── model/
│   └── iTransformer_stock.py # 改造版模型：(B,T,F) → (B,1)
│
├── layers/
│   ├── SelfAttention_Family.py  # FullAttention + AttentionLayer
│   ├── Transformer_EncDec.py    # Encoder / EncoderLayer
│   └── Embed.py                 # DataEmbedding_inverted
│
├── data_provider/
│   └── stock_dataset.py      # StockDataset + DayBatchSampler
│
├── data/
│   ├── universe.py           # 股票池管理（HS300/ZZ500/全市场等）
│   ├── data_utils.py         # 特征计算工具函数
│   ├── selected_factors.txt  # 筛选的 Alpha191 因子列表（20个）
│   ├── selected_factors_meta.csv
│   │
│   ├── feature_lib/          # 特征库（一次性构建，约 5.5 GB）
│   │   ├── fea1_price_basic.parquet   # 基础价格特征（截面z-score）
│   │   ├── fea2_price_new.parquet     # 量价+估值+资金流特征（39维）
│   │   ├── fea3_alpha191.parquet      # 筛选后的 Alpha191 因子（20维）
│   │   └── label_lib.parquet          # 多种收益率标签
│   │
│   ├── cache/                # numpy 缓存（build_cache.py 生成）
│   │   ├── cache_fea2_ret5do/           # 全市场 + fea2 + ret_5d_open
│   │   └── cache_fea2_hs300_ret5do/     # HS300 + fea2 + ret_5d_open
│   │
│   └── scripts/              # 一次性数据构建脚本
│       ├── build_feature_lib.py  # 构建 fea1/fea2/fea3 parquet
│       ├── build_label_lib.py    # 构建 label_lib.parquet
│       ├── build_cache.py        # 构建 numpy 缓存
│       ├── compute_fea3_fullmkt.py
│       └── select_factors.py     # Alpha191 因子筛选
│
├── checkpoints/              # 模型检查点
├── backtest_results/         # 回测结果（图表 + CSV + summary.md）
└── logs/                     # 训练日志
```

---

## 模型架构

```
输入 (B, T, F)          T=30日历史窗口，F=39个因子
    ↓
DataEmbedding_inverted  (B, F, d_model)  — 对每个因子序列做独立 embedding
    ↓
Transformer Encoder     (B, F, d_model)  — attention in 因子维度（variate-level）
    ↓
Mean pooling / CLS      (B, d_model)
    ↓
MLP Head                (B, 1)           — 预测未来收益率
```

核心思路：iTransformer 将 attention 作用于**因子维度**而非时间步，使模型能捕捉因子间的交互关系，而每个因子的时序模式由 embedding 层编码。

---

## 快速开始

### 1. 构建特征库（一次性）

```bash
# 构建 fea1 / fea2（基础价格+量价估值）
python data/scripts/build_feature_lib.py --fea 1 2

# 构建 label 库
python data/scripts/build_label_lib.py

# 构建 fea3（Alpha191，依赖 Qlib）—— 可选
python data/scripts/compute_fea3_fullmkt.py

python data/scripts/build_fea3_v5_770.py

# Alpha191 因子筛选（输出 data/selected_factors.txt）—— 构建 fea3 后运行
python data/scripts/select_factors.py
```

输出目录 `data/feature_lib/`：

| 文件 | 说明 |
|------|------|
| `fea1_price_basic.parquet` | 基础价格特征（截面 z-score） |
| `fea2_price_new.parquet` | 量价+估值+资金流，39 维 |
| `fea3_alpha191.parquet` | 筛选后的 Alpha191 因子，20 维 |
| `label_lib.parquet` | 多种收益率标签（ret_5d_open 等） |

### 2. 构建 numpy 缓存

```bash
# HS300 + fea2 + 5日开开收益
python data/scripts/build_cache.py --fea fea2 --label ret_5d_open --universe hs300

# 全市场 + fea2 + 5日开开收益
python data/scripts/build_cache.py --fea fea2 --label ret_5d_open --universe all

python data/scripts/build_cache.py \
  --fea fea3 \
  --fea3_file fea3_v5_770.parquet \
  --label ret_5d_avg_open \
  --universe all \
```

缓存保存至 `data/cache/cache_{tag}/`，包含：
- `feat_arr.npy`：`(T, S, F)` float32 特征数组
- `label_arr.npy`：`(T, S)` 标签
- `close_arr.npy`：`(T, S)` 原始收盘价
- `dates.npy` / `stocks.npy` / `feature_cols.npy`：坐标索引
- `universe_mask.npy`：`(T, S)` 点时间成分股掩码（非全市场时）
- `meta.json`：构建参数记录

### 4. 训练（训练完自动回测）

训练结束后默认自动调用 `backtest.py`，回测结果目录与 `--ckpt_dir` 同名，自动写入 `backtest_results/<ckpt_name>/`。

```bash
nohup python scripts/run_stock.py \
  --cache_dir data/cache/cache_fea2_ret5do_new \
  --seq_len 30 --horizon 5 \
  --loss combined --ic_weight 0.05 \
  --ckpt_dir checkpoints/my_run \
  --topk 30 --n_drop 6 \
  > logs/my_run.log 2>&1 &
# 训练完成后自动回测，结果写入 backtest_results/my_run/
```

加 `--no_backtest` 可跳过自动回测，只保存 checkpoint。

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache_dir` | `data/cache/cache_fea2_ret5do` | 训练+回测缓存目录（同一份） |
| `--seq_len` | 30 | 历史窗口（交易日） |
| `--horizon` | 5 | 预测窗口，用于 label embargo |
| `--d_model` | 256 | Transformer 嵌入维度 |
| `--n_heads` | 4 | 注意力头数 |
| `--e_layers` | 2 | Encoder 层数 |
| `--d_ff` | 512 | FFN 隐层维度 |
| `--mlp_hidden` | 64 | Head MLP 隐层维度 |
| `--loss` | `combined` | `mse` / `rankic` / `combined` |
| `--ic_weight` | 0.05 | combined loss 中 IC 项权重，需按 label 量纲校准 |
| `--class_strategy` | `mean` | `mean`（均值池化）/ `cls_token` |
| `--no_backtest` | — | 加上则跳过自动回测 |
| `--backtest_cache_dir` | 同 `--cache_dir` | 仅在回测用不同股票池时指定（如训练全市场、回测 HS300） |
| `--topk` | — | 回测 custom 组 TopK（不传只跑 baseline k30d3） |
| `--n_drop` | — | 回测 custom 组 Drop 数 |
| `--slippage_bps` | 0 | 回测单边滑点（bps） |
| `--weight_mode` | `drift` | 回测权重口径（`drift` 正确口径 / `equal` 旧口径） |

### 5. 单独回测（可选）

自动回测通常已够用。需要单独重跑或调整参数时：

```bash
nohup python scripts/backtest.py \
  --ckpt checkpoints/my_run/best.pt \
  --cache_dir data/cache/cache_fea2_ret5do_new \
  --out_dir backtest_results/my_run \
  > logs/backtest_my_run.log 2>&1 &
```

回测策略：TopKDrop，每日持有预测收益率最高的 `topk` 只股，每日最多换仓 `n_drop` 只。
时点对齐：T 日信号 → T+1 开盘买入 → T+2 开盘结算。
baseline（topk=30 drop=3）每次都自动运行，`--topk/--n_drop` 指定 custom 组进行对比。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weight_mode` | `drift` | `drift`：持仓权重随收益漂移，只对实际换仓计费（正确口径）；`equal`：每日拉回等权（旧口径，等于免费日频再平衡） |
| `--slippage_bps` | 0 | 单边滑点（bps），叠加在 `buy_cost`/`sell_cost` 上 |
| `--buy_cost` / `--sell_cost` | 0.0012 / 0.0017 | 佣金+印花税 |
| `--benchmark` | `sh.000001.csv` | 基准指数 CSV |

> **两个口径提醒**：
> 1. `--slippage_bps` 默认 0，且假设 T+1 开盘价（集合竞价单一价格）全额成交，
>    净收益偏乐观。年化换手约 51 倍，单边 20bps 滑点就是每年 10 个点以上的差异，
>    评估真实可实现收益务必设非 0 值。
> 2. 默认基准是**上证综指**。全市场 + topk=30 等权必然重仓小市值，
>    用上证综指或沪深300 都会低估 beta 暴露，超额收益里混了 size 因子。
>    做归因请另外加同风格基准（全市场等权 / 中证2000）对照。

IC 统计部分会同时输出含重叠的 ICIR、多相位不重叠 ICIR 和 Newey-West t 统计量。
**判断信号是否显著请看 `t_NW`（|t| > 2）**，含重叠的 ICIR 会高估约 √span 倍。

---

## 数据流

```
panel.parquet（BaoStock原始数据）
    ↓
data/scripts/build_feature_lib.py   →  data/feature_lib/fea{1,2,3}_*.parquet
data/scripts/build_label_lib.py     →  data/feature_lib/label_lib.parquet
    ↓
data/scripts/select_factors.py      →  data/selected_factors.txt（Alpha191筛选，fea3依赖）
    ↓
data/scripts/build_cache.py         →  data/cache/cache_*/（numpy缓存）
    ↓
scripts/run_stock.py                →  checkpoints/*/best.pt
    ↓ （训练完自动触发，out_dir = backtest_results/<ckpt_name>/）
scripts/backtest.py                 →  backtest_results/*/{backtest.png,exec_metrics.csv,summary.md}
```

---

## 特征集说明

| 特征集 | 维度 | 内容 |
|--------|------|------|
| `fea1` | 基础价格特征 | 开高低收量、移动均线、价格动量，截面 z-score |
| `fea2` | 量价+估值+资金流 | 39维，含换手率、振幅、PB/PE 估算、资金流强度等 |
| `fea3` | Alpha191 | 筛选后的 20 个日频 Alpha 因子（IC > 0.025，ICIR > 0.15） |

### Label 类型

| Label | 定义 | 用途 |
|-------|------|------|
| `ret_5d_open` | log(open_{t+6}/open_{t+1}) | 主训练目标（开开收益，对齐回测结算） |
| `ret_10d_log` | log(close_{t+10}/close_t) | 经典10日持有收益 |
| `ret_1d_open` | log(open_{t+2}/open_{t+1}) | 日度结算（回测内部使用） |

---

## 股票池

`data/universe.py` 支持以下股票池，均为点时间历史成分股（无未来函数）：

| 名称 | 说明 |
|------|------|
| `all` | 全市场（排除 ST、停牌） |
| `hs300` | 沪深300（季度快照） |
| `zz500` | 中证500（流通市值301-800名） |
| `zz1000` | 中证1000（流通市值801-1800名） |
| `zz800` | 沪深300 + 中证500 |
| `zz2000` | 中证2000（2023年后数据完整） |

---

## 依赖

```bash
pip install torch numpy pandas scipy matplotlib reformer-pytorch einops
```

数据依赖：
- [BaoStock](http://baostock.com) 行情数据（`/root/dmd/BaoStock/panel.parquet`）
- Qlib 因子库（仅 fea3 构建时需要）

---

## 参考

- [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)（ICLR 2024）
- [原始代码仓库](https://github.com/thuml/iTransformer)
