# iTransformer-Stock

将 [iTransformer](https://arxiv.org/abs/2310.06625)（ICLR 2024 Spotlight）适配为 A 股截面收益率预测模型。

原版 iTransformer 在 time-step 维度做 attention（inverted Transformer），天然适合捕捉多因子间的相互关系。本项目将其从多步时序预测改造为单日截面因子模型，输出每只股票的预期收益率排序信号，结合 TopKDrop 策略完成组合回测。

---

## 项目结构

```
iTransformer/
├── run_stock.py              # 训练入口
├── backtest.py               # 回测（TopKDrop，双 drop 对比）
├── build_cache.py            # 构建 numpy 缓存（特征+label）
├── select_factors.py         # Alpha191 因子筛选
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
│       ├── build_feature_lib.py
│       ├── build_label_lib.py
│       └── compute_fea3_fullmkt.py
│
├── checkpoints/              # 模型检查点
├── backtest_results/         # 回测结果（图表 + CSV）
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

### 1. 构建缓存

```bash
# HS300 + fea2 + 5日开开收益
python data/scripts/build_cache.py --fea fea2 --label ret_5d_open --universe hs300

# 全市场 + fea2 + 5日开开收益
python data/scripts/build_cache.py --fea fea2 --label ret_5d_open --universe all
```

缓存保存至 `data/cache/cache_{tag}/`，包含：
- `feat_arr.npy`：`(T, S, F)` float32 特征数组
- `label_arr.npy`：`(T, S)` 标签
- `close_arr.npy`：`(T, S)` 原始收盘价
- `dates.npy` / `stocks.npy` / `feature_cols.npy`：坐标索引
- `universe_mask.npy`：`(T, S)` 点时间成分股掩码（非全市场时）
- `meta.json`：构建参数记录

### 2. 训练

```bash
nohup python scripts/run_stock.py \
  --cache_dir data/cache/cache_fea2_hs300_ret5do \
  --seq_len 30 --horizon 5 \
  --d_model 256 --n_heads 4 --e_layers 2 --d_ff 512 \
  --epochs 20 --lr 1e-4 \
  --loss combined --ic_weight 10 \
  --train_start 2018-04-24 --train_end 2024-04-23 \
  --val_start   2024-04-24 --val_end   2025-04-23 \
  --test_start  2025-04-24 --test_end  2026-08-14 \
  --ckpt_dir checkpoints/my_run \
  > logs/my_run.log 2>&1 &
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache_dir` | `data/cache/cache_fea2_ret5do` | 缓存目录 |
| `--seq_len` | 30 | 历史窗口（交易日） |
| `--horizon` | 10 | 预测窗口，用于 label embargo |
| `--d_model` | 256 | Transformer 嵌入维度 |
| `--n_heads` | 4 | 注意力头数 |
| `--e_layers` | 2 | Encoder 层数 |
| `--d_ff` | 512 | FFN 隐层维度 |
| `--mlp_hidden` | 64 | Head MLP 隐层维度 |
| `--loss` | `combined` | `mse` / `rankic` / `combined` |
| `--ic_weight` | 10 | combined loss 中 IC 项权重 |
| `--class_strategy` | `mean` | `mean`（均值池化）/ `cls_token` |

### 3. 回测

```bash
nohup python scripts/backtest.py \
  --ckpt checkpoints/my_run/best.pt \
  --cache_dir data/cache/cache_fea2_hs300_ret5do \
  --topk 30 --n_drop 6 \
  --out_dir backtest_results/my_run \
  > logs/backtest_my_run.log 2>&1 &
```

回测策略：TopKDrop，每日持有预测收益率最高的 `topk` 只股，每日最多换仓 `n_drop` 只。
时点对齐：T 日信号 → T+1 开盘买入 → T+2 开盘结算。
baseline（topk=30 drop=3）每次都自动运行，`--topk/--n_drop` 指定 custom 组进行对比。

---

## 数据流

```
panel.parquet（BaoStock原始数据）
    ↓
data/scripts/build_feature_lib.py   →  data/feature_lib/*.parquet
data/scripts/build_label_lib.py     →  data/feature_lib/label_lib.parquet
    ↓
select_factors.py                   →  data/selected_factors.txt（Alpha191筛选）
    ↓
build_cache.py                      →  data/cache/cache_*/（numpy缓存）
    ↓
run_stock.py                        →  checkpoints/*/best.pt
    ↓
backtest.py                         →  backtest_results/*/
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

## 实验结果

测试期：2025-04-24 ~ 2026-08-14（315 个交易日）
回测成本：买 12bps / 卖 17bps；基准：沪深300

> 回测框架固定跑两组：baseline（topk=30 drop=3）与 custom（由 `--topk/--n_drop` 指定）。

### HS300 模型（fea2 + ret_5d_open）

测试集 IC=-0.019 / RankIC=-0.006 / RankICIR=-0.06

| 指标 | 2025（baseline） | 2026（baseline） | 全测试期（baseline） | 全测试期（drop=6） |
|------|-----------------|-----------------|---------------------|-------------------|
| 年化收益（净） | +3.7% | -11.1% | -3.4% | -13.7% |
| Sharpe | 0.37 | -0.75 | -0.20 | -0.98 |
| 最大回撤 | -7.4% | -13.5% | -16.8% | -23.1% |
| 月胜率 | 55.6% | 50.0% | 52.9% | 29.4% |
| 年化换手 | 51.6x | 50.4x | 51.0x | 100.8x |

验证集 RankIC：~0.033

### 全市场模型（fea2 + ret_5d_open）

测试集 IC=0.047 / RankIC=0.072 / RankICIR=0.48

| 指标 | 2025（baseline） | 2026（baseline） | 全测试期（baseline） | 全测试期（drop=6） |
|------|-----------------|-----------------|---------------------|-------------------|
| 年化收益（净） | +62.4% | +6.9% | +34.1% | +26.7% |
| Sharpe | 3.14 | 0.40 | 1.59 | 1.21 |
| 最大回撤 | -6.6% | -20.1% | -20.1% | -23.6% |
| 月胜率 | 88.9% | 62.5% | 76.5% | 76.5% |
| 年化换手 | 51.6x | 50.4x | 51.0x | 101.3x |

验证集 RankIC：~0.097

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
