# 修改说明 & 指标解读

分两部分：**Part A** 是需要落地的代码改动，**Part B** 是各项指标该怎么看。

配套文件：

| 文件 | 放置位置 | 状态 |
|---|---|---|
| `iTransformer_stock.py` | `model/` | 整份替换 |
| `linear_baseline.py` | `scripts/` | 整份替换 |
| `attribution.py` | `scripts/` | 新增 |
| `build_fea3_v5_770.py` | `data/scripts/` | 整份替换（此前已给） |
| `run_stock.py` | `scripts/` | 按下方补丁改 |
| `backtest.py` | `scripts/` | 按下方补丁改 |

---

# Part A · 代码改动

## A1. `run_stock.py`

### A1.1 MSE 形状（必改，会静默算错）

新模型 `forward` 返回 `(B,)`，`stock_collate_fn` 返回 `(B,1)`。
`nn.MSELoss()` 遇到这两个形状会**广播成 `(B,B)`**，只发一条 UserWarning，
照常返回一个数字。实测 5 个样本时是 1.438 vs 正确的 0.661。

`neg_pearson` 内部有 `.squeeze()` 所以不受影响 —— 结果是
`--loss rankic` 正常，`--loss mse` / `combined` 悄悄算错。

```python
def compute_loss(pred, target, mode: str, ic_weight: float) -> torch.Tensor:
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    assert pred.shape == target.shape, (pred.shape, target.shape)

    if mode == "rankic":
        return neg_pearson(pred, target)
    elif mode == "combined":
        return mse_loss_fn(pred, target) + ic_weight * neg_pearson(pred, target)
    else:
        return mse_loss_fn(pred, target)
```

### A1.2 接上 `--head_type`（必改，否则新功能不生效）

现在只有 `--class_strategy`，默认 `"mean"`。新模型的兼容逻辑会把它映射成
`head_type="mean"`，**gate 根本不会启用**。

```python
parser.add_argument("--head_type", default="gate", choices=["gate", "mean", "cls"],
                    help="pooling 方式。gate=可学习因子权重(默认)，mean=旧行为，cls=CLS token")
```

`--class_strategy` 保留不动，模型里 `head_type` 优先级更高。

### A1.3 checkpoint 存因子列名（必改，否则会静默错位）

`variate_emb` 第 i 行绑定第 i 个因子列。换 cache、改 yaml、或某因子这次
计算失败，列的对应关系就变了，加载旧 ckpt 不报错但整套位置编码全错。

```python
torch.save({
    "state_dict":   model.state_dict(),
    "args":         vars(args),
    "feature_cols": list(train_ds.feature_cols),
    "enc_in":       args.enc_in,
}, best_ckpt)
```

### A1.4 AdamW + weight decay（建议）

低信噪比场景下 weight decay 比减参数更有效。`variate_emb`、`gate`、
LayerNorm、bias 都是 1 维参数，必须排除在 wd 之外。

```python
decay, no_decay = [], []
for n_, p_ in model.named_parameters():
    if not p_.requires_grad:
        continue
    if p_.ndim <= 1 or "variate_emb" in n_ or "gate" in n_:
        no_decay.append(p_)
    else:
        decay.append(p_)
optimizer = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": args.weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr)
```

配套加参数：

```python
parser.add_argument("--weight_decay", type=float, default=1e-2)
```

### A1.5 容量默认值（建议）

`d_model=256, d_ff=512, e_layers=2` 约 100 万参数。日频扣掉 label 重叠和
截面相关后，有效独立样本大概 1 万~10 万量级，这个容量偏大。

```python
parser.add_argument("--d_model", type=int, default=128)   # was 256
parser.add_argument("--d_ff",    type=int, default=256)   # was 512
parser.add_argument("--dropout", type=float, default=0.2) # was 0.1
```

Encoder 占总参数 90% 以上，且随 `d_model` 平方增长 —— 调它比调别处有效。
往上试比往下砍好：加容量后验证 IC 不涨了，是个清晰的停止信号。

### A1.6 训练前的必做检查

正式扫参之前，先关掉全部正则（`--dropout 0 --weight_decay 0`），只用
200 天训练集，确认能过拟合到 train RankIC > 0.3。

- **过不去** → 容量或优化有问题，此时减小模型只会更糟
- **能过去** → 容量够了，剩下纯粹是正则化和数据的问题

五分钟能省掉大量瞎调参。

---

## A2. `backtest.py`

### A2.1 恢复 `head_type`（必改）

第 116 行从 `config.json` 恢复配置的 key 列表要加上 `head_type`，否则回测
用的是默认值而非训练时的设置：

```python
for k in ["d_model", "n_heads", "e_layers", "d_ff", "dropout", "mlp_hidden",
          "class_strategy", "head_type", "embed", "freq", "factor", "activation"]:
```

### A2.2 校验因子列（必改）

第 195 行附近，`load_state_dict` 之前：

```python
if "feature_cols" in raw_ckpt:
    ck_cols = list(raw_ckpt["feature_cols"])
    ds_cols = list(test_ds.feature_cols)
    assert ck_cols == ds_cols, (
        f"特征列与训练时不一致，variate_emb 会错位。\n"
        f"  ckpt 有而 cache 无: {set(ck_cols) - set(ds_cols)}\n"
        f"  cache 有而 ckpt 无: {set(ds_cols) - set(ck_cols)}")
```

### A2.3 `o2o` 缺失不要填 0（必改）

第 238 行 `df["o2o"] = df["o2o"].fillna(0.0)` 等于假设那天持仓不涨不跌。
退市、长期停牌的票会被静默地当成零收益持有，对尾部票有系统性乐观偏差。

```python
n_miss = df["o2o"].isna().sum()
if n_miss:
    print(f"  [WARN] {n_miss}/{len(df)} 条无 o2o（退市/停牌），剔除而非填 0")
df = df.dropna(subset=["o2o"]).reset_index(drop=True)
```

### A2.4 分层/多空要过滤涨跌停（必改）

第 686-706 行的分层和 Long-Short 用的是原始 `o2o`，**完全没走 TopkDropout
那套 `buyable`/`sellable` 过滤**。零成本 + top30 全额日换仓 + 不管涨跌停，
在 A 股上必然不可实现 —— 之前那张图里 Long-Short 冲到 10.8 倍就是这么来的。

```python
for date, grp in df.groupby("date"):
    if len(grp) < args.n_groups * 2:
        continue
    grp = grp.sort_values("pred", ascending=False).reset_index(drop=True)
    n = len(grp)
    for g in range(args.n_groups):
        lo, hi = int(g * n / args.n_groups), int((g + 1) * n / args.n_groups)
        group_rets[g].append(grp.iloc[lo:hi]["o2o"].mean())

    # 多空只在可交易样本上取，且明确这是「上限」而非可实现收益
    tradable = grp[grp["buyable"]] if "buyable" in grp.columns else grp
    if len(tradable) < 2 * BASELINE_TOPK:
        continue
    nl = min(BASELINE_TOPK, len(tradable) // 4)
    ls_rows.append({
        "date":      date,
        "long_ret":  tradable.iloc[:nl]["o2o"].mean(),
        "short_ret": tradable.iloc[-nl:]["o2o"].mean(),
    })
```

图标题也改一下，避免误读：

```python
ax.set_title("因子层多空净值（o2o，无成本、无冲击 —— 仅作单调性诊断，非可实现收益）")
```

### A2.5 多基准（建议）

`--benchmark` 改成可传多个，或直接用 `attribution.py`（下节）。最省事的
做法是保留 `backtest.py` 原样，跑完之后单跑一次 `attribution.py`。

### A2.6 滑点默认值（建议）

`--slippage_bps` 默认 0，且帮助文本已经写明这很乐观。如果 `attribution.py`
显示持仓偏小市值，把默认改成 30，评估真实收益时用 50。

---

## A3. `visualize_attention.py`

第 60 行构造 cfg 时没有 `enc_in`，新模型会直接 `AttributeError`：

```python
cfg = SimpleNamespace(
    ...,
    enc_in=len(np.load(os.path.join(cache_dir, "feature_cols.npy"), allow_pickle=True)),
    head_type="gate",
)
```

第 73 行 `strict=False` 会静默忽略缺失参数，加载旧 ckpt 时 `variate_emb`
会停在全 0（位置编码失效）。至少打印出来：

```python
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"[WARN] 未加载（将用初始值）: {missing}")
if unexpected:
    print(f"[WARN] ckpt 中多余的键: {unexpected}")
```

---

## A4. 数据脚本

`build_fea3_v5_770.py` 已单独给过，三处关键修正：

1. **ST/停牌过滤移到因子计算之后** —— 旧版在算因子前删行，`rolling`/
   `pct_change` 在被挖洞的序列上运算，停牌 10 天的票会算出一个 10 日累计
   涨幅当作单日收益，动量/波动率类因子全部污染
2. **先 winsorize 再 z-score** —— 旧版顺序反了，pe/pb 的极值先污染了
   mean/std 才去 clip，整个因子当天被压平
3. **market_return 用 `index.map` 并加断言** —— 旧版 `reindex().to_numpy()`
   一旦错位是静默的

`build_label_lib.py` 还有一处未修：`open_.shift(-1)` 是在**每只股票自己的
交易日序列**上做的。一只票停牌 10 天，`shift(-1)` 拿到的是 10 天后的开盘价，
却被当成 1 日收益。`backtest.py` 的 `o2o` 同理。这会给停复牌的票造出巨大假收益，
而这些票恰恰容易被小市值信号选中。修法是按自然交易日重建索引后再 shift，
或直接剔除跨停牌的样本。

---

## A5. 执行顺序

```bash
# 1. 重建特征（含三处修正）
python data/scripts/build_fea3_v5_770.py
python data/scripts/build_cache.py --fea fea3 --fea3_file fea3_v5_770.parquet \
       --label ret_5d_avg_open

# 2. 先跑 baseline —— 这一步决定后面几周该往哪使劲
python scripts/linear_baseline.py --cache_dir data/cache/... \
       --feat_win 30 --horizon 5

# 3. 训练
python scripts/run_stock.py --cache_dir data/cache/... \
       --head_type gate --d_model 128 --d_ff 256 --dropout 0.2 --loss rankic

# 4. 归因（关键）
python scripts/attribution.py --bt_dir backtest_results/<your_run>
```

第 2 步和第 4 步比第 3 步重要。

---

# Part B · 指标怎么看

## B1. 先看归因，再看收益

`attribution.py` 的输出决定其余指标是否值得看。

### 双因子回归

```
r_port = alpha + beta_MKT·全A等权 + beta_SMB·(最小市值组 − 最大市值组) + ε
```

| 看什么 | 判据 |
|---|---|
| 年化 alpha | 扣掉市场和市值风格后剩下的部分 |
| **t_NW** | **绝对值 < 2 就不能认为 alpha 显著** |
| beta_SMB | 显著为正 = 在赚小市值风格的钱，不是选股 |
| alpha 占总收益比 | < 30% 说明策略主体是风格暴露 |

`t_NW` 用 Newey-West(lag=10) 修正了重叠 label 造成的自相关。普通 t 值在
5 日重叠下会高估约 √6 ≈ 2.4 倍。

### 中性化 IC

预测值先对 `log(流通市值)` 和行业做截面回归取残差，再算 IC。

| 衰减幅度 | 含义 |
|---|---|
| > 50% | 信号主体就是市值因子，换什么模型架构都没意义 |
| 25~50% | 市值贡献可观，排序前应该做中性化再回测 |
| < 25% | IC 主要来自其他信息，可以继续优化模型 |

**这是整套诊断里最关键的一个数。** 半天能跑完，但它决定后续所有工作的方向。

### 多基准净值图

第二个子图是「策略 / 各基准」的相对净值。分母从上证换成最小市值组之后，
如果曲线拉平甚至向下，说明所谓超额就是小盘 beta。

---

## B2. 因子质量指标

### IC / RankIC

- **IC** = Pearson(pred, 收益)，量纲敏感，受极端值影响大
- **RankIC** = Spearman，只看排序，日频选股应以这个为准

| RankIC | 判断 |
|---|---|
| < 0.02 | 基本没有信号 |
| 0.02 ~ 0.04 | 弱但可用，需要低成本执行 |
| 0.04 ~ 0.06 | 不错 |
| > 0.08 | **先怀疑数据泄漏**，日频截面很难持续做到 |

### 单日 RankIC 的合理范围

零假设下，N 只股票的日 RankIC 标准误约 `1/√N`：

| N | SE | 3σ |
|---|---|---|
| 500 | 0.045 | 0.134 |
| 1500 | 0.026 | 0.077 |
| 3000 | 0.018 | 0.055 |

之前那张图里 2026-07 有几天 RankIC 到 0.4~0.55 —— 在 N=3000 下是 20~30 个
标准差。真实的日频选股能力做不到这个。两种可能：当天有效股票数暴跌
（大面积停牌导致 N 只剩几十），或者预测高度暴露于市值而市值因子当天出现
极端定向移动。`attribution.py` 第 4 节会直接列出这些天的 N 和 z 值。

### ICIR / RankICIR

`IC均值 / IC标准差`。**含重叠 label 时会高估约 √(horizon+1) 倍**，因为
逐日 IC 序列高度自相关。你的代码注释里已经写明了这点，但报数时容易忘。

对外报数用 `attribution.py` 的 `t_NW`，不要用 `ICIR × √n`。

### 分层单调性

G1 > G2 > G3 > G4 > G5 且首尾价差大，说明信号有排序能力。之前那次
（30.34 / 22.02 / 18.09 / 11.97 / -13.53）单调性很干净，这是好信号。

但要注意：**市值因子本身就是单调的**，所以单调性漂亮不能排除市值暴露。
还是要看中性化后的结果。

---

## B3. 策略层指标

### Sharpe 的可信区间

Sharpe 的标准误约 `√(1/T年)`：

| 回测长度 | SE | Sharpe=1.7 的 95%CI |
|---|---|---|
| 1.3 年 | 0.88 | [0.0, 3.5] |
| 3 年 | 0.58 | [0.5, 2.9] |
| 5 年 | 0.45 | [0.8, 2.6] |

**1.3 年的测试期，Sharpe 1.74 的置信区间包含 0。** 这不是说策略不好，
而是说这个样本量下无法区分 1.74 和 0。

### 分年结果的割裂比总数重要

```
2025: 年化 +91.08%  Sharpe 3.58
2026: 年化  -1.53%  Sharpe 0.05
```

整个业绩由前 8 个月贡献。这种形态几乎总是意味着**风格暴露而非稳定 alpha**：
好的那段刚好赶上顺风，风格一切换就归零。

判断方法：把好的那段和差的那段分别做归因回归，看 beta_SMB 是否都显著。
如果是，那就是风格。

### 换手与成本

`持有期 ≈ topk / n_drop`。topk=30, n_drop=3 → 约 10 天。

年化换手 51.7 倍，成本 7.47ppt/年 —— 这是按 12/17bps 算的。要注意：

- 以 T+1 开盘价**全额成交**本身就是乐观假设
- 小微盘 30 只集中持仓，单边冲击成本常态是 50~100bps，不是 12bps
- `attribution.py` 会打印 Top30 持仓的市值中位数，明显偏小就必须重估

用 `--slippage_bps 50` 重跑一遍，看净值还剩多少。如果收益大部分被吃掉，
那就该降换手（提高 n_drop 对应的持有期）或者限制股票池市值下限。

### InfoR

信息比率 = 超额收益 / 跟踪误差。**完全取决于基准选得对不对。**

用上证指数当基准，小盘策略的 InfoR 会虚高。换成中证 2000 或
`attribution.py` 里的最小市值组，才是有意义的数。

---

## B4. 模型层诊断

### `model.factor_weights()`

`head_type="gate"` 时可以直接读出学到的因子权重：

```python
w = model.factor_weights(feature_names=ds.feature_cols)
print(w.head(15))
```

三种形态：

- **接近均匀**（每个 ≈ 1/F）→ gate 没学到东西，等价于 mean pool。
  说明 pooling 不是瓶颈，问题在别处
- **少数几个占大头** → 正常。和 `linear_baseline.py` 打印的单因子 IC
  排序对照，排序差异很大通常是数据问题（某列错位、NaN 率过高）
- **塌到一个** → 回去查数据

### attention map

如果跨 F 的 attention 接近均匀，说明 cross-variate attention 没学到因子
交互，iTransformer 相对「因子加权平均 + MLP」没有优势。那时候该跟 GRU
基线比一比，而不是继续调 head。

### 基线对照表

`linear_baseline.py` 会把前三行填好，你需要补上后面几行：

| 模型 | 参数量 | 看时序 | 非线性 | val RankIC |
|---|---|---|---|---|
| 符号对齐等权 | 0 | ✗ | ✗ | |
| IC 加权 | 0 | ✗ | ✗ | |
| Ridge (seq=1) | F | ✗ | ✗ | |
| Ridge (seq=30) | 30F | ✓线性 | ✗ | |
| GRU | ~25k | ✓ | ✓ | |
| iTransformer | ~100k | ✓ | ✓ | |

读法：

- **全都差不多** → 信号全在因子里，去做因子而不是调模型
- **iTransformer ≈ Ridge** → 时序和非线性都没带来东西，`seq_len` 可以砍到 1
- **iTransformer 明显更高** → 架构有价值，继续优化

注意公平性：给 iTransformer 扫 20 组超参、给基线跑 1 组默认值，这个比较
不成立。至少各扫 5 组学习率，各跑 3 个随机种子看均值和标准差 —— 日频
IC 的种子间波动可能有 ±0.01，单次结果分不出 0.035 和 0.042。

---

## B5. 一页速查

跑完一次实验，按这个顺序看：

1. **中性化 IC 衰减** > 50%？→ 是市值因子，停下来改信号，别调模型
2. **alpha 的 t_NW** < 2？→ 结果不显著，样本不够或没有 alpha
3. **分年 Sharpe** 差距巨大？→ 风格暴露，不是稳定 alpha
4. **单日 RankIC** 有超过 `3/√N` 的？→ 查那几天的 universe
5. **Top30 持仓市值** 明显偏小？→ 成本假设重估，`--slippage_bps 50` 重跑
6. **iTransformer vs 基线** 没拉开？→ 简化模型，把精力放回数据
7. 以上都通过，再看年化收益和 Sharpe
