"""
股票收益率预测训练入口
用法:
  nohup python scripts/run_stock.py \
    --cache_dir data/cache/cache_fea2_ret5do_0826 \
    --loss combined --ic_weight 0.05 \
    --ckpt_dir checkpoints/my_run \
    > logs/my_run.log 2>&1 &
"""

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr

from data_provider.stock_dataset import StockDataset, stock_collate_fn, DayBatchSampler
import importlib


# ── 固定随机种子 ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── 参数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

# 数据
parser.add_argument("--cache_dir",        default="data/cache/cache_fea2_ret5do")
parser.add_argument("--seq_len",   type=int, default=30)
parser.add_argument("--horizon",   type=int, default=5)
# 绝对日期划分（推荐）
parser.add_argument("--train_start", default="2018-04-24")
parser.add_argument("--train_end",   default="2024-04-23")
parser.add_argument("--val_start",   default="2024-04-24")
parser.add_argument("--val_end",     default="2025-04-23")
parser.add_argument("--test_start",  default="2025-04-24")
parser.add_argument("--test_end",    default="2026-08-14")
# 比例划分（fallback，仅在不指定日期时生效）
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio",   type=float, default=0.15)

# 模型
parser.add_argument("--d_model",       type=int,   default=256)
parser.add_argument("--n_heads",       type=int,   default=4)
parser.add_argument("--e_layers",      type=int,   default=2)
parser.add_argument("--d_ff",          type=int,   default=512)
parser.add_argument("--dropout",       type=float, default=0.1)
parser.add_argument("--mlp_hidden",    type=int,   default=64)
parser.add_argument("--class_strategy", default="mean")   # mean / cls_token
parser.add_argument("--embed",  default="fixed")
parser.add_argument("--freq",   default="b")
parser.add_argument("--factor", type=int, default=1)
parser.add_argument("--activation", default="gelu")
parser.add_argument("--output_attention", action="store_true")

# 训练
parser.add_argument("--epochs",        type=int,   default=20)
parser.add_argument("--batch_size",    type=int,   default=512)
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--patience",      type=int,   default=5)
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--loss",          default="combined",  help="mse / rankic / combined")
parser.add_argument("--ic_weight",    type=float, default=0.05,
                    help="combined loss 的 IC 项权重，需按 label 量纲校准：\n"
                         "  raw ret_5d_open (std≈0.04): MSE~0.003, |IC|~0.05 → ic_weight≈0.05\n"
                         "  rank_sym ∈[-1,1]:           MSE~1.0,   |IC|~0.1  → ic_weight≈10")
parser.add_argument("--ckpt_dir",      default="checkpoints/stock")
parser.add_argument("--model",         default="iTransformer_stock", help="model 模块名，如 iTransformer_stock_1")
parser.add_argument("--gpu",           type=int, default=None, help="指定 GPU 编号，如 --gpu 1；不传则使用默认 cuda 设备")

# 自动回测
parser.add_argument("--no_backtest",   action="store_true", help="训练完成后不自动执行回测")
parser.add_argument("--backtest_cache_dir", default=None,
                    help="回测用 cache_dir，不传则复用 --cache_dir")
parser.add_argument("--topk",          type=int, default=None, help="回测 TopK，不传则只跑 baseline k30d3")
parser.add_argument("--n_drop",        type=int, default=None, help="回测 Drop 数")
parser.add_argument("--slippage_bps",  type=float, default=0.0, help="回测单边滑点（bps）")
parser.add_argument("--weight_mode",   default="drift", choices=["drift", "equal"], help="回测权重口径")

args = parser.parse_args()
os.makedirs(args.ckpt_dir, exist_ok=True)
Model = importlib.import_module(f"model.{args.model}").Model

# 训练开始时立即保存运行配置，方便事后溯源
import json
_config = {
    "args": vars(args),
    "cmd": "python " + " ".join(sys.argv),
}
with open(os.path.join(args.ckpt_dir, "config.json"), "w") as _f:
    json.dump(_config, _f, indent=2, ensure_ascii=False)

if args.gpu is not None:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 数据集 ──────────────────────────────────────────────────────────────────
dataset_kwargs = dict(
    cache_dir        = args.cache_dir,
    seq_len          = args.seq_len,
    horizon          = args.horizon,
    train_start      = args.train_start,
    train_end        = args.train_end,
    val_start        = args.val_start,
    val_end          = args.val_end,
    test_start       = args.test_start,
    test_end         = args.test_end,
    train_ratio      = args.train_ratio,
    val_ratio        = args.val_ratio,
)
train_ds = StockDataset(flag="train", **dataset_kwargs)
val_ds   = StockDataset(flag="val",   **dataset_kwargs)
test_ds  = StockDataset(flag="test",  **dataset_kwargs)

# DayBatchSampler: 一个 batch = 一整天全部有效股票，截面归一化才真实有效
# batch_size 参数在此模式下不生效（每天股票数不固定），保留仅供参考
train_loader = DataLoader(train_ds, batch_sampler=DayBatchSampler(train_ds, shuffle=True),
                          collate_fn=stock_collate_fn, num_workers=args.num_workers)
val_loader   = DataLoader(val_ds,   batch_sampler=DayBatchSampler(val_ds,   shuffle=False),
                          collate_fn=stock_collate_fn, num_workers=args.num_workers)
test_loader  = DataLoader(test_ds,  batch_sampler=DayBatchSampler(test_ds,  shuffle=False),
                          collate_fn=stock_collate_fn, num_workers=args.num_workers)

# ── 模型 ────────────────────────────────────────────────────────────────────
args.enc_in = train_ds.n_features   # F

model = Model(args).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {n_params:,}  |  Features: {args.enc_in}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def neg_pearson(pred, target):
    """−Pearson(pred, target)。
    这个函数本身只做线性相关，它等于什么取决于 target 的形式：
      target = 原始收益率          → 优化 IC   （Pearson IC）
      target = 截面 rank 归一化值  → 优化 RankIC（≈ Spearman 相关）
    训练早期 pred 方差极小时分母趋零，梯度可能不稳，建议配合 MSE 使用（combined）。
    """
    p = pred.squeeze()
    t = target.squeeze()
    p = p - p.mean()
    t = t - t.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)


mse_loss_fn = nn.MSELoss()


def compute_loss(pred, target, mode: str, ic_weight: float) -> torch.Tensor:
    """
    三种 loss 及推荐搭配的 label（通过 --label 指定，对应 build_label_lib.py 中的列）：

      mse      — MSELoss(pred, target)
                 label 推荐: ret_5d_open（原始收益）
                 直接优化预测误差，对极端值敏感，早期收敛最稳定

      rankic   — −Pearson(pred, target)
                 label 推荐: ret_5d_open（原始收益），此时 Pearson = IC
                 若用 ret_5d_open_cs_rank [0,1]，Pearson ≈ Spearman RankIC
                 注意：训练早期 pred 方差小时梯度不稳，建议配合 mse 使用（combined）

      combined — MSELoss + ic_weight × (−Pearson)
                 label 推荐: ret_5d_open（raw, std≈0.04）
                 此时 MSE~0.003，|IC|~0.05，ic_weight≈0.05 使两项量级相当
                 label 改用 ret_5d_open_cs_rank [0,1] 时 MSE~0.08，ic_weight 需重新校准至~1
    """
    if mode == "rankic":
        return neg_pearson(pred, target)
    elif mode == "combined":
        return mse_loss_fn(pred, target) + ic_weight * neg_pearson(pred, target)
    else:  # mse
        return mse_loss_fn(pred, target)


# ── 评估函数 ─────────────────────────────────────────────────────────────────
def evaluate(loader):
    """每个 batch = 一整天（DayBatchSampler），逐日分别计算 IC 和 RankIC，再汇总均值/ICIR/RankICIR。"""
    model.eval()
    daily_ic, daily_rankic, all_mse = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).cpu().squeeze().numpy()
            tgt = y.cpu().squeeze().numpy()
            if out.ndim == 0:
                out = out.reshape(1)
                tgt = tgt.reshape(1)
            all_mse.append(np.mean((out - tgt) ** 2))
            if len(out) >= 5:
                ic_val, _ = pearsonr(out, tgt)
                rank_ic_val, _ = spearmanr(out, tgt)
                if not np.isnan(ic_val):
                    daily_ic.append(ic_val)
                if not np.isnan(rank_ic_val):
                    daily_rankic.append(rank_ic_val)

    mse = float(np.mean(all_mse)) if all_mse else float("nan")

    ic_arr = np.array(daily_ic)
    rankic_arr = np.array(daily_rankic)

    ic      = float(np.mean(ic_arr))     if len(ic_arr) > 0     else float("nan")
    # 5日重叠 label 逐日算 IC 序列高度自相关，ICIR = mean/std 会高估约 √horizon 倍，仅供趋势参考
    icir    = float(np.mean(ic_arr)    / (np.std(ic_arr)    + 1e-8)) if len(ic_arr) > 1     else float("nan")
    rankic  = float(np.mean(rankic_arr)) if len(rankic_arr) > 0 else float("nan")
    rankicir= float(np.mean(rankic_arr) / (np.std(rankic_arr) + 1e-8)) if len(rankic_arr) > 1 else float("nan")

    return mse, ic, icir, rankic, rankicir


# ── 训练循环 ─────────────────────────────────────────────────────────────────
best_val_ic = -np.inf
patience_cnt = 0
best_ckpt = os.path.join(args.ckpt_dir, "best.pt")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        if epoch == 1 and step == 0:
            print(f"\n[Batch info] x={tuple(x.shape)}  y={tuple(y.shape)}")
            print(f"  当前为纯时序模式：输入 (B, T, F)，B = 截面股票数 N（batchsize=N 的特殊情况）")
            print(f"  x: (N={x.shape[0]}=当天有效股票数, T={x.shape[1]}=seq_len, F={x.shape[2]}=特征数)")
            print(f"  y: (N, 1)  IC loss 在 N 上计算截面相关")
            print(f"  DayBatchSampler: 一个 batch = 一整天完整截面，保证 IC loss 有效")
            print(f"  NOTE: 扩展为 STGNN 时输入变为 (B, N, T, F)，N 固定、pad 对齐")
            print(f"  train 共 {len(train_loader)} 个截面（天），val {len(val_loader)} 个\n")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = compute_loss(out, y, args.loss, args.ic_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    val_mse, val_ic, val_icir, val_rankic, val_rankicir = evaluate(val_loader)

    print(f"Epoch {epoch:03d}  train_loss={avg_loss:.4f}  val_mse={val_mse:.4f}  "
          f"IC={val_ic:.4f}  ICIR={val_icir:.4f}  RankIC={val_rankic:.4f}  RankICIR={val_rankicir:.4f}")

    if not np.isnan(val_rankic) and val_rankic > best_val_ic:
        best_val_ic = val_rankic
        patience_cnt = 0
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, best_ckpt)
        print(f"  -> Best model saved (val_RankIC={val_rankic:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ── 测试 ─────────────────────────────────────────────────────────────────────
print("\nLoading best model for test...")
ckpt = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
test_mse, test_ic, test_icir, test_rankic, test_rankicir = evaluate(test_loader)
print(f"Test  mse={test_mse:.4f}  IC={test_ic:.4f}  ICIR={test_icir:.4f}  "
      f"RankIC={test_rankic:.4f}  RankICIR={test_rankicir:.4f}")
print(f"注意：ICIR/RankICIR 含 {args.horizon + 1} 天重叠，会高估约 √span 倍；"
      f"严谨的显著性判据见 backtest.py 输出的 t_NW")

# ── 自动回测 ──────────────────────────────────────────────────────────────────
if not args.no_backtest:
    import subprocess
    # out_dir 与 ckpt_dir 同名（去掉 checkpoints/ 前缀后挂到 backtest_results/）
    ckpt_basename = os.path.basename(os.path.normpath(args.ckpt_dir))
    bt_out_dir    = os.path.join("backtest_results", ckpt_basename)
    bt_cache_dir  = args.backtest_cache_dir if args.backtest_cache_dir else args.cache_dir

    bt_cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "backtest.py"),
        "--ckpt",      best_ckpt,
        "--cache_dir", bt_cache_dir,
        "--out_dir",   bt_out_dir,
        "--slippage_bps", str(args.slippage_bps),
        "--weight_mode",  args.weight_mode,
    ]
    if args.topk   is not None: bt_cmd += ["--topk",   str(args.topk)]
    if args.n_drop is not None: bt_cmd += ["--n_drop", str(args.n_drop)]

    print(f"\n{'='*60}")
    print(f"[auto-backtest] 开始回测: {best_ckpt}")
    print(f"[auto-backtest] 结果目录: {bt_out_dir}")
    print(f"[auto-backtest] 命令: {' '.join(bt_cmd)}")
    print(f"{'='*60}\n")

    ret = subprocess.run(bt_cmd)
    if ret.returncode != 0:
        print(f"\n[auto-backtest] 回测进程返回非零状态 {ret.returncode}，请检查输出。")
    else:
        print(f"\n[auto-backtest] 回测完成，结果见 {bt_out_dir}/")
