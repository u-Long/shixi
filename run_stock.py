"""
股票收益率预测训练入口
用法:
  python run_stock.py                         # 默认参数
  python run_stock.py --seq_len 30 --horizon 10 --d_model 256
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from data_provider.stock_dataset import StockDataset, stock_collate_fn, DayBatchSampler
from model.iTransformer_stock import Model


# ── 固定随机种子 ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── 参数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

# 数据
parser.add_argument("--cache_dir",        default="data/cache")
parser.add_argument("--price_path",       default="data/price_features.parquet")   # 兼容旧接口
parser.add_argument("--factor_path",      default="data/factor_values.parquet")
parser.add_argument("--factor_list_path", default="data/selected_factors.txt")
parser.add_argument("--seq_len",   type=int, default=30)
parser.add_argument("--horizon",   type=int, default=10)
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
parser.add_argument("--loss",          default="mse",  help="mse / rankic")
parser.add_argument("--ckpt_dir",      default="checkpoints/stock")

args = parser.parse_args()
os.makedirs(args.ckpt_dir, exist_ok=True)

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


def rankic_loss(pred, target):
    """DayBatchSampler 保证 batch = 完整一天，target 已截面 rank 归一化。
    全局 Pearson 天然等于当日 IC，无需按日期分组。"""
    p = pred.squeeze()
    t = target.squeeze()
    p = p - p.mean()
    t = t - t.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)


mse_loss_fn = nn.MSELoss()


# ── 评估函数 ─────────────────────────────────────────────────────────────────
def evaluate(loader):
    """每个 batch = 一整天（DayBatchSampler），逐日算 RankIC 再取均值，避免跨日混排。"""
    model.eval()
    daily_ics, all_mse = [], []
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
                ic, _ = spearmanr(out, tgt)
                if not np.isnan(ic):
                    daily_ics.append(ic)
    mse = float(np.mean(all_mse)) if all_mse else float("nan")
    ic  = float(np.mean(daily_ics)) if daily_ics else float("nan")
    return mse, ic


# ── 训练循环 ─────────────────────────────────────────────────────────────────
best_val_ic = -np.inf
patience_cnt = 0
best_ckpt = os.path.join(args.ckpt_dir, "best.pt")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = rankic_loss(out, y) if args.loss == "rankic" else mse_loss_fn(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    val_mse, val_ic = evaluate(val_loader)

    print(f"Epoch {epoch:03d}  train_loss={avg_loss:.4f}  "
          f"val_mse={val_mse:.4f}  val_rankIC={val_ic:.4f}")

    if not np.isnan(val_ic) and val_ic > best_val_ic:
        best_val_ic = val_ic
        patience_cnt = 0
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, best_ckpt)
        print(f"  -> Best model saved (val_IC={val_ic:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ── 测试 ─────────────────────────────────────────────────────────────────────
print("\nLoading best model for test...")
ckpt = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
test_mse, test_ic = evaluate(test_loader)
print(f"Test  mse={test_mse:.4f}  rankIC={test_ic:.4f}")
