"""
可视化 iTransformer 跨因子 attention map，以及通过 mean-pool 反推因子重要性。

用法:
  python scripts/visualize_attention.py

输出:
  plots/attention_maps.png  -- 各模型各层各头的 F×F attention heatmap
  plots/factor_importance.png -- attention 对角线/行均值 → 因子重要性排名 vs IC排名
"""

import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from types import SimpleNamespace

# ---- 将项目根目录加入路径 ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.iTransformer_stock import Model

# ---- 要分析的 checkpoint ----
CKPTS = [
    {
        "name": "826330\nfea2 combined",
        "ckpt": "checkpoints/826330/best.pt",
        "config": "checkpoints/826330/config.json",
    },
    {
        "name": "826_v2_rankic\nfea2 rankIC",
        "ckpt": "checkpoints/826_v2_rankic/best.pt",
        "config": "checkpoints/826_v2_rankic/config.json",
    },
    {
        "name": "826_fea3_ret5do\nfea3 combined",
        "ckpt": "checkpoints/826_fea3_ret5do/best.pt",
        "config": "checkpoints/826_fea3_ret5do/config.json",
    },
    {
        "name": "827_fea3_ret5davgo\nfea3 combined(new)",
        "ckpt": "checkpoints/827_fea3_ret5davgo/best.pt",
        "config": "checkpoints/827_fea3_ret5davgo/config.json",
    },
]


def load_model(ckpt_path, cfg_path):
    with open(cfg_path) as f:
        raw = json.load(f)
    args_dict = raw.get("args", raw)
    # 默认补全
    defaults = dict(
        factor=1, activation="gelu", embed="fixed", freq="b",
        dropout=0.1, mlp_hidden=64, class_strategy="mean",
        output_attention=True,       # 强制开启
    )
    defaults.update(args_dict)
    defaults["output_attention"] = True  # 覆盖确保开启
    cfg = SimpleNamespace(**defaults)

    model = Model(cfg)
    state = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def get_feature_names(cfg):
    """从 cache 里读取特征列名"""
    cache_dir = os.path.join(ROOT, cfg.cache_dir)
    col_file = os.path.join(cache_dir, "feature_cols.npy")
    if os.path.exists(col_file):
        cols = np.load(col_file, allow_pickle=True).tolist()
        return [str(c) for c in cols]
    return [f"F{i}" for i in range(cfg.seq_len)]  # fallback


def sample_batch(cfg, n_stocks=256, device="cpu"):
    """从 cache 里随机抽一个测试日的截面数据作为 dummy input"""
    cache_dir = os.path.join(ROOT, cfg.cache_dir)
    feat_file = os.path.join(cache_dir, "feat_arr.npy")
    if os.path.exists(feat_file):
        # mmap 避免全量读入
        feat = np.load(feat_file, mmap_mode="r")
        T, S, F = feat.shape
        # 取最后 10% 日期里的一天
        t_idx = int(T * 0.9) + np.random.randint(0, int(T * 0.1))
        day_data = feat[t_idx]  # (S, F)
        # 随机选 n_stocks 只股票
        s_idx = np.random.choice(S, min(n_stocks, S), replace=False)
        day_data = day_data[s_idx]  # (n_stocks, F)
        # 构造 (B, T=seq_len, F)：用最近 seq_len 天
        seq_len = cfg.seq_len
        start_t = max(0, t_idx - seq_len)
        window = feat[start_t:t_idx, :, :]  # (seq_len, S, F)
        window = window[:, s_idx, :]         # (seq_len, n_stocks, F)
        x = torch.tensor(window.transpose(1, 0, 2), dtype=torch.float32)  # (B, T, F)
        # nan → 0
        x = torch.nan_to_num(x, nan=0.0)
        return x.to(device)
    # fallback: random
    F = getattr(cfg, "d_model", 64)
    return torch.randn(n_stocks, cfg.seq_len, 39)


@torch.no_grad()
def extract_attentions(model, x):
    """
    返回 attns: list of (B, H, F, F) tensor，对应 e_layers 层
    """
    out, attns = model(x)
    # attns 是 list，每元素 (B, H, F, F)
    # 有时包装在 tuple 里
    result = []
    for a in attns:
        if a is not None:
            result.append(a.cpu().float())
    return result  # len = e_layers


def compute_factor_importance(attns):
    """
    mean pool 策略下，每个因子对最终输出的贡献
    = 所有层、所有头的 attention 矩阵按列均值之积（近似）
    简化为：取所有层所有头的 attn 矩阵，平均 → (F, F)，再对行求和。

    直觉：attn[i,j] = "因子i 关注因子j 的程度"
    行和：因子 j 被总关注度（"被查询"程度 → 贡献度）
    列和：因子 i 主动关注别人的程度
    """
    # 将所有层拼起来做平均
    stacked = torch.stack(attns, dim=0)  # (L, B, H, F, F)
    # 均值 over layers, batch, heads
    mean_attn = stacked.mean(dim=(0, 1, 2))  # (F, F)
    # 列和 = 各因子被查询的重要性
    col_sum = mean_attn.sum(dim=0)   # (F,)
    # 行和 = 各因子主动查询别人
    row_sum = mean_attn.sum(dim=1)   # (F,)
    # 对角线 = 自注意力强度
    diag = torch.diag(mean_attn)     # (F,)
    return mean_attn.numpy(), col_sum.numpy(), row_sum.numpy(), diag.numpy()


def load_ic_for_cache(cfg):
    """
    从 cache 的 feat_arr / label_arr 直接计算每个因子的 IC（Pearson 相关，测试集）。
    返回 dict: {factor_name: ic_value}（带符号，方便方向对齐）
    """
    cache_dir = os.path.join(ROOT, cfg.cache_dir)
    feat_file = os.path.join(cache_dir, "feat_arr.npy")
    label_file = os.path.join(cache_dir, "label_arr.npy")
    col_file = os.path.join(cache_dir, "feature_cols.npy")
    dates_file = os.path.join(cache_dir, "dates.npy")

    if not (os.path.exists(feat_file) and os.path.exists(label_file)):
        # fallback: alpha191 meta
        import pandas as pd
        meta_path = os.path.join(ROOT, "data/selected_factors_meta.csv")
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            return dict(zip(df["name"], df["ic"]))
        return {}

    feat = np.load(feat_file, mmap_mode="r")   # (T, S, F)
    label = np.load(label_file, mmap_mode="r") # (T, S)
    cols = np.load(col_file, allow_pickle=True).tolist()
    T, S, F = feat.shape

    # 使用测试集日期范围（后 ~15% 或 val_start 之后）
    # 简单取后 200 个交易日做 IC 统计
    t_start = max(0, T - 200)
    feat_sub = feat[t_start:].astype(np.float32)   # (200, S, F)
    label_sub = label[t_start:].astype(np.float32) # (200, S)

    n_days = feat_sub.shape[0]
    ic_daily = np.full((n_days, F), np.nan)

    for t in range(n_days):
        lbl = label_sub[t]    # (S,)
        valid = np.isfinite(lbl)
        if valid.sum() < 50:
            continue
        for f in range(F):
            fv = feat_sub[t, :, f]
            mask = valid & np.isfinite(fv)
            if mask.sum() < 50:
                continue
            # Pearson 相关
            x, y = fv[mask], lbl[mask]
            x_d, y_d = x - x.mean(), y - y.mean()
            denom = np.sqrt((x_d**2).sum() * (y_d**2).sum())
            if denom < 1e-12:
                continue
            ic_daily[t, f] = (x_d * y_d).sum() / denom

    mean_ic = np.nanmean(ic_daily, axis=0)  # (F,)
    return {str(cols[f]): float(mean_ic[f]) for f in range(F)}


# ============================================================
#  绘图 1：每个模型每层每头的 attention heatmap
# ============================================================
def plot_attention_maps(ckpt_results, out_path):
    n_models = len(ckpt_results)
    # 找最大 layers × heads
    max_layers = max(len(r["attns"]) for r in ckpt_results)
    max_heads = max(r["attns"][0].shape[1] for r in ckpt_results if r["attns"])

    # 布局：行 = 模型，列 = 层*头（每层头数个子图）
    # 简化：每个模型展示所有层，每层取 head 均值
    n_cols = max_layers
    n_rows = n_models

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle("iTransformer 跨因子 Attention Map（各层 head 均值，test 集截面）",
                 fontsize=14, y=1.01)

    for row_idx, r in enumerate(ckpt_results):
        attns = r["attns"]    # list of (B, H, F, F)
        feat_names = r["feat_names"]
        n_f = len(feat_names)
        short_names = [n[:12] for n in feat_names]  # 截短方便显示

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(attns):
                ax.axis("off")
                continue

            # (B, H, F, F) → mean over B, H → (F, F)
            attn_mat = attns[col_idx].mean(dim=(0, 1)).numpy()

            # 绘制 heatmap
            im = ax.imshow(attn_mat, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=attn_mat.max())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if n_f <= 50:
                ax.set_xticks(range(n_f))
                ax.set_xticklabels(short_names, rotation=90, fontsize=5)
                ax.set_yticks(range(n_f))
                ax.set_yticklabels(short_names, fontsize=5)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            layer_label = f"Layer {col_idx+1}"
            title = f"{r['name']}\n{layer_label}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Key (被关注因子)", fontsize=7)
            ax.set_ylabel("Query (发起关注因子)", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[保存] {out_path}")


# ============================================================
#  绘图 2：每个模型的因子重要性排名 vs IC 排名
# ============================================================
def plot_factor_importance(ckpt_results, out_path):
    n_models = len(ckpt_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), squeeze=False)
    fig.suptitle("因子重要性（attention 列和）vs |IC| 排名对比", fontsize=13)

    for idx, r in enumerate(ckpt_results):
        ax = axes[0][idx]
        feat_names = r["feat_names"]
        _, col_sum, _, _ = r["importance"]  # col_sum = 被关注度

        # 按 col_sum 降序排列
        order = np.argsort(col_sum)[::-1]
        sorted_names = [feat_names[i] for i in order]
        sorted_vals = col_sum[order]

        # 对应 IC（如果有）
        ic_dict = r["ic_dict"]
        ic_vals = np.array([ic_dict.get(feat_names[i], np.nan) for i in order])

        # 绘制双轴
        ax2 = ax.twinx()

        x = np.arange(len(sorted_names))
        bars = ax.bar(x, sorted_vals, alpha=0.6, color="steelblue", label="Attn 列和（被关注度）")

        # IC 散点（只画有 IC 的点）
        mask = ~np.isnan(ic_vals)
        if mask.any():
            ax2.scatter(x[mask], ic_vals[mask], color="tomato", s=30, zorder=5, label="IC（带符号）")
            ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax2.set_ylabel("IC（带符号）", color="tomato", fontsize=8)
            ax2.tick_params(axis="y", colors="tomato")
            # 计算 rank 相关
            from scipy.stats import spearmanr
            attn_ranks = np.argsort(np.argsort(sorted_vals[mask]))
            ic_abs_ranks = np.argsort(np.argsort(np.abs(ic_vals[mask])))
            rho, pval = spearmanr(attn_ranks, ic_abs_ranks)
            ax.set_title(f"{r['name'].replace(chr(10), ' ')}\nSpearman ρ={rho:.3f} (p={pval:.3f})", fontsize=9)

        ax.set_title(r["name"].replace("\n", " "), fontsize=9)
        ax.set_ylabel("Attention 列和", fontsize=8)
        ax.set_xticks(x)
        short = [n[:10] for n in sorted_names]
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.legend(fontsize=7, loc="upper right")
        if mask.any():
            ax2.legend(fontsize=7, loc="upper center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[保存] {out_path}")


# ============================================================
#  绘图 3：每个模型的多头 attention heatmap（拆开每个头）
# ============================================================
def plot_per_head_attention(ckpt_results, out_path):
    """为每个模型，取第 1 层，展示每个 head 的 F×F attention"""
    n_models = len(ckpt_results)
    # 取最大 heads 数
    max_heads = max(r["attns"][0].shape[1] for r in ckpt_results if r["attns"])

    fig, axes = plt.subplots(
        n_models, max_heads,
        figsize=(4 * max_heads, 3.8 * n_models),
        squeeze=False,
    )
    fig.suptitle("各模型第1层 每个注意力头 的 Attention Map（test截面均值）",
                 fontsize=13, y=1.01)

    for row_idx, r in enumerate(ckpt_results):
        attn_layer0 = r["attns"][0]  # (B, H, F, F)
        n_heads = attn_layer0.shape[1]
        feat_names = r["feat_names"]
        n_f = len(feat_names)
        short_names = [n[:10] for n in feat_names]

        for h in range(max_heads):
            ax = axes[row_idx][h]
            if h >= n_heads:
                ax.axis("off")
                continue

            # 对 batch 求均值
            attn_h = attn_layer0[:, h, :, :].mean(0).numpy()  # (F, F)

            im = ax.imshow(attn_h, aspect="auto", cmap="Blues",
                           vmin=0, vmax=attn_h.max())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if n_f <= 50:
                ax.set_xticks(range(n_f))
                ax.set_xticklabels(short_names, rotation=90, fontsize=5)
                ax.set_yticks(range(n_f))
                ax.set_yticklabels(short_names, fontsize=5)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_title(f"{r['name'].split(chr(10))[0]}  Head {h+1}", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[保存] {out_path}")


# ============================================================
#  绘图 4：因子重要性数值排名表
# ============================================================
def print_factor_ranking_table(ckpt_results):
    print("\n" + "=" * 80)
    print("因子重要性排名（attention 列和 = 被关注度，降序 Top-20）")
    print("=" * 80)
    for r in ckpt_results:
        feat_names = r["feat_names"]
        _, col_sum, _, _ = r["importance"]
        order = np.argsort(col_sum)[::-1]
        ic_dict = r["ic_dict"]

        print(f"\n模型: {r['name'].replace(chr(10), ' ')}")
        print(f"{'Rank':>4}  {'Factor':<40}  {'Attn列和':>10}  {'IC':>8}  {'|IC|排名':>8}")
        print("-" * 80)
        # 计算 |IC| 排名（有 IC 的因子中）
        ic_vals_all = np.array([ic_dict.get(n, np.nan) for n in feat_names])
        valid_mask = ~np.isnan(ic_vals_all)
        ic_abs_rank = np.full(len(feat_names), np.nan)
        if valid_mask.any():
            ic_abs_vals = np.abs(ic_vals_all[valid_mask])
            ic_indices = np.where(valid_mask)[0]
            # 降序排名
            sorted_ic_order = ic_indices[np.argsort(ic_abs_vals)[::-1]]
            for rk, idx in enumerate(sorted_ic_order):
                ic_abs_rank[idx] = rk + 1

        for rank, i in enumerate(order[:20]):
            name = feat_names[i]
            val = col_sum[i]
            ic_v = ic_dict.get(name, float("nan"))
            ic_str = f"{ic_v:+.4f}" if not np.isnan(ic_v) else "     N/A"
            icr_str = f"{int(ic_abs_rank[i])}" if not np.isnan(ic_abs_rank[i]) else "N/A"
            print(f"{rank+1:>4}  {name:<40}  {val:>10.4f}  {ic_str:>8}  {icr_str:>8}")


# ============================================================
#  主流程
# ============================================================
def main():
    os.chdir(ROOT)
    np.random.seed(42)
    torch.manual_seed(42)

    results = []
    for spec in CKPTS:
        cfg_path = spec["config"]
        ckpt_path = spec["ckpt"]
        if not os.path.exists(cfg_path):
            print(f"[跳过] 找不到 config: {cfg_path}")
            continue
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 找不到 checkpoint: {ckpt_path}")
            continue

        print(f"\n[加载] {spec['name'].replace(chr(10),' ')} ...")
        model, cfg = load_model(ckpt_path, cfg_path)
        feat_names = get_feature_names(cfg)
        print(f"  特征数: {len(feat_names)}")

        x = sample_batch(cfg, n_stocks=512)
        print(f"  输入 shape: {x.shape}")

        attns = extract_attentions(model, x)
        print(f"  attention layers: {len(attns)}, shape[0]: {attns[0].shape if attns else 'N/A'}")

        if not attns:
            print("  [警告] 未获得 attention weights，跳过")
            continue

        importance = compute_factor_importance(attns)
        ic_dict = load_ic_for_cache(cfg)

        results.append({
            "name": spec["name"],
            "attns": attns,
            "feat_names": feat_names,
            "importance": importance,
            "ic_dict": ic_dict,
        })

    if not results:
        print("没有可用模型结果，退出")
        return

    # 输出路径
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

    print("\n[绘图] attention maps (head均值，各层)...")
    plot_attention_maps(results, os.path.join(ROOT, "plots/attention_maps.png"))

    print("[绘图] per-head attention maps (第1层)...")
    plot_per_head_attention(results, os.path.join(ROOT, "plots/attention_per_head.png"))

    print("[绘图] 因子重要性 vs IC...")
    plot_factor_importance(results, os.path.join(ROOT, "plots/factor_importance.png"))

    print_factor_ranking_table(results)

    print("\n完成。所有图片保存在 plots/ 目录。")


if __name__ == "__main__":
    main()
