"""
股票数据集（缓存版）
依赖 build_cache.py 预先生成的 numpy 缓存：
  data/cache/feat_arr.npy   (T, S, F)
  data/cache/close_arr.npy  (T, S)
  data/cache/dates.npy      (T,)
  data/cache/stocks.npy     (S,)

每个样本:
  x: (T_win, F)  seq_len 天特征
  y: scalar      horizon 天对数收益率（在 collate_fn 里做截面 rank 归一化）
  di: int        date index（供 collate_fn 用）
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class StockDataset(Dataset):
    def __init__(
        self,
        cache_dir: str = "data/cache",
        price_path: str = None,       # 兼容旧接口，忽略
        factor_path: str = None,
        factor_list_path: str = None,
        seq_len: int = 60,
        horizon: int = 10,            # 仅在 label_arr 不存在时用于动态计算
        flag: str = "train",
        # 绝对日期划分（优先）
        train_start: str = None,
        train_end: str = None,
        val_start: str = None,
        val_end: str = None,
        test_start: str = None,
        test_end: str = None,
        # 比例划分（fallback，仅在日期未指定时生效）
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        nan_thresh: float = 0.3,
        label_horizon: int = None,    # 覆盖 horizon，仅影响动态 label 计算
    ):
        self.seq_len = seq_len
        self.horizon = horizon

        # ── 加载缓存 ─────────────────────────────────────────────────────────
        print(f"[StockDataset] Loading cache from {cache_dir}")
        feat_arr  = np.load(os.path.join(cache_dir, "feat_arr.npy"),  mmap_mode="r")
        close_arr = np.load(os.path.join(cache_dir, "close_arr.npy"), mmap_mode="r")
        dates     = np.load(os.path.join(cache_dir, "dates.npy"),     allow_pickle=True)
        stocks    = np.load(os.path.join(cache_dir, "stocks.npy"),    allow_pickle=True)
        feat_cols = np.load(os.path.join(cache_dir, "feature_cols.npy"), allow_pickle=True)

        # label_arr（预计算，可选）
        label_path = os.path.join(cache_dir, "label_arr.npy")
        if os.path.exists(label_path):
            label_arr = np.load(label_path, mmap_mode="r")
            self.use_precomputed_label = True
        else:
            label_arr = None
            self.use_precomputed_label = False

        # universe_mask（可选，点时间成分过滤）
        univ_mask_path = os.path.join(cache_dir, "universe_mask.npy")
        if os.path.exists(univ_mask_path):
            universe_mask = np.load(univ_mask_path, mmap_mode="r")
            print(f"[StockDataset] universe_mask loaded: coverage={universe_mask.mean():.3f}")
        else:
            universe_mask = None

        self.feat_arr     = feat_arr
        self.close_arr    = close_arr
        self.label_arr    = label_arr
        self.universe_mask = universe_mask
        self.dates        = list(dates)
        self.stocks       = list(stocks)
        self.feature_cols = list(feat_cols)
        self.n_features   = len(feat_cols)
        self.horizon      = label_horizon if label_horizon is not None else horizon

        import pandas as _pd
        T, S, F = feat_arr.shape
        _horizon = label_horizon if label_horizon is not None else horizon

        # ── 时间切分 ──────────────────────────────────────────────────────────
        use_date_split = train_start is not None
        if use_date_split:
            dates_pd = _pd.to_datetime(dates)
            split_map = {
                "train": (_pd.Timestamp(train_start), _pd.Timestamp(train_end)),
                "val":   (_pd.Timestamp(val_start),   _pd.Timestamp(val_end)),
                "test":  (_pd.Timestamp(test_start),  _pd.Timestamp(test_end)),
            }
            # embargo: 每段最后一个样本的 label 结束日期必须严格早于下一段起始日期，
            # 防止 train/val label 时间窗口重叠（10日 label 与相邻段共享 9 天区间）
            embargo_end = {
                "train": _pd.Timestamp(val_start)  if val_start  else None,
                "val":   _pd.Timestamp(test_start) if test_start else None,
                "test":  None,
            }
            emb = embargo_end[flag]
            lo, hi = split_map[flag]
            di_range = [
                i for i, d in enumerate(dates_pd)
                if lo <= d <= hi
                and i >= seq_len - 1
                and i + _horizon < T
                and (emb is None or dates_pd[i + _horizon] < emb)
            ]
        else:
            t1 = int(T * train_ratio)
            t2 = int(T * (train_ratio + val_ratio))
            # embargo: 末尾去掉 _horizon 天，使 label 结束日不跨入下一段
            if flag == "train":
                di_range = range(seq_len - 1, max(seq_len - 1, t1 - _horizon))
            elif flag == "val":
                di_range = range(t1, max(t1, t2 - _horizon))
            else:
                di_range = range(t2, T - _horizon)

        if len(di_range) > 0:
            dates_pd_all = _pd.to_datetime(dates) if not use_date_split else dates_pd
            d0 = dates_pd_all[di_range[0]]  if not isinstance(di_range, range) else dates_pd_all[di_range[0]]
            d1 = dates_pd_all[di_range[-1]] if not isinstance(di_range, range) else dates_pd_all[di_range[-1]]
            d0_label_end = dates_pd_all[min(di_range[0]  + _horizon, T-1)]
            d1_label_end = dates_pd_all[min(di_range[-1] + _horizon, T-1)]
            print(f"[StockDataset] {flag}: {len(di_range)} dates  "
                  f"signal=[{d0.date()} ~ {d1.date()}]  "
                  f"label_end=[{d0_label_end.date()} ~ {d1_label_end.date()}]")
        else:
            print(f"[StockDataset] {flag}: 0 dates")
        print(f"[StockDataset] {flag}: {S} stocks, {F} features")

        # ── 构建样本索引（向量化，快速）───────────────────────────────────────
        print("[StockDataset] Building sample index...")
        samples = []
        for di in di_range:
            # 检查 label 是否有效
            if self.use_precomputed_label:
                lv = label_arr[di, :]
                valid_label = ~np.isnan(lv)
            else:
                c0 = close_arr[di, :]
                ch = close_arr[di + _horizon, :]
                valid_label = (c0 > 0) & (ch > 0) & ~np.isnan(c0) & ~np.isnan(ch)

            # 检查特征窗口 NaN 比例
            window = feat_arr[di - seq_len + 1: di + 1, :, :]
            nan_ratio = np.isnan(window).mean(axis=(0, 2))
            valid_feat = nan_ratio <= nan_thresh

            valid = valid_label & valid_feat
            # 点时间 universe 过滤
            if self.universe_mask is not None:
                valid = valid & self.universe_mask[di, :]
            for si in np.where(valid)[0]:
                samples.append((di, int(si)))

        self.samples = samples
        print(f"[StockDataset] {flag}: {len(samples)} total samples")

        # 按日期分组，供 DayBatchSampler 使用
        from collections import defaultdict
        day_to_indices = defaultdict(list)
        for idx, (di, si) in enumerate(samples):
            day_to_indices[di].append(idx)
        self.day_to_indices = dict(day_to_indices)
        self.valid_days = sorted(day_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        di, si = self.samples[idx]
        x = self.feat_arr[di - self.seq_len + 1: di + 1, si, :].copy()
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)

        if self.use_precomputed_label:
            y = np.float32(self.label_arr[di, si])
        else:
            c0 = float(self.close_arr[di, si])
            ch = float(self.close_arr[di + self.horizon, si])
            y  = np.float32(np.log(ch / c0))

        return torch.from_numpy(x), torch.tensor(y), torch.tensor(di)


def stock_collate_fn(batch):
    """stack 即可，rank 归一化已在 build_label_lib.py 离线完成（全 universe 截面）。
    返回 (x, y)，y shape = (B, 1)。"""
    xs, ys, _ = zip(*batch)
    return torch.stack(xs), torch.stack(ys).unsqueeze(-1)


class DayBatchSampler(Sampler):
    """每个 batch = 一整天的全部有效股票，保证截面完整性。
    shuffle=True 时打乱日期顺序（天与天之间随机），天内股票顺序不影响结果。"""

    def __init__(self, dataset, shuffle: bool = True):
        self.day_to_indices = dataset.day_to_indices
        self.valid_days = dataset.valid_days
        self.shuffle = shuffle

    def __iter__(self):
        days = list(self.valid_days)
        if self.shuffle:
            random.shuffle(days)
        for d in days:
            yield self.day_to_indices[d]

    def __len__(self):
        return len(self.valid_days)
