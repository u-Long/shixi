"""
MLP baseline for stock return prediction.

Input : (B, T, F)  — B stocks, T time steps, F features
Output: (B, 1)     — scalar return prediction per stock

Architecture:
  flatten (T*F) → Linear → BN → GELU → Dropout
                → Linear → BN → GELU → Dropout
                → Linear → 1

Uses args.seq_len, args.enc_in, args.mlp_hidden, args.dropout.
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        T = configs.seq_len
        F = configs.enc_in
        H = getattr(configs, "mlp_hidden", 128)
        p = getattr(configs, "dropout", 0.1)

        self.net = nn.Sequential(
            nn.Flatten(),                       # (B, T*F)
            nn.Linear(T * F, H * 2),
            nn.BatchNorm1d(H * 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(H * 2, H),
            nn.BatchNorm1d(H),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(H, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        return self.net(x)          # (B, 1)
