"""
iTransformer_stock: 针对股票收益率预测的改造版本

相对于原版 iTransformer 的改动:
  1. 去掉 projector (Linear -> pred_len)，改为 MLP head -> 1 (标量收益率预测)
  2. 去掉时序预测的 use_norm (RevIN)，改为外部截面 z-score 归一化
  3. forward 直接返回 (B, 1) 的收益率预测
  4. 支持 class_strategy: 'mean'(mean pool) / 'cls_token'(CLS聚合)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len        = configs.seq_len
        self.output_attention = configs.output_attention
        self.class_strategy = getattr(configs, "class_strategy", "mean")

        # 和原版相同的 inverted embedding: (B, T, F) -> (B, F, d_model)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # CLS token（仅在 cls_token 策略时使用）
        if self.class_strategy == "cls_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, configs.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # MLP head: d_model -> mlp_hidden -> 1
        mlp_hidden = getattr(configs, "mlp_hidden", 64)
        self.head = nn.Sequential(
            nn.Linear(configs.d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        纯时序模式：x_enc (B, T, F)，B = 截面股票数 N（batchsize=N 的特殊情况）
        F = 因子数，attention 作用在 F 维度（variate-level），股票间互相独立
        returns: (B, 1)

        扩展为 STGNN 时：输入改为 (B, N, T, F)，forward 内部
          reshape (B*N, T, F) → Transformer → reshape (B, N, 1)
        图结构在 N 维度上定义，接在 reshape 之后、head 之前
        """
        # (B, T, F) -> (B, F, d_model)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if self.class_strategy == "cls_token":
            cls = self.cls_token.expand(enc_out.size(0), -1, -1)  # (B, 1, d_model)
            enc_out = torch.cat([cls, enc_out], dim=1)              # (B, F+1, d_model)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)      # (B, F[+1], d_model)

        if self.class_strategy == "cls_token":
            pooled = enc_out[:, 0, :]    # 取 CLS token
        else:
            pooled = enc_out.mean(dim=1) # mean pool over F

        out = self.head(pooled)          # (B, 1)

        if self.output_attention:
            return out, attns
        return out
