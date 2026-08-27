"""
iTransformer_stock: 针对股票收益率预测的改造版本

相对原版 iTransformer:
  1. 去掉 projector (Linear -> pred_len)，改为 pooling + MLP head -> 1
  2. 去掉 RevIN (use_norm)。RevIN 是 instance-norm over time，会抹掉每只股票
     lookback 窗口内的均值和方差，而"这只股票最近在涨"正是动量信号本身。
     归一化改由外部截面 z-score 完成。
  3. forward 返回 (B,) 的截面预测分

本版新增两处（其余保持不变）:
  [1] variate embedding
      DataEmbedding_inverted 里 Linear(T -> d_model) 是 F 个因子**共享**的，
      attention 本身又是置换等变的。结果 token_i 和 token_j 的差异只来自数值，
      模型无法学到"第 7 个通道是换手率，权重给高一点"。
      原版靠"每个 token 预测它自己对应的变量"由输出位置隐含携带身份，
      改成 pool 成标量后这条链路断了，需要显式补一个可学习的因子编码。

  [2] gated pooling 替代 mean pool
      Encoder 有残差，token_i ≈ emb_i + Δ_i，emb_i 原封不动穿过整个 encoder。
      又因为 emb_i = W·x_i 中 W 共享，mean pool 的第一项为
          (1/F)·Σ W·x_i = W·( (1/F)·Σ x_i )
      即把 F 条含义、量纲都不同的因子时序直接相加求平均，这个量没有意义
      却直连 head。改为可学习的非均匀加权。
      gate 初始化为全 0 -> softmax 后均匀 -> 初始行为与 mean pool 完全等价，
      因此可以沿用原有超参起跑，不会引入训练不稳定。

configs 需要的字段:
  enc_in    int   因子数 F。必须等于实际输入列数（不是 yaml 里的因子总数，
                  要扣掉 SKIP_FACTORS 和计算失败的），以 df.shape[1] 为准。
  head_type str   'gate'(默认) | 'mean' | 'cls'，便于做消融对照
"""

import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len          = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in           = configs.enc_in

        # 兼容旧字段名 class_strategy
        head_type = getattr(configs, "head_type", None)
        if head_type is None:
            legacy = getattr(configs, "class_strategy", "gate")
            head_type = "cls" if legacy == "cls_token" else legacy
        self.head_type = head_type
        assert self.head_type in ("gate", "mean", "cls"), self.head_type

        d_model = configs.d_model

        # (B, T, F) -> (B, F, d_model)
        # 内部先 permute 成 (B, F, T)，再对最后一维做 Linear(T -> d_model)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, d_model, configs.embed, configs.freq, configs.dropout
        )

        # [1] 因子身份编码。打破对 F 的置换等变性。
        self.variate_emb = nn.Parameter(torch.zeros(1, self.enc_in, d_model))
        nn.init.trunc_normal_(self.variate_emb, std=0.02)

        if self.head_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
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
                        d_model, configs.n_heads,
                    ),
                    d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # [2] 只有 F 个参数，几乎不可能过拟合。全 0 初始化 => 起点等价于 mean pool。
        if self.head_type == "gate":
            self.gate = nn.Parameter(torch.zeros(self.enc_in))

        mlp_hidden = getattr(configs, "mlp_hidden", 64)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(mlp_hidden, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        纯时序模式: x_enc (B, T, F)。
        B 是并行度，模型对样本间无任何交互；B = N（当日截面股票数）纯粹是
        IC loss 的要求（需要完整截面才能算 mean/std），不是模型的要求。

        x_mark_enc 必须保持 None：传入时 DataEmbedding_inverted 会额外拼接
        时间协变量 token，enc_out 第 1 维将大于 enc_in，variate_emb 加法会失败。

        returns: (B,)   已 squeeze，直接送 loss

        扩展 STGNN 时: 输入改为 (B, N, T, F)
            reshape (B*N, T, F) -> embedding + encoder -> pooled (B*N, d)
            reshape (B, N, d)   -> [GNN / cross-stock attention] -> head -> (B, N)
        图结构在 N 维定义，插在 pooling 之后、head 之前，前面代码一行不用改。
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)          # (B, F, d)

        assert enc_out.size(1) == self.enc_in, (
            f"token 数 {enc_out.size(1)} != enc_in {self.enc_in}。"
            f"检查 configs.enc_in 是否等于实际因子列数，以及 x_mark_enc 是否为 None。"
        )
        enc_out = enc_out + self.variate_emb                     # [1]

        if self.head_type == "cls":
            cls = self.cls_token.expand(enc_out.size(0), -1, -1)
            enc_out = torch.cat([cls, enc_out], dim=1)           # (B, F+1, d)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)   # (B, F[+1], d)

        if self.head_type == "cls":
            pooled = enc_out[:, 0, :]
        elif self.head_type == "mean":
            pooled = enc_out.mean(dim=1)
        else:                                                    # [2] gate
            w = self.gate.softmax(dim=-1)                        # (F,)
            pooled = (enc_out * w.view(1, -1, 1)).sum(dim=1)     # (B, d)

        out = self.head(pooled).squeeze(-1)                      # (B,)

        if self.output_attention:
            return out, attns
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def factor_weights(self, feature_names=None):
        """
        读出学到的因子权重，用于和自己算的单因子 IC 排序做对照 —— 排序差异很大
        时通常是数据问题（某列错位、NaN 率过高、winsorize 没截干净）。
        head_type != 'gate' 时返回 None。
        """
        if self.head_type != "gate":
            return None
        w = self.gate.softmax(dim=-1).detach().cpu()
        if feature_names is None:
            return w
        assert len(feature_names) == len(w), \
            f"feature_names 长度 {len(feature_names)} != enc_in {len(w)}"
        import pandas as pd
        return pd.Series(w.numpy(), index=feature_names).sort_values(ascending=False)
