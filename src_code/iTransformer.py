"""
iTransformer.py — bản dựng lại iTransformer (Liu et al., ICLR 2024) bám sát mã nguồn gốc
=========================================================================================
Nguồn đối chiếu: thuml/Time-Series-Library
    models/iTransformer.py, layers/Embed.py (DataEmbedding_inverted),
    layers/Transformer_EncDec.py (Encoder, EncoderLayer),
    layers/SelfAttention_Family.py (FullAttention, AttentionLayer)

Giữ nguyên từng phép tính của tác giả (task long_term_forecast), chỉ đổi interface cho
khớp pipeline 03_scenario3:
    model = iTransformer(input_size=C, context_len=168, horizon=H, target_idx=tgt_idx)
    out   = model(Xb)                      # Xb: (B, context_len, C) -> (B, H)

Pipeline không có x_mark (đặc trưng giờ/ngày), nên embedding đảo chỉ là Linear(L -> d) trên
mỗi biến — đúng nhánh `x_mark is None` của DataEmbedding_inverted gốc.

KHÔNG có stats token, KHÔNG có cờ revin: đây là bản gốc của tác giả để làm mốc so sánh.
Chuẩn hóa từng biến (Non-stationary Transformer) là của chính iTransformer, luôn bật khi
use_norm=True (mặc định, giống tác giả).

Luồng (B batch, L = context_len, H = horizon, C biến, d = d_model):
    chuẩn hóa từng biến theo thời gian                          (B, L, C)
    -> đảo trục + Linear(L -> d) mỗi biến: C token              (B, C, d)
    -> e_layers x EncoderLayer (attention GIỮA các biến + FFN)  (B, C, d)
    -> projection Linear(d -> H), đảo lại, lấy C biến đầu        (B, H, C)
    -> khử chuẩn hóa, lấy kênh target_idx                       (B, H)

Mặc định theo script iTransformer của tác giả: d_model=512, d_ff=512, e_layers=3 (ở đây để
n_layers=2 cho nhẹ; chỉnh qua BEST_PARAMS), n_heads=8, dropout=0.1, activation="gelu".
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# DataEmbedding_inverted — layers/Embed.py (nhánh x_mark is None)
# --------------------------------------------------------------------------
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):              # x: (B, L, C)
        x = x.permute(0, 2, 1)                      # (B, C, L)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                      # (B, C, d)  (biến là token)


# --------------------------------------------------------------------------
# FullAttention + AttentionLayer — layers/SelfAttention_Family.py
# --------------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)


# --------------------------------------------------------------------------
# EncoderLayer + Encoder — layers/Transformer_EncDec.py
# --------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.attn_layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# --------------------------------------------------------------------------
# Model — models/iTransformer.py (forecast)
# --------------------------------------------------------------------------
class iTransformer(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=512, d_ff=512, n_layers=2, n_heads=8,
                 dropout=0.1, activation="gelu", target_idx=0,
                 use_norm=True, use_future=False, **kwargs):
        super().__init__()
        if kwargs:   # print thay vì warnings (notebook đang tắt warnings)
            print(f"[CẢNH BÁO] iTransformer bỏ qua tham số không dùng: {sorted(kwargs)}")
        self.seq_len, self.pred_len = context_len, horizon
        self.target_idx, self.use_future, self.aux_loss = target_idx, use_future, None
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(context_len, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation)
             for _ in range(n_layers)],
            norm_layer=nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, horizon, bias=True)

    def forecast(self, x_enc, x_mark_enc=None):     # (B, L, C) -> (B, H, C)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         # (B, C, d)
        enc_out = self.encoder(enc_out, attn_mask=None)         # (B, C, d)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # (B, H, C)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x, future=None):              # (B, L, C) -> (B, H)
        dec_out = self.forecast(x)
        return dec_out[:, -self.pred_len:, self.target_idx]


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    B, L, C = 4, 168, 3
    for H in (6, 12, 24, 48):
        m = iTransformer(input_size=C, context_len=L, horizon=H, d_model=128,
                         d_ff=128, n_layers=2, target_idx=1)
        x = torch.randn(B, L, C)
        y = m(x)
        y.sum().backward()
        ng = [n for n, p in m.named_parameters() if p.grad is None]
        assert y.shape == (B, H) and not ng, (H, ng)
        print(f"H={H:>2}  params={sum(p.numel() for p in m.parameters()):>9,}")
    print("OK.")
