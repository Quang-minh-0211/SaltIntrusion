"""
Transformer.py — Vanilla Transformer (Vaswani et al., 2017) cho dự báo chuỗi
thời gian, bản trung thực theo Time-Series-Library (encoder-decoder, full
attention, start-token + zero placeholder cho horizon).

Khớp pipeline:
    model = Transformer(input_size=C, context_len=168, horizon=24,
                        d_model=512, n_layers=2, n_heads=8, d_ff=2048,
                        dropout=0.1, target_idx=0)
    out = model(Xb)          # Xb: (B, context_len, C) -> (B, horizon)

Ghi chú trung thực: bản gốc KHÔNG dùng RevIN; đầu vào dựa trên StandardScaler
toàn cục của pipeline (giống cách Informer/Transformer được đánh giá trên ETT...).
Có cờ revin=False (mặc định tắt) nếu bạn muốn bật cho công bằng với các model khác.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Embedding -----------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                   padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="leaky_relu")

    def forward(self, x):                      # (B, L, c_in) -> (B, L, d_model)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                      # x_mark = None (không có đặc trưng thời gian)
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# ----------------------------- Attention -----------------------------
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.triu(torch.ones((B, 1, L, L), dtype=torch.bool),
                                    diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(out.view(B, L, -1))


# ----------------------------- Encoder / Decoder -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
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
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


# ----------------------------- Model -----------------------------
class Transformer(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=512, d_ff=2048, n_layers=2, d_layers=1, n_heads=8,
                 dropout=0.1, activation="gelu", target_idx=0,
                 label_len=None, use_future=False, revin=False, **kwargs):
        super().__init__()
        if kwargs:
            print(f"[CẢNH BÁO] Transformer bỏ qua tham số không dùng: {sorted(kwargs)}")
        self.pred_len = horizon
        self.label_len = label_len if label_len is not None else context_len // 2
        self.target_idx = target_idx
        self.use_future = use_future
        self.aux_loss = None
        c_out = input_size                     # chế độ M: dự báo mọi kênh rồi lấy kênh đích

        self.enc_embedding = DataEmbedding(input_size, d_model, dropout)
        self.dec_embedding = DataEmbedding(input_size, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False, attention_dropout=dropout),
                                         d_model, n_heads),
                          d_model, d_ff, dropout, activation) for _ in range(n_layers)],
            norm_layer=nn.LayerNorm(d_model))
        self.decoder = Decoder(
            [DecoderLayer(
                AttentionLayer(FullAttention(True, attention_dropout=dropout), d_model, n_heads),
                AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                d_model, d_ff, dropout, activation) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True))

    def forward(self, x, future=None):         # x: (B, L, C)
        B, L, C = x.shape
        dec_zeros = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
        dec_inp = torch.cat([x[:, -self.label_len:, :], dec_zeros], dim=1)

        enc_out = self.encoder(self.enc_embedding(x), attn_mask=None)
        dec_out = self.decoder(self.dec_embedding(dec_inp), enc_out,
                               x_mask=None, cross_mask=None)   # (B, label_len+pred_len, c_out)
        return dec_out[:, -self.pred_len:, self.target_idx]    # (B, pred_len)
