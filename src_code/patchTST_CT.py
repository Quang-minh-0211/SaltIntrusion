# ============================================================
# patchTST_CT.py — CT-PatchTST (Channel-Time PatchTST)
# Theo Huo et al., "CT-PatchTST: Channel-Time Patch Time-Series
# Transformer for Long-Term Renewable Energy Forecasting" (arXiv:2501.08620)
#
# Ý tưởng của bài báo: PatchTST gốc độc lập kênh hoàn toàn nên bỏ sót
# quan hệ giữa các kênh. CT-PatchTST thêm CHANNEL ATTENTION đặt TRƯỚC
# time attention:
#
#   RevIN -> patchify + projection + pos
#     -> [Channel attention]: tại MỖI vị trí patch n, attention giữa
#        M kênh với nhau (câu dài M token)          (B*N, M, D)
#     -> [Time attention]: PatchTST chuẩn, độc lập kênh (B*M, N, D)
#     -> flatten head -> target -> denorm
#
# Khác biệt so với PATCHTST_ATTENTION của ta: (1) attend trên TOÀN BỘ
# M kênh chứ không chỉ 3 kênh mặn; (2) đặt TRƯỚC encoder thời gian;
# (3) không có cổng alpha (channel attention là block transformer đầy
# đủ với residual + FFN như bài báo).
#
# Tương thích pipeline: forward(x, x_future=None) -> (B, horizon);
# kế thừa PatchTST_BASE nên hỗ trợ cả use_future.
# ============================================================

import torch
import torch.nn as nn

from patchTST_Base import PatchTST_BASE


class ChannelAttentionBlock(nn.Module):
    """Một block transformer pre-norm chạy attention theo chiều KÊNH.

    Đầu vào/ra: (B, M, N, D). Tại mỗi vị trí patch n, "câu" gồm M token
    (mỗi token = một kênh) attend lẫn nhau.
    self.last_attn: (B*N, M, M) — bản đồ kênh-nhìn-kênh để vẽ hình.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, M, D)   # câu M token/patch

        z = self.norm1(h)
        out, attn_w = self.attn(z, z, z, need_weights=True,
                                average_attn_weights=True)
        self.last_attn = attn_w.detach()                 # (B*N, M, M)
        h = h + self.drop(out)
        h = h + self.drop(self.ffn(self.norm2(h)))

        return h.reshape(B, N, M, D).permute(0, 2, 1, 3)


class CT_PatchTST(PatchTST_BASE):
    """CT-PatchTST: channel attention (trước) + time attention (sau).

    Tham số thêm
    ------------
    n_channel_layers : số block channel attention (bài báo dùng ít; 1 là
                       hợp lý với dữ liệu nhỏ)
    channel_heads    : số head của channel attention
    """

    def __init__(self, *args, n_channel_layers: int = 1,
                 channel_heads: int = 4, channel_dropout: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.embed.out_features
        d_ff = self.encoder.layers[0].linear1.out_features
        # Embedding danh tính kênh: channel attention cần phân biệt
        # kênh nào là gì (trọng số projection dùng chung mọi kênh)
        self.channel_emb = nn.Parameter(
            torch.randn(self.input_size, 1, d_model) * 0.02)
        self.channel_blocks = nn.ModuleList([
            ChannelAttentionBlock(d_model, channel_heads, d_ff,
                                  channel_dropout)
            for _ in range(n_channel_layers)
        ])

    def forward(self, x: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.context_len, (
            f"Model build với context_len={self.context_len}, nhận L={L}.")

        # (0) RevIN
        if self.revin is not None:
            x = self.revin.normalize(x)

        # (2)(3)(4) tách kênh + patchify + embedding (như PatchTST)
        x = x.permute(0, 2, 1).reshape(B * M, L)
        x = self._patchify(x)                              # (B*M, N, P)
        x = self.embed_dropout(self.embed(x) + self.pos)   # (B*M, N, D)

        # ---- (4.5) CHANNEL ATTENTION (điểm mới của CT-PatchTST) ----
        x = x.view(B, M, self.n_patches, -1)               # (B, M, N, D)
        x = x + self.channel_emb.unsqueeze(0)              # danh tính kênh
        for blk in self.channel_blocks:
            x = blk(x)                                     # (B, M, N, D)
        x = x.reshape(B * M, self.n_patches, -1)
        # ------------------------------------------------------------

        # (5) TIME ATTENTION: encoder PatchTST chuẩn, độc lập kênh
        x = self.encoder_norm(self.encoder(x))             # (B*M, N, D)

        # (6)(7) head + target
        y = self.head(x).view(B, M, self.horizon)
        y = y[:, self.target_idx, :]
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# TÍCH HỢP trong run_experiment:
#
#   from patchTST_CT import CT_PatchTST
#   ...
#   elif model_name == "CT_PATCHTST":
#       model = CT_PatchTST(
#           input_size  = ds["train"][0].shape[2],
#           context_len = lookback,
#           horizon     = horizon,
#           d_model     = p["hidden_size"],
#           n_layers    = p["num_layers"],
#           patch_len   = p.get("patch_len", 6),
#           stride      = p.get("stride", 6),
#           target_idx  = 0,
#           n_channel_layers = p.get("n_channel_layers", 1),
#       )
#
#   BEST_PARAMS["CT_PATCHTST"] = dict(BEST_PARAMS["PATCHTST"])
#
# Phân tích kênh-nhìn-kênh sau khi train (kênh 0..2 = 3 trạm mặn):
#   w = model.channel_blocks[0].last_attn.mean(dim=0)   # (M, M)
#   -> hàng 0 (BenLuc) cho biết nó "nghe" kênh nào nhiều nhất
# ============================================================


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("— Test 1: forward/backward các cấu hình —")
    for lb, hz in [(24, 12), (60, 60), (84, 24)]:
        m = CT_PatchTST(input_size=12, context_len=lb, horizon=hz)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  lookback={lb} horizon={hz} -> out {tuple(y.shape)} | "
              f"n_patches={m.n_patches} | params={n_params:,} | backward OK")

    print("— Test 2: channel attention map đúng shape và chuẩn hóa —")
    m = CT_PatchTST(input_size=12, context_len=36, horizon=12)
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 36, 12))
    w = m.channel_blocks[0].last_attn                 # (B*N, M, M)
    print(f"  attn shape: {tuple(w.shape)} | tổng hàng: "
          f"{w.sum(-1).min():.4f}/{w.sum(-1).max():.4f}")
    assert w.shape[-1] == 12 and abs(w.sum(-1).mean().item() - 1) < 1e-4

    print("— Test 3: tương thích use_future —")
    m3 = CT_PatchTST(input_size=12, context_len=24, horizon=12,
                     use_future=True)
    y3 = m3(torch.randn(4, 24, 12), torch.randn(4, 12, 6))
    F.mse_loss(y3, torch.randn(4, 12)).backward()
    print(f"  out {tuple(y3.shape)} | OK")

    print("\nTất cả sanity test PASS.")