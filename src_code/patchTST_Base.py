# ============================================================
# patchtst.py — PatchTST chuẩn cho bài toán dự báo xâm nhập mặn
# Tương thích pipeline hiện có: forward(x) -> (B, horizon)
#   x: (batch, lookback, n_features)  — giống LSTM/GRU của bạn
#
# Thành phần đúng theo bài gốc (Nie et al., ICLR 2023):
#   1. RevIN            — chuẩn hóa từng cửa sổ, từng kênh (chống lệch mùa)
#   2. Channel independence — (B, L, M) -> (B*M, L)
#   3. Patching         — cắt chuỗi thành patch (mặc định 6 bước = 12h ≈ 1 con nước)
#   4. Linear embedding + learnable positional encoding
#   5. Transformer encoder (pre-norm, GELU)
#   6. Flatten head     — (n_patches * d_model) -> horizon, dùng chung mọi kênh
#   7. Lấy kênh target + RevIN denorm
#
# CẢI TIẾN (tùy chọn, use_future=True): nhánh hiệp biến tương lai —
# 6 đặc trưng chu kỳ (hour/lunar/year sin-cos) là TẤT ĐỊNH nên biết
# trước được cho mọi bước tương lai; nhánh này tiêm pha triều tương lai
# vào dự báo, zero-init để tại init tương đương baseline.
#
# LƯU Ý TÍCH HỢP: cần truyền context_len = lookback khi khởi tạo
# (xem hướng dẫn ở cuối file).
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPatchEmbed(nn.Module):
    """Nhúng patch bằng tích chập 1D — thay cho nn.Linear(patch_len -> d_model).
 
    Drop-in: vào (B*M, N, patch_len) -> ra (B*M, N, d_model), y hệt Linear cũ.
    Có .out_features để patchTST_CT.py / patchTST_CT_v2.py (đọc
    self.embed.out_features) không bị vỡ.
 
    Giữ được nhịp triều vì: conv trượt cùng một kernel dọc patch (bất biến
    tịnh tiến -> bắt cạnh lên/xuống con nước ở mọi pha) + phi tuyến GELU;
    sau conv KHÔNG pooling nên không làm phẳng lại dao động trong patch.
    """
 
    def __init__(self, patch_len: int, d_model: int,
                 conv_channels: int = 16, kernel_size: int = 3,
                 n_conv: int = 2):
        super().__init__()
        self.out_features = d_model
        self.patch_len = patch_len
        layers, in_ch = [], 1
        for _ in range(n_conv):
            layers += [
                nn.Conv1d(in_ch, conv_channels, kernel_size,
                          padding=kernel_size // 2),
                nn.GELU(),
            ]
            in_ch = conv_channels
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(conv_channels * patch_len, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bm, N, P = x.shape
        h = x.reshape(Bm * N, 1, P)
        h = self.conv(h)
        h = h.reshape(Bm * N, -1)
        h = self.proj(h)
        return h.reshape(Bm, N, self.out_features)

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Chuẩn hóa mỗi kênh của MỖI cửa sổ về mean 0, std 1 (tính trên trục thời
    gian), rồi trả ngược thống kê đó vào dự báo ở đầu ra. Giúp mô hình không
    bị lệch khi mức mặn nền thay đổi giữa các mùa (vd 2020 hạn nặng vs 2022).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, M)
        self.mean = x.mean(dim=1, keepdim=True).detach()          # (B, 1, M)
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()                                                 # (B, 1, M)
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize_target(self, y: torch.Tensor, target_idx: int) -> torch.Tensor:
        # y: (B, horizon) — dự báo của kênh target (đang ở không gian RevIN)
        if self.affine:
            y = (y - self.beta[target_idx]) / (self.gamma[target_idx] + self.eps)
        return y * self.std[:, 0, target_idx: target_idx + 1] \
                 + self.mean[:, 0, target_idx: target_idx + 1]


class PatchTST_BASE(nn.Module):
    """PatchTST (supervised, chế độ dự báo — "7 bước").

    Parameters
    ----------
    input_size  : số kênh đầu vào M (với Scenario 3 của bạn là 12)
    context_len : lookback L tính theo số bước (PHẢI truyền, vd 24 = 48h)
    horizon     : số bước dự báo H
    d_model     : chiều embedding (tương đương "hidden_size")
    n_layers    : số layer encoder
    n_heads     : số attention head (d_model phải chia hết cho n_heads)
    patch_len   : độ dài patch theo bước. Mặc định 6 bước = 12h ≈ 1 chu kỳ
                  bán nhật triều — mỗi token là một "con nước".
    stride      : bước trượt giữa các patch. stride = patch_len -> không
                  chồng lấn (chuẩn cho masked pretraining sau này).
    target_idx  : vị trí kênh target trong feature_cols
                  (Scenario 3: Salinity_BenLuc = cột 0)
    revin       : bật/tắt RevIN
    use_future  : bật nhánh hiệp biến tương lai. Khi True, forward nhận
                  thêm x_future: (B, horizon, 6) — 6 đặc trưng chu kỳ
                  tại các bước TƯƠNG LAI (biết trước vì tất định).
    n_future    : số đặc trưng tương lai (mặc định 6, nằm ở 6 CỘT CUỐI
                  của feature_cols: hour/lunar/year sin-cos)
    """

    def __init__(self, input_size: int, context_len: int, horizon: int,
                 d_model: int = 128, n_layers: int = 3, n_heads: int = 8,
                 patch_len: int = 6, stride: int = 6, d_ff: int = 256,
                 dropout: float = 0.2, head_dropout: float = 0.1,
                 target_idx: int = 0, revin: bool = True,
                 use_future: bool = False, n_future: int = 6, patch_embed: str = "linear", conv_channels: int = 16, embed_kernel: int = 3):
        super().__init__()
        if context_len < patch_len:
            raise ValueError(
                f"context_len ({context_len}) phải >= patch_len ({patch_len}). "
                f"Với lookback ngắn, hãy giảm patch_len (vd patch_len=3)."
            )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads}).")

        self.input_size = input_size
        self.context_len = context_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx

        # Số patch; nếu (L - P) không chia hết cho stride thì pad cuối chuỗi
        # bằng cách lặp giá trị cuối (replication) để phủ hết dữ liệu.
        leftover = (context_len - patch_len) % stride
        self.pad_len = 0 if leftover == 0 else stride - leftover
        self.n_patches = (context_len + self.pad_len - patch_len) // stride + 1

        self.revin = RevIN(input_size) if revin else None

        # (3) -> (4): chiếu patch P chiều lên d_model + positional encoding học được
        if patch_embed == "linear":
            self.embed = nn.Linear(patch_len, d_model)
        elif patch_embed == "conv":
            self.embed = ConvPatchEmbed(patch_len, d_model, conv_channels=conv_channels, kernel_size=embed_kernel)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # (5): Transformer encoder — pre-norm + GELU cho ổn định trên dữ liệu nhỏ
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # (6): flatten head — dùng chung cho mọi kênh (channel independence)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),                      # (.., N, D) -> (.., N*D)
            nn.Dropout(head_dropout),
            nn.Linear(self.n_patches * d_model, horizon),
        )

        # (6.5) Nhánh hiệp biến tương lai (zero-init => init = baseline).
        # Học một hiệu chỉnh điều hòa theo pha triều tương lai, cộng vào
        # dự báo TRƯỚC khi RevIN denorm (hiệu chỉnh trong không gian chuẩn hóa).
        self.use_future = use_future
        self.n_future = n_future
        if use_future:
            self.future_head = nn.Linear(horizon * n_future, horizon)
            nn.init.zeros_(self.future_head.weight)
            nn.init.zeros_(self.future_head.bias)

        # Để train_model của bạn không phải sửa (giống UDF_LSTM)
        self.aux_loss = None

    # --------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, L) -> (B*M, n_patches, patch_len)"""
        if self.pad_len > 0:
            x = F.pad(x.unsqueeze(1), (0, self.pad_len), mode="replicate").squeeze(1)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, M) ; x_future: (B, horizon, n_future) hoặc None
        B, L, M = x.shape
        assert L == self.context_len, (
            f"Model được build với context_len={self.context_len} "
            f"nhưng nhận đầu vào L={L}. Truyền đúng lookback khi khởi tạo."
        )

        # (0) RevIN normalize
        if self.revin is not None:
            x = self.revin.normalize(x)

        # (2) Channel independence: (B, L, M) -> (B*M, L)
        x = x.permute(0, 2, 1).reshape(B * M, L)

        # (3) Patching: (B*M, N, P)
        x = self._patchify(x)

        # (4) Embedding + vị trí: (B*M, N, D)
        x = self.embed_dropout(self.embed(x) + self.pos)

        # (5) Encoder: (B*M, N, D)
        x = self.encoder_norm(self.encoder(x))

        # (6) Head: (B*M, N, D) -> (B*M, H) -> (B, M, H)
        y = self.head(x).view(B, M, self.horizon)

        # (7) Lấy kênh target
        y = y[:, self.target_idx, :]                       # (B, H)

        # (6.5) Cộng hiệu chỉnh từ pha triều tương lai (nếu bật)
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))

        # (7b) Denorm
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# HƯỚNG DẪN TÍCH HỢP VÀO NOTEBOOK
# ============================================================
# (1) Trong run_experiment, PatchTST cần biết lookback nên build trực tiếp
#     thay vì qua build_model:
#
#     from patchtst import PatchTST
#     ...
#     if model_name == "PATCHTST":
#         model = PatchTST(
#             input_size  = ds["train"][0].shape[2],
#             context_len = lookback,
#             horizon     = horizon,
#             d_model     = p["hidden_size"],
#             n_layers    = p["num_layers"],
#             patch_len   = p.get("patch_len", 6),
#             stride      = p.get("stride", 6),
#             target_idx  = 0,          # Salinity_BenLuc là cột 0 trong SC3
#             use_future  = p.get("use_future", False),
#         )
#     else:
#         model = build_model(model_name, ds["train"][0].shape[2], horizon,
#                             hidden_size=p["hidden_size"],
#                             num_layers=p["num_layers"], **extra)
#
# (2) Siêu tham số (LƯU Ý: Transformer cần learning rate THẤP hơn LSTM):
#
#     BEST_PARAMS["PATCHTST"] = {
#         "hidden_size":   128,     # d_model
#         "num_layers":    3,
#         "learning_rate": 1e-4,    # KHÔNG dùng 1e-3 — dễ phân kỳ
#         "batch_size":    32,
#         "patch_len":     6,       # 12h ~ 1 con nước
#         "stride":        6,
#         "use_future":    True,    # bật nhánh hiệp biến tương lai
#     }
#
#     rồi thêm "PATCHTST" vào MODEL_NAMES.
#
# (3) Khi use_future=True, pipeline notebook phải cắt thêm mảng tương lai
#     (B, horizon, 6) từ 6 CỘT CUỐI của feature_cols — 5 chỗ sửa:
#     make_windows trả thêm Fs; make_windows_gapaware gom thêm Fa;
#     build_dataset trả bộ ba (X, y, F); SalinityDataset nhận 3 mảng;
#     train_model/evaluate unpack 3 tensor và gọi:
#         out = model(Xb, Fb) if getattr(model, "use_future", False) else model(Xb)
#     (chi tiết từng đoạn đã gửi ở tin nhắn trước — RNN/LSTM/GRU không có
#     use_future nên tự đi nhánh cũ, baseline không bị ảnh hưởng.)
#
# (4) Khuyến nghị lookback: với patch_len=6, lookback=6 chỉ tạo 1 token
#     (attention vô nghĩa), lookback=12 tạo 2 token. PatchTST phát huy sức
#     mạnh với lookback dài: nên thêm 84 (168h = 14 token) vào LOOKBACKS,
#     hoặc dùng patch_len=3 cho các lookback ngắn.
#
# (5) Đọc mức độ sử dụng nhánh tương lai sau khi train:
#         model.future_head.weight.norm()    # ~0 = mô hình không dùng
# ============================================================


if __name__ == "__main__":
    # Sanity test nhanh: chạy `python patchtst.py`
    torch.manual_seed(0)

    print("— Test 1: chế độ thường (không future) —")
    for lb, hz, pl in [(24, 12, 6), (12, 6, 6), (6, 6, 3), (84, 24, 6)]:
        m = PatchTST(input_size=12, context_len=lb, horizon=hz, patch_len=pl)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        print(f"  lookback={lb:3d} horizon={hz:3d} patch={pl} "
              f"-> out {tuple(y.shape)} | n_patches={m.n_patches} | backward OK")

    print("— Test 2: use_future=True, forward + backward —")
    m = PatchTST(input_size=12, context_len=24, horizon=12, use_future=True)
    x, xf = torch.randn(4, 24, 12), torch.randn(4, 12, 6)
    y = m(x, xf)
    F.mse_loss(y, torch.randn(4, 12)).backward()
    g = m.future_head.weight.grad.abs().sum().item()
    print(f"  out {tuple(y.shape)} | grad future_head = {g:.4f} (phải > 0) | OK")

    print("— Test 3: zero-init => use_future=True tại init trùng baseline —")
    torch.manual_seed(1)
    base = PatchTST(input_size=12, context_len=24, horizon=12, use_future=False)
    torch.manual_seed(1)
    ext = PatchTST(input_size=12, context_len=24, horizon=12, use_future=True)
    base.eval(); ext.eval()
    x, xf = torch.randn(8, 24, 12), torch.randn(8, 12, 6)
    with torch.no_grad():
        diff = (base(x) - ext(x, xf)).abs().max().item()
    print(f"  chênh lệch tại init: {diff:.2e} (kỳ vọng ~0)")
    assert diff < 1e-5, "Zero-init nhánh future không hoạt động đúng!"

    print("— Test 4: use_future=True nhưng quên truyền x_future -> vẫn chạy —")
    y2 = ext(x)
    print(f"  out {tuple(y2.shape)} | OK (nhánh future bị bỏ qua an toàn)")

    print("\nTất cả sanity test PASS.")
