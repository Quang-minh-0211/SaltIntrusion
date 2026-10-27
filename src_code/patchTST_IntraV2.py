"""
patchTST_IntraTCN_v2.py
====================================================================
Phiên bản tối ưu hóa cho bài toán cực trị chuỗi thời gian đa biến.
Đã giải quyết các nút thắt:
1. Bùng nổ tham số: Dùng Attention Pooling để nén S -> 1 thay vì Flatten P*S.
2. Over-engineering cục bộ: Thay TCN bằng Conv1D nhẹ + Attention Pooling.
3. Tương tác vật lý: Chuyển đổi từ Late Channel Mixing (trộn dự báo) 
   sang Early Feature Mixing (trộn đặc trưng) để học tương tác giữa các trạm/kênh.
====================================================================
"""

import math
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# RevIN - Giữ nguyên: Chống distribution shift
# --------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x):                       
        self._mean = x.mean(dim=-1, keepdim=True)
        self._std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return (x - self._mean) / self._std

    def denormalize_target(self, y, target_idx):  
        mean = self._mean[:, target_idx, :]
        std = self._std[:, target_idx, :]
        return y * std + mean


# --------------------------------------------------------------------------
# Attention Pooling: Tự động hội tụ trọng số vào điểm bất thường (đỉnh mặn)
# trong một patch ngắn, nén chiều S xuống 1 token duy nhất.
# --------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)  # Softmax dọc theo trục S
        )

    def forward(self, x):
        # x: (B*, S, d_model)
        attn_weights = self.attention(x)         # (B*, S, 1)
        # Nhân có trọng số và tổng hợp dọc theo S
        pooled = torch.sum(x * attn_weights, dim=1) # (B*, d_model)
        return pooled, attn_weights


# --------------------------------------------------------------------------
# Nhánh Intra-patch mới: Bắt hình thái cục bộ cực nhẹ
# Dùng Conv1D để lấy đạo hàm/sườn dốc cục bộ, sau đó nén bằng Attention Pooling
# --------------------------------------------------------------------------
class IntraPatchBlockV2(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        # Chiếu patch thô lên d_model và trộn cục bộ
        self.local_conv = nn.Conv1d(in_channels=1, out_channels=d_model, 
                                    kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.pool = AttentionPooling(d_model)

    def forward(self, x):
        # x: (B*, S)
        h = x.unsqueeze(1)                       # (B*, 1, S)
        h = self.local_conv(h)                   # (B*, d_model, S)
        h = self.activation(h)
        h = h.transpose(1, 2)                    # (B*, S, d_model)
        
        pooled_h, _ = self.pool(h)               # (B*, d_model)
        return pooled_h


# --------------------------------------------------------------------------
# Nhánh Inter-patch (PatchTST gốc)
# --------------------------------------------------------------------------
class InterPatchBlock(nn.Module):
    def __init__(self, patch_len, num_patch, d_model, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_patch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, patches):
        # patches: (B*, P, S)
        tok = self.embed(patches)                # (B*, P, d_model)
        tok = tok + self.pos
        tok = self.sa(tok)                       # (B*, P, d_model)
        return tok


# --------------------------------------------------------------------------
# Khối mô hình chính
# --------------------------------------------------------------------------
class PatchTST_IntraTCN_V2(nn.Module):
    def __init__(self,
                 input_size,               # C = 12
                 context_len,              # L = 168
                 horizon,                  # H = 24
                 d_model=128,
                 n_layers=3,
                 patch_len=6,
                 stride=6,
                 target_idx=0,             # Trạm mục tiêu
                 n_heads=8,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.context_len = context_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx

        if d_model % n_heads != 0:
            n_heads = max(1, d_model // (d_model // n_heads))
            
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        L_pad = context_len + self.pad
        self.num_patch = (L_pad - patch_len) // stride + 1

        self.revin = RevIN()
        
        # Intra và Inter
        self.intra = IntraPatchBlockV2(patch_len, d_model)
        self.inter = InterPatchBlock(patch_len, self.num_patch, d_model, 
                                     n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.fuse_norm = nn.LayerNorm(d_model)

        # Trộn kênh sớm (Early Channel Mixing): Trộn đặc trưng trước khi dự báo
        # Ánh xạ C kênh về 1 biểu diễn duy nhất
        self.channel_mixer = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Dropout(dropout)
        )

        # Đầu dự báo (Prediction Head): Đã hết bùng nổ tham số
        # Chỉ còn P * d_model thay vì P * S * d_model
        flat_dim = self.num_patch * d_model
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, horizon)
        )

    def _patchify(self, x):
        if self.pad > 0:
            x = nn.functional.pad(x, (0, self.pad), mode="replicate")
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x                                    # (B, C, P, S)

    def forward(self, x, future=None):
        B, L, C = x.shape
        x = x.transpose(1, 2)                       # (B, C, L)
        x = self.revin.normalize(x)                 

        patches = self._patchify(x)                 # (B, C, P, S)
        P, S = self.num_patch, self.patch_len

        # Chuyển (B, C) thành B*C để xử lý qua backbone
        pc = patches.reshape(B * C, P, S)           # (B*C, P, S)

        # --- Nhánh Inter ---
        inter = self.inter(pc)                      # (B*C, P, d_model)

        # --- Nhánh Intra (Nén S -> 1) ---
        pts = pc.reshape(B * C * P, S)              # (B*C*P, S)
        intra = self.intra(pts)                     # (B*C*P, d_model)
        intra = intra.reshape(B * C, P, -1)    # (B*C, P, d_model)

        # --- Hợp nhất hai nhánh ---
        # Không cần broadcast nữa, hai vector cùng không gian d_model
        z = self.fuse_norm(inter + intra)           # (B*C, P, d_model)
        
        # --- EARLY CHANNEL MIXING ---
        # Đưa về (B, P, d_model, C) để trộn thông tin giữa các trạm
        z = z.reshape(B, C, P, -1).permute(0, 2, 3, 1) # (B, P, d_model, C)
        
        # Trộn 12 kênh thành 1 biểu diễn hợp nhất mang thông tin tương tác
        z_mixed = self.channel_mixer(z).squeeze(-1)    # (B, P, d_model)

        # --- Đầu dự báo (Rất nhẹ) ---
        y = self.head(z_mixed)                         # (B, H)

        # --- Đảo chuẩn hoá ---
        y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ==========================================================================
# Cùng các metric đánh giá đỉnh
# ==========================================================================
def peak_metrics(preds, trues, pct=0.90):
    import numpy as np
    p = np.asarray(preds).flatten()
    t = np.asarray(trues).flatten()
    thr = np.quantile(t, pct)
    mask = t >= thr
    if mask.sum() == 0:
        return {"peak_rmse": float("nan"), "peak_mae": float("nan"),
                "peak_bias": float("nan"), "n_peak": 0}
    err = p[mask] - t[mask]
    return {
        "peak_rmse": float(np.sqrt((err ** 2).mean())),
        "peak_mae": float(np.abs(err).mean()),
        "peak_bias": float(err.mean()),
        "n_peak": int(mask.sum()),
        "peak_threshold": float(thr),
    }

# ==========================================================================
# Smoke test
# ==========================================================================
if __name__ == "__main__":
    B, L, C, H = 4, 168, 12, 24
    model = PatchTST_IntraTCN_V2(input_size=C, context_len=L, horizon=H,
                                 d_model=128, n_layers=3, patch_len=6, stride=6,
                                 target_idx=0)
    x = torch.randn(B, L, C)
    y = model(x)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}   (mong đợi ({B}, {H}))")
    print(f"tham số  : {n_par:,} (giảm thiểu đáng kể hiện tượng phình to layer Linear)")
    assert y.shape == (B, H), "SAI SHAPE!"
    print("OK.")