"""
Patch_In_Patch.py — Patch Transformer hai luồng, trao đổi thông tin xen kẽ
===========================================================================
Hai luồng biểu diễn chạy song hành suốt mạng:
    Y : luồng CHI TIẾT   (B, P, S, d)  — mỗi điểm một vector   (vai trò intra)
    Z : luồng TỔNG QUÁT  (B, P, d)     — mỗi patch một vector  (vai trò inter)

Mỗi khối k = 1..n_blocks:
    [xuống]  Y <- Y + Down(Z)   (phát token patch về các điểm; chỉ giữa các khối,
                                 chỉ ở fusion_mode="interleave_both")
    intra    Y <- attention giữa S điểm trong mỗi patch      (intra_per_block tầng)
    [lên]    Z <- Z + Up(Y)     (tóm tắt các điểm của patch vào token patch)
    inter    Z <- attention giữa P token patch               (inter_per_block tầng)
Đầu ra: LayerNorm(Y + Z phát về S điểm) -> flatten -> Linear -> H -> RevIN đảo.

Cờ:
  fusion_mode : "parallel"        — không trao đổi; tương đương PatchTST_IntraTCN
                                    (mixed, intra attn, inter attn) với
                                    intra_layers = n_blocks*intra_per_block,
                                    inter_layers = n_blocks*inter_per_block
                "interleave_up"   — chỉ chiều lên (kiểu TNT)
                "interleave_both" — cả lên lẫn xuống (mặc định)
  up_mode     : "flatten" (mặc định, giữ vị trí từng điểm) | "attn" | "mean"
  zero_init_adapters : True -> Up/Down khởi tạo 0, nên lúc khởi tạo model xen kẽ
                       cho ra đúng như "parallel" rồi tự học cách trao đổi.
  revin_mode  : "all" | "target_only"
  stats_token : True (mặc định) — đưa (μ, log σ) của RevIN vào cả Y và Z.

Chỉ hỗ trợ trộn kênh sớm (mixed). Interface khớp pipeline:
    model = PatchInPatch(input_size=C, context_len=168, horizon=H, d_model=256,
                         patch_len=6, stride=6, target_idx=0)
    out = model(Xb)          # Xb: (B, context_len, C) -> (B, H)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# RevIN (không affine), có chế độ target_only và trả thống kê cho stats token.
# --------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps=1e-5, mode="all", target_idx=0):
        super().__init__()
        if mode not in ("all", "target_only"):
            raise ValueError(f"revin_mode không hợp lệ: {mode}")
        self.eps, self.mode, self.target_idx = eps, mode, target_idx
        self._mean = self._std = None

    def normalize(self, x):                        # (B, C, L)
        self._mean = x.mean(-1, keepdim=True)
        self._std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + self.eps)
        xn = (x - self._mean) / self._std
        if self.mode == "all":
            return xn
        mask = torch.zeros(1, x.shape[1], 1, device=x.device, dtype=x.dtype)
        mask[:, self.target_idx] = 1.0
        return xn * mask + x * (1.0 - mask)

    def stats(self):                               # (B, 2C): [μ ; log σ]
        return torch.cat([self._mean.squeeze(-1),
                          torch.log(self._std.squeeze(-1))], dim=-1)

    def denormalize_target(self, y, target_idx):   # (B, H)
        return y * self._std[:, target_idx, :] + self._mean[:, target_idx, :]


def _encoder(d_model, n_heads, n_layers, dropout):
    """Chồng tầng Transformer pre-norm, cùng siêu tham số với bản RIVF."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


# --------------------------------------------------------------------------
# Bộ chuyển LÊN: Y (B, P, S, d) -> (B, P, d), cộng vào token patch.
# --------------------------------------------------------------------------
class UpProj(nn.Module):
    def __init__(self, d_model, patch_len, mode="flatten", zero_init=True):
        super().__init__()
        if mode not in ("flatten", "attn", "mean"):
            raise ValueError(f"up_mode không hợp lệ: {mode}")
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        if mode == "flatten":
            self.proj = nn.Linear(patch_len * d_model, d_model)
        else:
            self.proj = nn.Linear(d_model, d_model)
        if mode == "attn":                         # pooling có trọng số học được
            self.q = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            self.key = nn.Linear(d_model, d_model)
            self.scale = d_model ** 0.5
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, Y):                          # (B, P, S, d)
        Yn = self.norm(Y)
        if self.mode == "flatten":
            return self.proj(Yn.flatten(2))        # (B, P, S*d) -> (B, P, d)
        if self.mode == "mean":
            return self.proj(Yn.mean(dim=2))
        a = torch.softmax((self.key(Yn) * self.q).sum(-1) / self.scale, dim=-1)  # (B,P,S)
        return self.proj((a.unsqueeze(-1) * Yn).sum(dim=2))


# --------------------------------------------------------------------------
# Bộ chuyển XUỐNG: Z (B, P, d) -> (B, P, 1, d), phát về S điểm của patch.
# --------------------------------------------------------------------------
class DownProj(nn.Module):
    def __init__(self, d_model, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, Z):                          # (B, P, d)
        return self.proj(self.norm(Z)).unsqueeze(2)


# --------------------------------------------------------------------------
# Mô hình chính.
# --------------------------------------------------------------------------
class PatchInPatch(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=256, patch_len=6, stride=6, target_idx=0,
                 n_heads=8, dropout=0.1,
                 n_blocks=2, intra_per_block=1, inter_per_block=1,
                 fusion_mode="interleave_both", up_mode="flatten",
                 zero_init_adapters=True,
                 revin_mode="all", stats_token=True,
                 use_future=False, **kwargs):
        super().__init__()
        if kwargs:   # print thay vì warnings (notebook đang tắt warnings)
            print(f"[CẢNH BÁO] PatchInPatch bỏ qua tham số không dùng: {sorted(kwargs)}")
        if fusion_mode not in ("parallel", "interleave_up", "interleave_both"):
            raise ValueError(f"fusion_mode không hợp lệ: {fusion_mode}")
        if min(n_blocks, intra_per_block, inter_per_block) < 1:
            raise ValueError("n_blocks, intra_per_block, inter_per_block phải >= 1")
        if d_model % n_heads != 0:
            n_heads = 1

        self.horizon, self.patch_len, self.stride = horizon, patch_len, stride
        self.target_idx, self.use_future, self.aux_loss = target_idx, use_future, None
        self.fusion_mode, self.n_blocks = fusion_mode, n_blocks
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        self.num_patch = (context_len + self.pad - patch_len) // stride + 1
        P, S, C = self.num_patch, patch_len, input_size

        self.revin = RevIN(mode=revin_mode, target_idx=target_idx)
        self.stats_proj = nn.Linear(2 * C, d_model) if stats_token else None

        # Embedding — cùng dạng với nhánh intra / inter của bản RIVF
        self.point_embed = nn.Linear(C, d_model)                 # mỗi điểm
        self.point_pos = nn.Parameter(torch.randn(1, S, d_model) * 0.02)
        self.patch_embed = nn.Linear(S * C, d_model)             # mỗi patch
        self.patch_pos = nn.Parameter(torch.randn(1, P, d_model) * 0.02)

        # Các khối: mỗi khối có intra_per_block tầng intra và inter_per_block tầng inter
        self.intra = nn.ModuleList(_encoder(d_model, n_heads, intra_per_block, dropout)
                                   for _ in range(n_blocks))
        self.inter = nn.ModuleList(_encoder(d_model, n_heads, inter_per_block, dropout)
                                   for _ in range(n_blocks))
        self.up = (nn.ModuleList(UpProj(d_model, S, up_mode, zero_init_adapters)
                                 for _ in range(n_blocks))
                   if fusion_mode != "parallel" else None)
        self.down = (nn.ModuleList(DownProj(d_model, zero_init_adapters)
                                   for _ in range(n_blocks - 1))
                     if fusion_mode == "interleave_both" else None)

        # Đầu ra — giống fuse_norm + head flatten của bản RIVF
        self.fuse_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                  nn.Linear(P * S * d_model, horizon))

    def _patchify(self, x):                        # (B, C, L) -> (B, C, P, S)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def encode(self, x):
        """Trả về hai luồng cuối (Y, Z) — dùng cho forward và cho đo CKA."""
        B = x.shape[0]
        xn = self.revin.normalize(x.transpose(1, 2))           # (B, C, L)
        p = self._patchify(xn).permute(0, 2, 3, 1)             # (B, P, S, C)
        P, S, C = p.shape[1], p.shape[2], p.shape[3]

        Y = self.point_embed(p) + self.point_pos               # (B, P, S, d)
        Z = self.patch_embed(p.reshape(B, P, S * C)) + self.patch_pos   # (B, P, d)
        if self.stats_proj is not None:
            s = self.stats_proj(self.revin.stats())            # (B, d)
            Y = Y + s[:, None, None, :]
            Z = Z + s[:, None, :]
        d = Y.shape[-1]

        for k in range(self.n_blocks):
            if self.down is not None and k > 0:                # xuống: bối cảnh -> điểm
                Y = Y + self.down[k - 1](Z)
            Y = self.intra[k](Y.reshape(B * P, S, d)).reshape(B, P, S, d)
            if self.up is not None:                            # lên: chi tiết -> patch
                Z = Z + self.up[k](Y)
            Z = self.inter[k](Z)
        return Y, Z

    def forward(self, x, future=None):             # (B, L, C) -> (B, H)
        Y, Z = self.encode(x)
        z = self.fuse_norm(Y + Z.unsqueeze(2))                 # (B, P, S, d)
        y = self.head(z)
        return self.revin.denormalize_target(y, self.target_idx)


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    B, L, C, H = 4, 168, 3, 6
    x = torch.randn(B, L, C)
    for fm in ["parallel", "interleave_up", "interleave_both"]:
        for um in ["flatten", "attn", "mean"]:
            for nb in [1, 2, 3]:
                m = PatchInPatch(input_size=C, context_len=L, horizon=H, d_model=256,
                                 fusion_mode=fm, up_mode=um, n_blocks=nb)
                y = m(x); y.sum().backward()
                ng = sum(p.grad is None for p in m.parameters())
                assert y.shape == (B, H) and ng == 0, (fm, um, nb, ng)
    for fm in ["parallel", "interleave_up", "interleave_both"]:
        m = PatchInPatch(input_size=C, context_len=L, horizon=H, d_model=256, fusion_mode=fm)
        print(f"  {fm:16s} n_blocks=2 up=flatten -> params={cnt(m):>10,}")
    print("OK.")
