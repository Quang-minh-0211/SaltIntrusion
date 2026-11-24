"""
TimesCrossDecomp.py — TimesCross với phép tách thô/residual ở mức patch
=======================================================================
Sửa gốc bệnh trùng vai: hai nhánh nhận HAI đầu vào TÁCH BIỆT (trực giao).

Tách mỗi patch của mỗi biến thành:
    thô      = trung bình patch                 (thành phần chậm)
    residual = điểm − trung bình patch          (dao động tinh trong patch, tổng = 0)
  thô + residual = tín hiệu gốc, nên không mất thông tin, chỉ chia cho hai nhánh.

  Nhánh INTER (thô):  mỗi patch -> 1 token từ trung bình patch (Linear(1->d)),
                      gắn nhãn biến + vị trí patch, self-attention giữa các token patch
                      (inter_scope="all": xuyên biến; "ci": trong từng biến).
                      Head trên token patch của biến mục tiêu -> (B,H).

  Nhánh INTRA (residual): mỗi điểm residual -> 1 token (Linear(1->d)), gắn nhãn biến +
                      vị trí điểm, self-attention TRỰC TIẾP giữa các điểm residual
                      (residual_scope="all": xuyên biến; "ci": trong từng biến).
                      KHÔNG nén summary. Head trên điểm residual của biến mục tiêu -> (B,H).

  Hợp nhất: ŷ = head_inter + head_intra  (fuse_mode="sum"), hoặc inter làm xương sống và
  head_intra khởi tạo 0 (fuse_mode="residual").

Cờ:
  decomp        : True  -> inter ăn THÔ, intra ăn RESIDUAL (trực giao, mặc định)
                  False -> inter ăn patch ĐẦY ĐỦ, intra ăn điểm ĐẦY ĐỦ (để so sánh)
  inter_mode / intra_mode : "attn" | "none"
  inter_scope / residual_scope : "all" | "ci"
  fuse_mode     : "sum" | "residual"
  lag_tag       : True/False   nhãn (biến, vị trí)
  revin_mode    : "all" | "target_only"
  stats_token   : True/False

Interface khớp pipeline:
    model = TimesCrossDecomp(input_size=C, context_len=168, horizon=H, target_idx=tgt)
    out   = model(Xb)                      # (B, L, C) -> (B, H)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, eps=1e-5, mode="all", target_idx=0):
        super().__init__()
        if mode not in ("all", "target_only"):
            raise ValueError(f"revin_mode không hợp lệ: {mode}")
        self.eps, self.mode, self.target_idx = eps, mode, target_idx
        self._mean = self._std = None

    def normalize(self, x):                         # (B, L, C)
        self._mean = x.mean(1, keepdim=True)
        self._std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps)
        xn = (x - self._mean) / self._std
        if self.mode == "all":
            return xn
        mask = torch.zeros(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        mask[..., self.target_idx] = 1.0
        return xn * mask + x * (1.0 - mask)

    def stats(self):                                # (B, 2C)
        return torch.cat([self._mean.squeeze(1), torch.log(self._std.squeeze(1))], dim=-1)

    def denormalize_target(self, y, target_idx):    # (B, H)
        return y * self._std[:, 0, target_idx:target_idx + 1] + self._mean[:, 0, target_idx:target_idx + 1]


def _encoder(d_model, n_heads, n_layers, dropout):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class TimesCrossDecomp(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=128, patch_len=6, stride=6, target_idx=0,
                 n_heads=8, n_layers=2, dropout=0.1,
                 decomp=True, inter_mode="attn", intra_mode="attn",
                 inter_scope="all", residual_scope="all", fuse_mode="sum",
                 lag_tag=True, revin_mode="all", stats_token=False,
                 use_future=False, **kwargs):
        super().__init__()
        if kwargs:
            print(f"[CẢNH BÁO] TimesCrossDecomp bỏ qua tham số không dùng: {sorted(kwargs)}")
        for name, val in [("inter_mode", inter_mode), ("intra_mode", intra_mode)]:
            if val not in ("attn", "none"):
                raise ValueError(f"{name} phải là 'attn' hoặc 'none'")
        if inter_mode == "none" and intra_mode == "none":
            raise ValueError("phải bật ít nhất một nhánh")
        for name, val in [("inter_scope", inter_scope), ("residual_scope", residual_scope)]:
            if val not in ("all", "ci"):
                raise ValueError(f"{name} phải là 'all' hoặc 'ci'")
        if fuse_mode not in ("sum", "residual"):
            raise ValueError("fuse_mode phải là 'sum' hoặc 'residual'")
        if d_model % n_heads != 0:
            n_heads = 1

        self.horizon, self.patch_len, self.stride = horizon, patch_len, stride
        self.target_idx, self.use_future, self.aux_loss = target_idx, use_future, None
        self.decomp, self.inter_mode, self.intra_mode = decomp, inter_mode, intra_mode
        self.inter_scope, self.residual_scope, self.lag_tag = inter_scope, residual_scope, lag_tag
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        self.num_patch = (context_len + self.pad - patch_len) // stride + 1
        P, S, C = self.num_patch, patch_len, input_size
        self.n_point = P * S

        self.revin = RevIN(mode=revin_mode, target_idx=target_idx)
        self.stats_proj = nn.Linear(2 * C, d_model) if stats_token else None
        if lag_tag:
            self.var_emb = nn.Parameter(torch.randn(C, d_model) * 0.02)

        if inter_mode == "attn":
            self.inter_embed = nn.Linear(1 if decomp else S, d_model)   # thô: 1; đầy đủ: S
            self.patchpos_emb = nn.Parameter(torch.randn(P, d_model) * 0.02)
            self.inter_enc = _encoder(d_model, n_heads, n_layers, dropout)
            self.head_inter = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                            nn.Linear(P * d_model, horizon))
        if intra_mode == "attn":
            self.point_embed = nn.Linear(1, d_model)
            self.point_pos = nn.Parameter(torch.randn(self.n_point, d_model) * 0.02)
            self.intra_enc = _encoder(d_model, n_heads, n_layers, dropout)
            self.head_intra = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                            nn.Linear(self.n_point * d_model, horizon))
            if fuse_mode == "residual" and inter_mode == "attn":
                nn.init.zeros_(self.head_intra[-1].weight)
                nn.init.zeros_(self.head_intra[-1].bias)

    def _patchify(self, x):                          # (B, C, L) -> (B, C, P, S)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x, future=None):               # (B, L, C) -> (B, H)
        B, _, C = x.shape
        xn = self.revin.normalize(x)
        patches = self._patchify(xn.transpose(1, 2))  # (B, C, P, S)
        P, S = self.num_patch, self.patch_len
        cond = self.stats_proj(self.revin.stats()) if self.stats_proj is not None else None

        if self.decomp:
            coarse = patches.mean(-1, keepdim=True)   # (B, C, P, 1)
            resid = patches - coarse                  # (B, C, P, S), tổng mỗi patch = 0
            inter_in = coarse                          # (B, C, P, 1)
        else:
            resid = patches                            # (B, C, P, S)
            inter_in = patches                         # (B, C, P, S)

        y = 0.0

        # ---------- INTER trên THÔ ----------
        if self.inter_mode == "attn":
            tok = self.inter_embed(inter_in)           # (B, C, P, d)
            if self.lag_tag:
                tok = tok + self.var_emb[None, :, None, :] + self.patchpos_emb[None, None, :, :]
            if cond is not None:
                tok = tok + cond[:, None, None, :]
            if self.inter_scope == "all":
                z = self.inter_enc(tok.reshape(B, C * P, -1)).reshape(B, C, P, -1)
            else:
                z = self.inter_enc(tok.reshape(B * C, P, -1)).reshape(B, C, P, -1)
            y = y + self.head_inter(z[:, self.target_idx])

        # ---------- INTRA trên RESIDUAL (điểm trực tiếp) ----------
        if self.intra_mode == "attn":
            pts = resid.reshape(B, C, self.n_point, 1)
            h = self.point_embed(pts) + self.point_pos[None, None, :, :]   # (B, C, n_point, d)
            if self.lag_tag:
                h = h + self.var_emb[None, :, None, :]
            if cond is not None:
                h = h + cond[:, None, None, :]
            if self.residual_scope == "all":
                h = self.intra_enc(h.reshape(B, C * self.n_point, -1)).reshape(B, C, self.n_point, -1)
            else:
                h = self.intra_enc(h.reshape(B * C, self.n_point, -1)).reshape(B, C, self.n_point, -1)
            y = y + self.head_intra(h[:, self.target_idx])

        return self.revin.denormalize_target(y, self.target_idx)


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    B, L, C = 4, 168, 3
    for H in (6, 12, 24):
        for dc in (True, False):
            for sc in ("all", "ci"):
                m = TimesCrossDecomp(input_size=C, context_len=L, horizon=H, d_model=64,
                                     target_idx=1, decomp=dc, inter_scope=sc, residual_scope=sc)
                x = torch.randn(B, L, C)
                out = m(x); out.sum().backward()
                ng = [n for n, p in m.named_parameters() if p.grad is None]
                assert out.shape == (B, H) and not ng, (H, dc, sc, ng)
        print(f"H={H:>2}  full decomp params={cnt(m):>10,}")
    print("OK.")
