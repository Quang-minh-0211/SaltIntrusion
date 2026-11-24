"""
TimesCross.py — dual-branch đa biến có ý thức về độ trễ
=======================================================
Mở rộng ý tưởng intra/inter của bài RIVF sang bài toán ĐA BIẾN, để bắt quan hệ dẫn–trễ
giữa các biến mà iTransformer và TimeXer không biểu diễn được.

Hai nhánh SONG SONG, chạy trên patch của MỌI biến (B batch, L=context_len, C biến,
H=horizon, P số patch, d=d_model, k=k_summary):

  Nhánh INTER (thô):  mỗi patch của mỗi biến -> 1 token (Linear(patch_len->d)),
                      gắn nhãn biến + vị trí patch, rồi self-attention giữa
                      - TẤT CẢ patch của mọi biến      (inter_scope="all")   hoặc
                      - chỉ các patch trong cùng biến  (inter_scope="ci").
                      Đầu ra: token patch tinh chỉnh -> head_inter -> (B,H).

  Nhánh INTRA (tinh): mỗi patch của mỗi biến -> k vector tóm tắt (k query học được
                      cross-attention vào patch_len điểm của patch), gắn nhãn biến +
                      vị trí patch -> tập k·P·C vector ngữ cảnh.
                      TỪNG điểm của biến mục tiêu (query) cross-attention với toàn bộ
                      tập ngữ cảnh đó. Đầu ra: điểm mục tiêu tinh chỉnh -> head_intra -> (B,H).

  Hợp nhất: ŷ = head_inter + head_intra  (cộng trong không gian dự báo, kiểu RIVF).

Cờ (một chuỗi mỗi thành phần, "none" để tắt — giống intra_mode/channel_mode của RIVF):
  inter_mode  : "attn" | "none"            bật/tắt nhánh inter
  intra_mode  : "attn" | "none"            bật/tắt nhánh intra  (phải còn >= 1 nhánh)
  inter_scope : "all"  | "ci"              inter trộn xuyên biến hay độc lập từng biến
  lag_tag     : True/False                 gắn embedding (biến, vị trí patch) — CƠ CHẾ TRỄ.
                                           Tắt -> rơi về kiểu "cùng thời điểm" như Crossformer.
  k_summary   : số vector tóm tắt mỗi patch (1 ~ gần TimeXer; lớn ~ gần giữ nguyên điểm)
  revin_mode  : "all" | "target_only"      chuẩn hóa mọi biến hay chỉ biến mục tiêu
  stats_token : True/False                 đưa (μ, log σ) mọi biến vào token (khôi phục mức nền)

Interface khớp pipeline 03_scenario3:
    model = TimesCross(input_size=C, context_len=168, horizon=H, target_idx=tgt_idx)
    out   = model(Xb)                      # Xb: (B, context_len, C) -> (B, H)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# RevIN (không affine), có target_only và thống kê cho stats token.
# --------------------------------------------------------------------------
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

    def stats(self):                                # (B, 2C): [μ ; log σ]
        return torch.cat([self._mean.squeeze(1), torch.log(self._std.squeeze(1))], dim=-1)

    def denormalize_target(self, y, target_idx):    # (B, H)
        return y * self._std[:, 0, target_idx:target_idx + 1] + self._mean[:, 0, target_idx:target_idx + 1]


class CrossBlock(nn.Module):
    """Một tầng cross-attention (query <- context) + FFN, pre-norm."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, q, kv):
        h = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)[0]
        q = q + h
        return q + self.ffn(self.ffn_norm(q))


class PatchSummarizer(nn.Module):
    """Mỗi patch (patch_len điểm) -> k vector, bằng k query học được cross-attention."""

    def __init__(self, d_model, n_heads, k, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pts):                          # pts: (M, patch_len, d) -> (M, k, d)
        q = self.queries.expand(pts.shape[0], -1, -1)
        out = self.attn(q, self.norm(pts), self.norm(pts), need_weights=False)[0]
        return out


class TimesCross(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=128, patch_len=6, stride=6, target_idx=0,
                 n_heads=8, n_layers=2, k_summary=4, dropout=0.1,
                 inter_mode="attn", intra_mode="attn", inter_scope="all",
                 lag_tag=True, revin_mode="all", stats_token=False,
                 use_future=False, **kwargs):
        super().__init__()
        if kwargs:
            print(f"[CẢNH BÁO] TimesCross bỏ qua tham số không dùng: {sorted(kwargs)}")
        if inter_mode not in ("attn", "none") or intra_mode not in ("attn", "none"):
            raise ValueError("inter_mode/intra_mode phải là 'attn' hoặc 'none'")
        if inter_mode == "none" and intra_mode == "none":
            raise ValueError("phải bật ít nhất một nhánh")
        if inter_scope not in ("all", "ci"):
            raise ValueError("inter_scope phải là 'all' hoặc 'ci'")
        if d_model % n_heads != 0:
            n_heads = 1

        self.horizon, self.patch_len, self.stride = horizon, patch_len, stride
        self.target_idx, self.use_future, self.aux_loss = target_idx, use_future, None
        self.inter_mode, self.intra_mode, self.inter_scope = inter_mode, intra_mode, inter_scope
        self.lag_tag = lag_tag
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        self.num_patch = (context_len + self.pad - patch_len) // stride + 1
        P, C = self.num_patch, input_size

        self.revin = RevIN(mode=revin_mode, target_idx=target_idx)
        self.stats_proj = nn.Linear(2 * C, d_model) if stats_token else None

        # nhãn dùng chung cho cả hai nhánh
        if lag_tag:
            self.var_emb = nn.Parameter(torch.randn(C, d_model) * 0.02)
            self.patchpos_emb = nn.Parameter(torch.randn(P, d_model) * 0.02)

        # ---- nhánh INTER -------------------------------------------------
        if inter_mode == "attn":
            self.patch_embed = nn.Linear(patch_len, d_model)
            enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
            self.inter_enc = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
            self.head_inter = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                            nn.Linear(P * d_model, horizon))

        # ---- nhánh INTRA -------------------------------------------------
        if intra_mode == "attn":
            self.summ_point_embed = nn.Linear(1, d_model)
            self.summ_pos = nn.Parameter(torch.randn(1, patch_len, d_model) * 0.02)
            self.summarizer = PatchSummarizer(d_model, n_heads, k_summary, dropout)
            self.q_point_embed = nn.Linear(1, d_model)
            self.q_pos = nn.Parameter(torch.randn(1, P * patch_len, d_model) * 0.02)
            self.intra_blocks = nn.ModuleList(CrossBlock(d_model, n_heads, dropout)
                                              for _ in range(n_layers))
            self.intra_norm = nn.LayerNorm(d_model)
            self.head_intra = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                            nn.Linear(P * patch_len * d_model, horizon))

    def _patchify(self, x):                          # (B, C, L) -> (B, C, P, patch_len)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x, future=None):               # (B, L, C) -> (B, H)
        B, _, C = x.shape
        xn = self.revin.normalize(x)                 # (B, L, C)
        patches = self._patchify(xn.transpose(1, 2))  # (B, C, P, patch_len)
        P, S = self.num_patch, self.patch_len
        cond = self.stats_proj(self.revin.stats()) if self.stats_proj is not None else None  # (B,d)

        y = 0.0

        # ---------- nhánh INTER ----------
        if self.inter_mode == "attn":
            tok = self.patch_embed(patches)          # (B, C, P, d)
            if self.lag_tag:
                tok = tok + self.var_emb[None, :, None, :] + self.patchpos_emb[None, None, :, :]
            if cond is not None:
                tok = tok + cond[:, None, None, :]
            if self.inter_scope == "all":
                z = tok.reshape(B, C * P, -1)
                z = self.inter_enc(z).reshape(B, C, P, -1)
            else:                                    # channel-independent: attention trong từng biến
                z = tok.reshape(B * C, P, -1)
                z = self.inter_enc(z).reshape(B, C, P, -1)
            y = y + self.head_inter(z[:, self.target_idx])      # (B, H)

        # ---------- nhánh INTRA ----------
        if self.intra_mode == "attn":
            # tóm tắt mỗi patch của mọi biến -> k vector
            pts = patches.reshape(B * C * P, S, 1)
            pts = self.summ_point_embed(pts) + self.summ_pos
            summ = self.summarizer(pts)                          # (B*C*P, k, d)
            k = summ.shape[1]
            summ = summ.reshape(B, C, P, k, -1)
            if self.lag_tag:
                summ = summ + self.var_emb[None, :, None, None, :] \
                            + self.patchpos_emb[None, None, :, None, :]
            if cond is not None:
                summ = summ + cond[:, None, None, None, :]
            context = summ.reshape(B, C * P * k, -1)             # (B, k·P·C, d)

            # query: từng điểm của biến mục tiêu
            tgt_pts = patches[:, self.target_idx].reshape(B, P * S, 1)
            q = self.q_point_embed(tgt_pts) + self.q_pos         # (B, P·S, d)
            if cond is not None:
                q = q + cond[:, None, :]
            for blk in self.intra_blocks:
                q = blk(q, context)
            y = y + self.head_intra(self.intra_norm(q))          # (B, H)

        return self.revin.denormalize_target(y, self.target_idx)


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    B, L, C = 4, 168, 3
    for H in (6, 12, 24):
        for im, ia, sc in [("attn", "attn", "all"), ("attn", "none", "all"),
                           ("none", "attn", "all"), ("attn", "attn", "ci")]:
            m = TimesCross(input_size=C, context_len=L, horizon=H, d_model=128, target_idx=1,
                           inter_mode=im, intra_mode=ia, inter_scope=sc, k_summary=4,
                           lag_tag=True, stats_token=True)
            x = torch.randn(B, L, C)
            out = m(x); out.sum().backward()
            ng = [n for n, p in m.named_parameters() if p.grad is None]
            assert out.shape == (B, H) and not ng, (H, im, ia, sc, ng)
        print(f"H={H:>2}  full params={cnt(m):>10,}")
    print("OK.")