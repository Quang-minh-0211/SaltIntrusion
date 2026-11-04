"""
patchTST_IntraTCN.py  —  bản gộp 3 fix dưới dạng cờ để ablation
====================================================================
Biến thể PatchTST cho bài toán xâm nhập mặn. Ba trục thay đổi, mỗi trục
một cờ, mặc định giữ hành vi để tái lập baseline khi cần:

  FIX 1  intra_mode   : "conv" | "attn" | "tcn_attn" | "none"
         Nhánh intra-patch (Gap 1). Trên patch ngắn, "attn"/"conv" gọn
         hơn nhiều so với "tcn_attn" (bản TCN+SA nặng, dễ hại).

  FIX 2  pool_mode    : "flatten" | "max" | "mean" | "attn"
         Gộp dọc trục S trước head. "flatten" giữ trọn P*S*d_model (head
         to). "max"/"mean"/"attn" đưa head về P*d_model (gọn như gốc).

  FIX 3  channel_mode : "ci_late" | "mixed"
         "ci_late" = channel-independent + Linear(C->1) ở cuối (dự báo 12
         kênh độc lập rồi cộng tuyến tính — SAI vật lý cho bài đa biến 1
         mục tiêu). "mixed" = mỗi patch mang CẢ C kênh, trộn kênh NGAY khi
         trích đặc trưng (Conv/attention/Linear trên chiều kênh), ra thẳng
         1 chuỗi mục tiêu, không cộng kênh cuối.

Interface khớp pipeline:
    model = PatchTST_IntraTCN(input_size=C, context_len=168, horizon=24,
                              d_model=128, n_layers=3, patch_len=6, stride=6,
                              target_idx=0,
                              intra_mode="attn", pool_mode="flatten",
                              channel_mode="mixed")
    out = model(Xb)          # Xb: (B, context_len, input_size) -> (B, horizon)
====================================================================
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# RevIN (không affine): chuẩn hoá theo cửa sổ, từng kênh; đảo bằng thống kê
# của kênh mục tiêu (target_idx).
# --------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x):                       # x: (B, C, L)
        self._mean = x.mean(dim=-1, keepdim=True)
        self._std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return (x - self._mean) / self._std

    def denormalize_target(self, y, target_idx):  # y: (B, H)
        mean = self._mean[:, target_idx, :]
        std = self._std[:, target_idx, :]
        return y * std + mean


# --------------------------------------------------------------------------
# TCN: tích chập giãn nở có residual. Vào (B*, in_ch, S) -> ra (B*, d_model, S).
# (dùng cho intra_mode="tcn_attn"; in_ch=1 ở ci_late, =C ở mixed)
# --------------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, d_model, in_ch=1, n_levels=2, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        c = in_ch
        for i in range(n_levels):
            dilation = 2 ** i
            pad = dilation * (kernel_size - 1) // 2
            layers += [nn.Conv1d(c, d_model, kernel_size, padding=pad, dilation=dilation),
                       nn.GELU(), nn.Dropout(dropout)]
            c = d_model
        self.net = nn.ModuleList(layers)
        self.res_proj = nn.Conv1d(in_ch, d_model, 1)

    def forward(self, x):                          # x: (B*, in_ch, S)
        res = self.res_proj(x)
        out = x
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, nn.Dropout):
                out = out + res
                res = out
        return out                                 # (B*, d_model, S)


# --------------------------------------------------------------------------
# FIX 1 — Khối intra-patch. Vào (B*, S, in_ch) -> ra (B*, S, d_model) | None.
# Ở channel_mode="mixed", in_ch=C nên Conv/Linear/TCN TỰ TRỘN KÊNH tại đây.
# --------------------------------------------------------------------------
class IntraPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, intra_mode="conv", in_ch=1,
                 n_heads=8, tcn_levels=2, dropout=0.1):
        super().__init__()
        self.mode = intra_mode
        if intra_mode == "none":
            return

        if intra_mode == "conv":
            self.conv = nn.Conv1d(in_ch, d_model, kernel_size=3,
                                  padding=1, padding_mode="replicate")
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            return

        # các chế độ có self-attention
        self.pos = nn.Parameter(torch.randn(1, patch_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=1, enable_nested_tensor=False)
        if intra_mode == "attn":
            self.embed = nn.Linear(in_ch, d_model)          # trộn kênh nếu in_ch=C
        elif intra_mode == "tcn_attn":
            self.tcn = TCN(d_model, in_ch=in_ch, n_levels=tcn_levels, dropout=dropout)
        else:
            raise ValueError(f"intra_mode không hợp lệ: {intra_mode}")

    def forward(self, x):                          # x: (B*, S, in_ch)
        if self.mode == "none":
            return None
        if self.mode == "conv":
            h = self.conv(x.transpose(1, 2))       # (B*, d_model, S)
            return self.drop(self.act(h)).transpose(1, 2)     # (B*, S, d_model)
        if self.mode == "attn":
            h = self.embed(x)                      # (B*, S, d_model)
            return self.sa(h + self.pos)
        # tcn_attn
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)       # (B*, S, d_model)
        return self.sa(h + self.pos)


# --------------------------------------------------------------------------
# FIX 2 — Attention pooling dọc S: (N, P, S, d_model) -> (N, P, d_model).
# --------------------------------------------------------------------------
class AttnPoolS(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, z):                          # (N, P, S, d_model)
        k = self.key(z)
        att = (k * self.q).sum(-1) / self.scale    # (N, P, S)
        att = torch.softmax(att, dim=-1)
        return (att.unsqueeze(-1) * z).sum(dim=2)  # (N, P, d_model)


# --------------------------------------------------------------------------
# Khối inter-patch (kiểu PatchTST): mỗi patch -> 1 token d_model, SA giữa P
# token, rồi broadcast ra S. Vào (B*, P, S, in_ch) -> ra (B*, P, d_model).
# Ở mixed, in_ch=C nên embed Linear(S*C -> d_model) cũng trộn kênh.
# --------------------------------------------------------------------------
class InterPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, num_patch, in_ch=1,
                 n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(patch_len * in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_patch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, patches):                    # (B*, P, S, in_ch)
        B_, P, S, ic = patches.shape
        tok = self.embed(patches.reshape(B_, P, S * ic))      # (B*, P, d_model)
        return self.sa(tok + self.pos)             # (B*, P, d_model)


# --------------------------------------------------------------------------
# HƯỚNG 3 — Inter-patch bằng TOKEN-MIXING (MLP-Mixer) thay cho self-attention.
# Giữ ĐÚNG interface của InterPatchBlock:
#     Vào (B*, P, S, in_ch)  ->  Ra (B*, P, d_model)
#
# Ý tưởng: embed mỗi patch -> 1 token d_model (y hệt InterPatchBlock), rồi
# TRỘN GIỮA P patch bằng một MLP TĨNH trên trục patch (token-mixing) thay vì
# attention động. Hợp lý khi quan hệ giữa các patch ổn định theo vị trí/pha
# triều (không biến động theo nội dung như trong NLP).
#
#   token-mixing : ma trận trộn P x P cố định, học được  (RẺ: ~2*P*hidden)
#   channel_mix  : (tuỳ chọn) thêm MLP trộn trục đặc trưng d, kiểu Mixer đầy đủ.
#                  Mặc định TẮT, vì channel-mixing đã được FFN ở nơi khác lo.
# --------------------------------------------------------------------------
class InterPatchMixerBlock(nn.Module):
    def __init__(self, d_model, patch_len, num_patch, in_ch=1,
                 n_layers=1, token_hidden=None, channel_mix=False,
                 channel_hidden=None, dropout=0.1, **kwargs):
        super().__init__()
        self.embed = nn.Linear(patch_len * in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_patch, d_model) * 0.02)
        token_hidden = token_hidden or max(num_patch * 2, 16)
        channel_hidden = channel_hidden or d_model * 2
        self.channel_mix = channel_mix

        self.token_norms = nn.ModuleList()
        self.token_mlps = nn.ModuleList()
        self.chan_norms = nn.ModuleList() if channel_mix else None
        self.chan_mlps = nn.ModuleList() if channel_mix else None
        for _ in range(n_layers):
            self.token_norms.append(nn.LayerNorm(d_model))
            # token-MLP hoạt động TRÊN TRỤC P (num_patch): P -> hidden -> P
            self.token_mlps.append(nn.Sequential(
                nn.Linear(num_patch, token_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(token_hidden, num_patch),
                nn.Dropout(dropout)))
            if channel_mix:
                self.chan_norms.append(nn.LayerNorm(d_model))
                self.chan_mlps.append(nn.Sequential(
                    nn.Linear(d_model, channel_hidden), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(channel_hidden, d_model),
                    nn.Dropout(dropout)))

    def forward(self, patches):                    # (B*, P, S, in_ch)
        B_, P, S, ic = patches.shape
        x = self.embed(patches.reshape(B_, P, S * ic))        # (B*, P, d)
        x = x + self.pos
        for i in range(len(self.token_mlps)):
            # --- token-mixing: trộn dọc trục P (giữa các patch) ---
            y = self.token_norms[i](x)             # (B*, P, d)
            y = y.transpose(1, 2)                  # (B*, d, P)  MLP chạy trên P
            y = self.token_mlps[i](y)              # (B*, d, P)
            y = y.transpose(1, 2)                  # (B*, P, d)
            x = x + y                              # residual
            # --- channel-mixing (tuỳ chọn): trộn trục đặc trưng d ---
            if self.channel_mix:
                y = self.chan_norms[i](x)
                y = self.chan_mlps[i](y)           # (B*, P, d) MLP chạy trên d
                x = x + y                          # residual
        return x                                   # (B*, P, d_model)


# --------------------------------------------------------------------------
# HƯỚNG 2 — Cross-attention bất đối xứng tầm xa.
# Mỗi patch phát k query (phái sinh nội dung) đọc TOÀN BỘ L=P*S điểm ở độ
# phân giải điểm, rồi rót ngược ngữ cảnh về từng điểm. Chi phí O(P*k*L)
# thay vì O(L^2). Residual nên tệ nhất là trung tính (không hại).
#   Vào  z: (N, P, S, d)  ->  Ra z': (N, P, S, d)   (N = B*C hoặc B)
# --------------------------------------------------------------------------
class LongRangeCrossAttention(nn.Module):
    def __init__(self, d_model, num_patch, patch_len, n_queries=4,
                 n_heads=8, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            n_heads = 1
        self.k = n_queries
        self.h = n_heads
        self.dh = d_model // n_heads
        L = num_patch * patch_len

        self.to_q = nn.Linear(d_model, n_queries * d_model)   # k query/ patch
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.pos_kv = nn.Parameter(torch.randn(1, L, d_model) * 0.02)  # vị trí ~ pha
        self.out = nn.Linear(d_model, d_model)
        self.point_q = nn.Linear(d_model, d_model)            # rót ngược về điểm
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):                          # z: (N, P, S, d)
        N, P, S, d = z.shape
        h, dh = self.h, self.dh
        Z = z.reshape(N, P * S, d)                 # kho điểm (N, L, d)

        K = self.to_k(Z) + self.pos_kv             # key mức điểm + mã hoá vị trí
        V = self.to_v(Z)
        s = z.mean(dim=2)                          # tóm tắt patch (N, P, d)
        Q = self.to_q(s).reshape(N, P * self.k, d) # (N, P*k, d)

        def split(t):                              # (N, T, d) -> (N, h, T, dh)
            return t.reshape(N, -1, h, dh).transpose(1, 2)
        att = (split(Q) @ split(K).transpose(-2, -1)) / (dh ** 0.5)   # (N,h,P*k,L)
        att = self.drop(torch.softmax(att, dim=-1))
        ctx = (att @ split(V)).transpose(1, 2).reshape(N, P, self.k, d)  # (N,P,k,d)
        ctx = self.out(ctx)

        # rót ngược: mỗi điểm attend tới k ngữ cảnh của patch mình
        pq = self.point_q(z)                                   # (N,P,S,d)
        sc = (pq.unsqueeze(3) * ctx.unsqueeze(2)).sum(-1) / (d ** 0.5)   # (N,P,S,k)
        sc = torch.softmax(sc, dim=-1)
        z_ctx = (sc.unsqueeze(-1) * ctx.unsqueeze(2)).sum(3)   # (N,P,S,d)

        return self.norm(z + z_ctx)                # residual


# --------------------------------------------------------------------------
# Mô hình chính.
# --------------------------------------------------------------------------
class PatchTST_IntraTCN(nn.Module):
    def __init__(self,
                 input_size, context_len, horizon,
                 d_model=128, n_layers=3, patch_len=6, stride=6, target_idx=0,
                 n_heads=8,
                 intra_mode="conv",        # FIX 1
                 pool_mode="flatten",      # FIX 2
                 channel_mode="ci_late",   # FIX 3: "ci_late" | "mixed"
                 longrange=False,          # HƯỚNG 2: bật cross-attention tầm xa
                 n_queries=4,              # số query mỗi patch cho khối tầm xa
                 tcn_levels=2, dropout=0.1,
                 inter_layers=None,        # số lớp inter-patch; None -> dùng n_layers
                 inter_mode="attn",        # HƯỚNG 3: "attn" (gốc) | "mixer" (token-mixing)
                 inter_channel_mix=False,  # chỉ dùng khi inter_mode="mixer"
                 inter_token_hidden=None,  # hidden cho token-MLP (None -> 2*P)
                 use_future=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.context_len = context_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx
        self.use_future = use_future
        self.aux_loss = None
        self.intra_mode = intra_mode
        self.pool_mode = pool_mode
        self.channel_mode = channel_mode

        if d_model % n_heads != 0:
            n_heads = 1
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        self.num_patch = (context_len + self.pad - patch_len) // stride + 1
        P = self.num_patch

        # FIX 3: số kênh vào mỗi khối. ci_late xử lý B*C luồng đơn biến (in_ch=1);
        # mixed xử lý B luồng, mỗi patch mang cả C kênh (in_ch=C).
        block_in_ch = 1 if channel_mode == "ci_late" else input_size

        self.revin = RevIN()
        self.intra = IntraPatchBlock(d_model, patch_len, intra_mode=intra_mode,
                                     in_ch=block_in_ch, n_heads=n_heads,
                                     tcn_levels=tcn_levels, dropout=dropout)
        _inter_L = inter_layers if inter_layers is not None else n_layers
        if inter_mode == "attn":
            self.inter = InterPatchBlock(d_model, patch_len, P, in_ch=block_in_ch,
                                         n_heads=n_heads, n_layers=_inter_L,
                                         dropout=dropout)
        elif inter_mode == "mixer":
            self.inter = InterPatchMixerBlock(d_model, patch_len, P, in_ch=block_in_ch,
                                              n_layers=_inter_L,
                                              token_hidden=inter_token_hidden,
                                              channel_mix=inter_channel_mix,
                                              dropout=dropout)
        else:
            raise ValueError(f"inter_mode không hợp lệ: {inter_mode}")
        self.fuse_norm = nn.LayerNorm(d_model)

        # HƯỚNG 2: khối cross-attention tầm xa (đặt sau fuse, trước pool)
        self.longrange = (LongRangeCrossAttention(d_model, P, patch_len,
                          n_queries=n_queries, n_heads=n_heads, dropout=dropout)
                          if longrange else None)

        # FIX 2: pooling dọc S
        self.pool = AttnPoolS(d_model) if pool_mode == "attn" else None
        if pool_mode == "flatten":
            head_in = P * patch_len * d_model
        elif pool_mode in ("max", "mean", "attn"):
            head_in = P * d_model
        else:
            raise ValueError(f"pool_mode không hợp lệ: {pool_mode}")
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                  nn.Linear(head_in, horizon))

        # FIX 3: chỉ ci_late mới cộng kênh cuối
        self.channel_mix = nn.Linear(input_size, 1) if channel_mode == "ci_late" else None

    def _patchify(self, x):                         # (B, C, L) -> (B, C, P, S)
        if self.pad > 0:
            x = nn.functional.pad(x, (0, self.pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def _pool(self, z):                             # (N, P, S, d) -> (N, P, S, d) | (N, P, d)
        if self.pool_mode == "flatten":
            return z
        if self.pool_mode == "max":
            return z.max(dim=2).values
        if self.pool_mode == "mean":
            return z.mean(dim=2)
        return self.pool(z)                         # attn

    def forward(self, x, future=None):
        B, L, C = x.shape
        x = self.revin.normalize(x.transpose(1, 2))          # (B, C, L)
        patches = self._patchify(x)                          # (B, C, P, S)
        P, S = self.num_patch, self.patch_len

        if self.channel_mode == "ci_late":
            # ---- channel-independent: B*C luồng đơn biến ----
            pc = patches.reshape(B * C, P, S).unsqueeze(-1)  # (B*C, P, S, 1)
            inter = self.inter(pc)                           # (B*C, P, d)
            inter = inter.unsqueeze(2).expand(-1, -1, S, -1) # (B*C, P, S, d)
            intra = self.intra(pc.reshape(B * C * P, S, 1))  # (B*C*P, S, d) | None
            if intra is not None:
                intra = intra.reshape(B * C, P, S, -1)
                z = self.fuse_norm(intra + inter)
            else:
                z = self.fuse_norm(inter)
            if self.longrange is not None:                   # HƯỚNG 2
                z = self.longrange(z)
            y = self.head(self._pool(z))                     # (B*C, H)
            y = y.reshape(B, C, self.horizon)                # (B, C, H)
            y = self.channel_mix(y.transpose(1, 2)).squeeze(-1)   # (B, H)
        else:
            # ---- mixed: mỗi patch mang cả C kênh, trộn kênh sớm ----
            pm = patches.permute(0, 2, 3, 1)                 # (B, P, S, C)
            inter = self.inter(pm)                           # (B, P, d)
            inter = inter.unsqueeze(2).expand(-1, -1, S, -1) # (B, P, S, d)
            intra = self.intra(pm.reshape(B * P, S, C))      # (B*P, S, d) | None
            if intra is not None:
                intra = intra.reshape(B, P, S, -1)
                z = self.fuse_norm(intra + inter)
            else:
                z = self.fuse_norm(inter)
            if self.longrange is not None:                   # HƯỚNG 2
                z = self.longrange(z)
            y = self.head(self._pool(z))                     # (B, H) trực tiếp

        return self.revin.denormalize_target(y, self.target_idx)


# ==========================================================================
# Chỉ số vùng đỉnh — dùng cạnh RMSE/MAE tổng (đỉnh hiếm nên RMSE tổng che mất).
# ==========================================================================
def peak_metrics(preds, trues, pct=0.90):
    import numpy as np
    p = np.asarray(preds).flatten(); t = np.asarray(trues).flatten()
    thr = np.quantile(t, pct); mask = t >= thr
    if mask.sum() == 0:
        return {"peak_rmse": float("nan"), "n_peak": 0}
    err = p[mask] - t[mask]
    return {"peak_rmse": float(np.sqrt((err ** 2).mean())),
            "peak_mae": float(np.abs(err).mean()),
            "peak_bias": float(err.mean()),        # <0 = dự báo HỤT đỉnh
            "n_peak": int(mask.sum()), "peak_threshold": float(thr)}


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    B, L, C, H = 4, 168, 12, 24
    x = torch.randn(B, L, C)
    print("Kiểm tra tổ hợp cờ:")
    for cm in ["ci_late", "mixed"]:
        for im in ["conv", "attn", "none"]:
            for pm in ["flatten", "max"]:
                m = PatchTST_IntraTCN(input_size=C, context_len=L, horizon=H,
                                      d_model=128, n_layers=3, patch_len=6, stride=6,
                                      target_idx=0, intra_mode=im, pool_mode=pm,
                                      channel_mode=cm)
                y = m(x)
                assert y.shape == (B, H), (cm, im, pm)
                n = sum(p.numel() for p in m.parameters())
                print(f"  channel={cm:8s} intra={im:5s} pool={pm:8s} "
                      f"-> out={tuple(y.shape)} params={n:>10,}")
    print("OK.")