"""
patchTST_decomp.py — PatchTST phân rã trend/seasonal + Auto-Correlation
=======================================================================
Kiến trúc theo thiết kế đã thống nhất:

  RevIN -> Series Decomposition (trend, seasonal)
    TREND    -> patchify -> inter (self-attention giữa các patch)   -> trend_rep (B,P,d)
    SEASONAL -> | Auto-Correlation (point-level)  -> period (B,P,S,d)   (tính chu kỳ)
                | intra (attention trong patch)   -> fast   (B,P,S,d)   (biến động nhanh)
                -> gated fusion(period, fast)      -> seasonal_rep (B,P,S,d)
  Dual head (cộng ở KHÔNG GIAN DỰ BÁO):
    ŷ = head_seasonal(flatten seasonal_rep) + head_trend(flatten trend_rep)
  RevIN denormalize (kênh target_idx)

Ghi chú:
  - channel mixing SỚM (mixed): intra/inter/period_embed đều nhận in_ch=C.
  - Auto-Correlation chạy point-level trên chuỗi L rồi reshape về (B,P,S,d);
    điều này yêu cầu stride == patch_len (patch không chồng lấn) để căn thời
    gian giữa `period` và `fast` khớp từng điểm. Có assert ở __init__.

Interface:
  model = PatchTST_Decomp(input_size=C, context_len=168, horizon=18,
                          d_model=256, patch_len=6, stride=6, target_idx=0)
  out = model(Xb)            # Xb: (B, context_len, C) -> (B, horizon)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# RevIN (không affine): chuẩn hoá theo cửa sổ, từng kênh; đảo bằng thống kê
# của kênh mục tiêu.
# --------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x):                        # (B, C, L)
        self._mean = x.mean(-1, keepdim=True)
        self._std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + self.eps)
        return (x - self._mean) / self._std

    def denormalize_target(self, y, target_idx):   # y: (B, H)
        return y * self._std[:, target_idx, :] + self._mean[:, target_idx, :]


# --------------------------------------------------------------------------
# Series Decomposition: trend = moving average; seasonal = x - trend.
# Vào/ra: (B, C, L).
# --------------------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=13):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size phải lẻ để đệm đối xứng."
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):                          # (B, C, L)
        pad = (self.kernel_size - 1) // 2
        xp = torch.cat([x[..., :1].repeat(1, 1, pad), x,
                        x[..., -1:].repeat(1, 1, pad)], dim=-1)
        trend = self.avg(xp)                       # (B, C, L)
        return trend, x - trend


# --------------------------------------------------------------------------
# Intra-patch: attention giữa S điểm TRONG một patch. Trộn kênh ở embed.
# Vào (B*, S, in_ch) -> ra (B*, S, d_model).
# --------------------------------------------------------------------------
class IntraPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, in_ch=1, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, patch_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=1, enable_nested_tensor=False)

    def forward(self, x):                          # (B*, S, in_ch)
        return self.sa(self.embed(x) + self.pos)   # (B*, S, d)


# --------------------------------------------------------------------------
# Inter-patch (kiểu PatchTST): mỗi patch -> 1 token d_model, attention giữa P
# token. Trộn kênh ở embed. Vào (B, P, S, in_ch) -> ra (B, P, d_model).
# --------------------------------------------------------------------------
class InterPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, num_patch, in_ch=1,
                 n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(patch_len * in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_patch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=n_layers,
                                        enable_nested_tensor=False)

    def forward(self, patches):                    # (B, P, S, in_ch)
        B_, P, S, ic = patches.shape
        tok = self.embed(patches.reshape(B_, P, S * ic))      # (B, P, d)
        return self.sa(tok + self.pos)             # (B, P, d)


# --------------------------------------------------------------------------
# Auto-Correlation (Autoformer) mức điểm. Vào (B, L, d) -> ra (B, L, d).
# Tìm top-k độ trễ (chu kỳ) bằng tự tương quan qua FFT, rồi gộp các đoạn
# cùng pha (time-delay aggregation). Bỏ độ trễ 0 (tự tương quan tầm thường).
# --------------------------------------------------------------------------
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads=8, top_k=3, factor=1, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            n_heads = 1
        self.h = n_heads
        self.dh = d_model // n_heads
        self.top_k = top_k
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                          # (B, L, d)
        B, L, D = x.shape
        h, dh = self.h, self.dh
        q = self.q(x).view(B, L, h, dh).permute(0, 2, 3, 1)   # (B, h, dh, L)
        k = self.k(x).view(B, L, h, dh).permute(0, 2, 3, 1)
        v = self.v(x).view(B, L, h, dh).permute(0, 2, 3, 1)

        # tự tương quan theo thời gian qua FFT (định lý Wiener-Khinchin)
        qf = torch.fft.rfft(q, dim=-1)
        kf = torch.fft.rfft(k, dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)   # (B, h, dh, L)

        # chọn top-k độ trễ từ tương quan trung bình (bỏ lag 0)
        k_top = self.top_k if self.top_k is not None else max(2, int(self.factor * math.log(L)))
        k_top = max(1, min(k_top, L - 1))
        mean_corr = corr.mean(dim=(0, 1, 2))       # (L,)
        mean_corr = mean_corr.clone()
        mean_corr[0] = float("-inf")
        top_vals, top_lags = torch.topk(mean_corr, k_top)         # (k,)
        w = torch.softmax(top_vals, dim=-1)        # (k,)

        # time-delay aggregation: cuộn V theo từng chu kỳ rồi cộng có trọng số
        agg = torch.zeros_like(v)
        for i in range(k_top):
            lag = int(top_lags[i].item())
            agg = agg + torch.roll(v, shifts=-lag, dims=-1) * w[i]

        agg = agg.permute(0, 3, 1, 2).reshape(B, L, D)            # (B, L, d)
        return self.out(self.drop(agg))


# --------------------------------------------------------------------------
# Gated fusion của hai nhánh seasonal (period, fast). Vào hai tensor (B,P,S,d)
# -> ra (B,P,S,d). Một cổng/vị trí, softmax trên 2 nhánh.
# --------------------------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, period, fast):               # (B, P, S, d)
        g = torch.softmax(self.gate(torch.cat([period, fast], dim=-1)), dim=-1)  # (B,P,S,2)
        z = g[..., 0:1] * period + g[..., 1:2] * fast
        return self.norm(z), g                     # trả thêm g để diễn giải


# --------------------------------------------------------------------------
# Mô hình chính.
# --------------------------------------------------------------------------
class PatchTST_Decomp(nn.Module):
    def __init__(self,
                 input_size, context_len, horizon,
                 d_model=256, patch_len=6, stride=6, target_idx=0,
                 n_heads=8, decomp_kernel=13, trend_layers=2,
                 ac_top_k=3, ac_factor=1, dropout=0.1,
                 use_trend=True, use_period=True, use_fast=True,
                 return_gate=False, **kwargs):
        super().__init__()
        assert stride == patch_len, (
            "patchTST_decomp yêu cầu stride == patch_len (patch không chồng lấn) "
            "để căn thời gian period/fast. Nhận stride=%d, patch_len=%d."
            % (stride, patch_len))
        if d_model % n_heads != 0:
            n_heads = 1

        self.input_size = input_size
        self.context_len = context_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx
        self.return_gate = return_gate
        self._last_gate = None

        self.pad = (stride - (context_len - patch_len) % stride) % stride
        self.num_patch = (context_len + self.pad - patch_len) // stride + 1
        P = self.num_patch
        C = input_size

        self.use_trend = use_trend
        self.use_period = use_period
        self.use_fast = use_fast
        assert use_trend or use_period or use_fast, \
            "Phải bật ít nhất một nhánh (trend/period/fast)."

        self.revin = RevIN()
        self.decomp = SeriesDecomp(decomp_kernel)

        # TREND -> inter
        self.inter = (InterPatchBlock(d_model, patch_len, P, in_ch=C,
                                      n_heads=n_heads, n_layers=trend_layers,
                                      dropout=dropout) if use_trend else None)
        # SEASONAL: period (Auto-Correlation)
        if use_period:
            self.period_embed = nn.Linear(C, d_model)          # embed riêng cho AC
            self.autocorr = AutoCorrelation(d_model, n_heads=n_heads,
                                            top_k=ac_top_k, factor=ac_factor,
                                            dropout=dropout)
        else:
            self.period_embed = None
            self.autocorr = None
        # SEASONAL: fast (intra)
        self.intra = (IntraPatchBlock(d_model, patch_len, in_ch=C,
                                      n_heads=n_heads, dropout=dropout)
                      if use_fast else None)

        # Hợp seasonal: gate khi có cả hai; LayerNorm khi chỉ còn một
        if use_period and use_fast:
            self.fusion = GatedFusion(d_model)
            self.seasonal_norm = None
        elif use_period or use_fast:
            self.fusion = None
            self.seasonal_norm = nn.LayerNorm(d_model)
        else:
            self.fusion = None
            self.seasonal_norm = None

        # Dual head — cộng ở không gian dự báo
        self.head_seasonal = (nn.Sequential(
            nn.Flatten(1), nn.Dropout(dropout),
            nn.Linear(P * patch_len * d_model, horizon))
            if (use_period or use_fast) else None)             # flatten (giữ điểm)
        self.head_trend = (nn.Sequential(
            nn.Flatten(1), nn.Dropout(dropout),
            nn.Linear(P * d_model, horizon))
            if use_trend else None)                            # nhẹ (không có trục S)

    def _patchify(self, x):                        # (B, C, L) -> (B, C, P, S)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x, future=None):             # x: (B, L, C)
        B, L, C = x.shape
        P, S = self.num_patch, self.patch_len
        xn = self.revin.normalize(x.transpose(1, 2))           # (B, C, L)
        trend, seasonal = self.decomp(xn)                      # (B, C, L) mỗi cái

        # --- TREND -> inter ---
        trend_rep = None
        if self.inter is not None:
            tr = self._patchify(trend).permute(0, 2, 3, 1)     # (B, P, S, C)
            trend_rep = self.inter(tr)                         # (B, P, d)

        # --- SEASONAL: period (Auto-Correlation point-level) ---
        period = None
        if self.autocorr is not None:
            se_pad = seasonal if self.pad == 0 else F.pad(
                seasonal, (0, self.pad), mode="replicate")     # (B, C, L_pad)
            se_pts = se_pad.transpose(1, 2)                    # (B, L_pad, C)
            period = self.autocorr(self.period_embed(se_pts))  # (B, L_pad, d)
            period = period.reshape(B, P, S, -1)               # (B, P, S, d)

        # --- SEASONAL: fast (intra patch-level) ---
        fast = None
        if self.intra is not None:
            se_pm = self._patchify(seasonal).permute(0, 2, 3, 1)   # (B, P, S, C)
            fast = self.intra(se_pm.reshape(B * P, S, C)).reshape(B, P, S, -1)

        # --- hợp seasonal: gate khi đủ hai, không thì dùng nhánh còn lại ---
        gate = None
        if period is not None and fast is not None:
            seasonal_rep, gate = self.fusion(period, fast)     # (B, P, S, d)
        elif period is not None:
            seasonal_rep = self.seasonal_norm(period)
        elif fast is not None:
            seasonal_rep = self.seasonal_norm(fast)
        else:
            seasonal_rep = None
        self._last_gate = gate

        # --- dual head: cộng ở không gian dự báo ---
        parts = []
        if seasonal_rep is not None:
            parts.append(self.head_seasonal(seasonal_rep.flatten(1)))
        if trend_rep is not None:
            parts.append(self.head_trend(trend_rep.flatten(1)))
        y = sum(parts)                                         # luôn còn >=1 nhánh

        y = self.revin.denormalize_target(y, self.target_idx)
        if self.return_gate:
            return y, gate
        return y


# ==========================================================================
# Chỉ số vùng đỉnh (đồng bộ với file cũ).
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
            "peak_bias": float(err.mean()),
            "n_peak": int(mask.sum()), "peak_threshold": float(thr)}


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    def cnt(m):
        return 0 if m is None else sum(p.numel() for p in m.parameters())
    B, L, C, H = 4, 168, 3, 18
    x = torch.randn(B, L, C)
    configs = [
        ("full             ", dict()),
        ("w/o period (AC)  ", dict(use_period=False)),
        ("w/o fast (intra) ", dict(use_fast=False)),
        ("w/o trend        ", dict(use_trend=False)),
        ("period only       ", dict(use_fast=False, use_trend=False)),
        ("fast only         ", dict(use_period=False, use_trend=False)),
    ]
    for name, kw in configs:
        m = PatchTST_Decomp(input_size=C, context_len=L, horizon=H,
                            d_model=256, patch_len=6, stride=6, target_idx=0,
                            decomp_kernel=13, trend_layers=2, ac_top_k=3, **kw)
        y = m(x); y.sum().backward()
        ng = sum(p.grad is None for p in m.parameters())
        print(f"{name} out={tuple(y.shape)} total={cnt(m):>9,} no_grad={ng}")
    try:
        PatchTST_Decomp(input_size=C, context_len=L, horizon=H,
                        use_trend=False, use_period=False, use_fast=False)
    except AssertionError:
        print("guard (tắt hết) OK")
    print("OK.")