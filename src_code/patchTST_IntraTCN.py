"""
patchTST_IntraTCN.py
====================================================================
Biến thể PatchTST cho bài toán xâm nhập mặn — hiện thực đúng thay đổi
Gap 1 mà ta đã bàn:

    Nhánh intra-patch của Pathformer (pooling một-query, nén patch về
    MỘT vector) được thay bằng:  TCN (dilated) -> self-attention S x S
    ĐẦY ĐỦ, giữ trọn S vector mỗi patch (không thắt cổ chai).

Mọi thứ còn lại giữ đúng tinh thần PatchTST một tỉ lệ để CÔ LẬP tác động
của Gap 1 (không có router / multi-scale / aggregator của Pathformer):

    RevIN  ->  chia patch  ->  [ intra (TCN + SA S x S) ] +
                               [ inter (kiểu PatchTST: 1 token / patch) ]
           ->  Flatten -> Linear  ->  RevIN đảo chuẩn hoá

Nhánh inter dùng KIỂU PatchTST (mỗi patch -> 1 token d_model) rồi
broadcast ra S vị trí để cộng với intra — nhẹ hơn kiểu Pathformer
~200 lần về tham số, và giữ thí nghiệm sạch (thứ DUY NHẤT khác một
PatchTST chuẩn là khối intra).

Interface khớp pipeline hiện tại:
    model = PatchTST_IntraTCN(input_size=12, context_len=168, horizon=24,
                              d_model=128, n_layers=3, patch_len=6,
                              stride=6, target_idx=0)
    out = model(Xb)          # Xb: (B, context_len, input_size) -> (B, horizon)

Xử lý đa biến theo kiểu channel-independent (mỗi kênh qua cùng backbone),
đầu ra là MỘT chuỗi mục tiêu (target_idx) — hợp với "12 kênh vào, 1 trạm ra".
====================================================================
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# RevIN (không affine) — chuẩn hoá theo từng cửa sổ, từng kênh.
# Giúp chống distribution shift (năm test 2025 lệch phân phối so với train).
# Đảo chuẩn hoá đầu ra bằng thống kê của kênh mục tiêu (target_idx).
# --------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x):                       # x: (B, C, L)
        self._mean = x.mean(dim=-1, keepdim=True)          # (B, C, 1)
        self._std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return (x - self._mean) / self._std

    def denormalize_target(self, y, target_idx):  # y: (B, H) -> (B, H)
        mean = self._mean[:, target_idx, :]                # (B, 1)
        std = self._std[:, target_idx, :]                  # (B, 1)
        return y * std + mean


# --------------------------------------------------------------------------
# TCN: chồng vài lớp tích chập giãn nở (dilated) có residual.
# Vào  (B*, 1, S)  ->  ra  (B*, d_model, S)  — GIỮ nguyên độ dài S.
# Vai trò: bắt "hình dạng cục bộ" trong patch (sườn dốc, spike/đỉnh),
# bất biến tịnh tiến, rẻ (tuyến tính theo S). Dilation cho tầm nhìn phủ
# gần hết patch chỉ với vài tầng.
# --------------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, d_model, n_levels=2, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(n_levels):
            dilation = 2 ** i
            pad = dilation * (kernel_size - 1) // 2       # 'same' padding (kernel lẻ)
            layers.append(nn.Conv1d(in_ch, d_model, kernel_size,
                                    padding=pad, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = d_model
        self.net = nn.ModuleList(layers)
        # chiếu residual 1 -> d_model cho lần cộng tắt đầu tiên
        self.res_proj = nn.Conv1d(1, d_model, 1)

    def forward(self, x):                          # x: (B*, 1, S)
        res = self.res_proj(x)                     # (B*, d_model, S)
        out = x
        for layer in self.net:
            out = layer(out)
            # cộng residual sau mỗi cụm (Conv, GELU, Dropout)
            if isinstance(layer, nn.Dropout):
                out = out + res
                res = out
        return out                                 # (B*, d_model, S)


# --------------------------------------------------------------------------
# Khối intra-patch (THAY ĐỔI Gap 1): TCN -> self-attention S x S đầy đủ.
# Vào  (B*, S)  (giá trị thô của 1 patch, 1 kênh)
# Ra   (B*, S, d_model)  — GIỮ trọn S vector, KHÔNG pooling về 1 vector.
# --------------------------------------------------------------------------
class IntraPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, n_heads=8, n_sa_layers=1,
                 tcn_levels=2, dropout=0.1):
        super().__init__()
        self.tcn = TCN(d_model, n_levels=tcn_levels, dropout=dropout)
        self.pos = nn.Parameter(torch.randn(1, patch_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=n_sa_layers, enable_nested_tensor=False)

    def forward(self, x):                          # x: (B*, S)
        h = self.tcn(x.unsqueeze(1))               # (B*, d_model, S)
        h = h.transpose(1, 2)                      # (B*, S, d_model)
        h = h + self.pos                           # vị trí trong patch
        h = self.sa(h)                             # (B*, S, d_model) — SA S x S
        return h


# --------------------------------------------------------------------------
# Khối inter-patch (KIỂU PatchTST): mỗi patch -> 1 token d_model,
# self-attention giữa P token-patch, rồi broadcast ra S vị trí để cộng.
# --------------------------------------------------------------------------
class InterPatchBlock(nn.Module):
    def __init__(self, d_model, patch_len, num_patch, n_heads=8, n_layers=3,
                 dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(patch_len, d_model)          # S -> d_model (1 token/patch)
        self.pos = nn.Parameter(torch.randn(1, num_patch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.sa = nn.TransformerEncoder(enc, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, patches):                    # patches: (B*, P, S)
        tok = self.embed(patches)                  # (B*, P, d_model)
        tok = tok + self.pos
        tok = self.sa(tok)                         # (B*, P, d_model) — SA giữa patch
        return tok                                 # broadcast ra S ở nơi gọi


# --------------------------------------------------------------------------
# Mô hình chính.
# --------------------------------------------------------------------------
class PatchTST_IntraTCN(nn.Module):
    def __init__(self,
                 input_size,               # số kênh C (vd 12)
                 context_len,              # lookback L theo BƯỚC (vd 168)
                 horizon,                  # số bước dự báo H (vd 24)
                 d_model=128,
                 n_layers=3,               # số lớp cho inter-patch attention
                 patch_len=6,
                 stride=6,
                 target_idx=0,             # kênh mục tiêu (Salinity_BenLuc = 0 ở SC3)
                 n_heads=8,
                 intra_sa_layers=1,        # số lớp SA trong intra (cục bộ, để nhỏ)
                 tcn_levels=2,
                 dropout=0.1,
                 use_future=False,         # model này KHÔNG dùng future covariates
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.context_len = context_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx
        self.use_future = use_future        # để train loop gọi model(Xb) đúng nhánh
        self.aux_loss = None                # để train loop không cộng aux

        # d_model phải chia hết cho n_heads
        if d_model % n_heads != 0:
            n_heads = max(1, d_model // (d_model // n_heads))
        # đệm cuối chuỗi để chia patch khít
        self.pad = (stride - (context_len - patch_len) % stride) % stride
        L_pad = context_len + self.pad
        self.num_patch = (L_pad - patch_len) // stride + 1

        self.revin = RevIN()
        self.intra = IntraPatchBlock(d_model, patch_len, n_heads=n_heads,
                                     n_sa_layers=intra_sa_layers,
                                     tcn_levels=tcn_levels, dropout=dropout)
        self.inter = InterPatchBlock(d_model, patch_len, self.num_patch,
                                     n_heads=n_heads, n_layers=n_layers,
                                     dropout=dropout)
        self.fuse_norm = nn.LayerNorm(d_model)

        # Đầu dự báo (chia sẻ giữa các kênh): flatten (P*S*d_model) -> H
        flat = self.num_patch * patch_len * d_model
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(flat, horizon),
        )
        # Hợp nhất kênh: (B, C, H) -> (B, H) bằng tổ hợp tuyến tính học được
        self.channel_mix = nn.Linear(input_size, 1)

    # ---- chia patch: (B, C, L) -> (B, C, P, patch_len) --------------------
    def _patchify(self, x):
        if self.pad > 0:
            x = nn.functional.pad(x, (0, self.pad), mode="replicate")
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x                                    # (B, C, P, patch_len)

    def forward(self, x, future=None):
        # x: (B, L, C) theo pipeline -> đổi về (B, C, L)
        B, L, C = x.shape
        x = x.transpose(1, 2)                       # (B, C, L)
        x = self.revin.normalize(x)                 # chuẩn hoá theo cửa sổ

        patches = self._patchify(x)                 # (B, C, P, S)
        P, S = self.num_patch, self.patch_len

        # gộp (B, C) để xử lý channel-independent
        pc = patches.reshape(B * C, P, S)           # (B*C, P, S)

        # --- nhánh inter (kiểu PatchTST) ---
        inter = self.inter(pc)                      # (B*C, P, d_model)
        inter = inter.unsqueeze(2).expand(-1, -1, S, -1)   # broadcast -> (B*C, P, S, d_model)

        # --- nhánh intra (Gap 1: TCN + SA S x S) ---
        pts = pc.reshape(B * C * P, S)              # (B*C*P, S)
        intra = self.intra(pts)                     # (B*C*P, S, d_model)
        intra = intra.reshape(B * C, P, S, -1)      # (B*C, P, S, d_model)

        # --- hợp nhất ---
        z = self.fuse_norm(intra + inter)           # (B*C, P, S, d_model)

        # --- đầu dự báo, chia sẻ theo kênh ---
        y = self.head(z)                            # (B*C, H)
        y = y.reshape(B, C, self.horizon)           # (B, C, H)

        # --- hợp nhất kênh -> 1 chuỗi mục tiêu ---
        y = self.channel_mix(y.transpose(1, 2)).squeeze(-1)   # (B, H)

        # --- đảo chuẩn hoá bằng thống kê kênh mục tiêu ---
        y = self.revin.denormalize_target(y, self.target_idx)  # (B, H)
        return y


# ==========================================================================
# Chỉ số riêng cho VÙNG ĐỈNH — bắt buộc dùng cạnh RMSE/MAE tổng.
# Vì đỉnh mặn hiếm, RMSE tổng bị các điểm phẳng chi phối; cải thiện ở đỉnh
# có thể không hiện lên trong RMSE tổng. Gọi trên preds/trues (đã inverse
# về thang vật lý) ở không gian phẳng.
#   pct: lấy các điểm có giá trị THỰC thuộc top (1-pct) cao nhất.
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
        "peak_bias": float(err.mean()),          # <0 nghĩa là dự báo HỤT đỉnh (san phẳng)
        "n_peak": int(mask.sum()),
        "peak_threshold": float(thr),
    }


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    B, L, C, H = 4, 168, 12, 24
    model = PatchTST_IntraTCN(input_size=C, context_len=L, horizon=H,
                              d_model=128, n_layers=3, patch_len=6, stride=6,
                              target_idx=0)
    x = torch.randn(B, L, C)
    y = model(x)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}   (mong đợi ({B}, {H}))")
    print(f"num_patch = {model.num_patch}, patch_len = {model.patch_len}")
    print(f"tham số  : {n_par:,}")
    assert y.shape == (B, H), "SAI SHAPE!"

    # thử patch_len khác để chắc chắn linh hoạt
    for pl in [4, 8, 12, 24]:
        m = PatchTST_IntraTCN(input_size=C, context_len=L, horizon=H,
                              d_model=64, n_layers=2, patch_len=pl, stride=pl)
        yy = m(x)
        print(f"patch_len={pl:2d} -> P={m.num_patch:2d}, out={tuple(yy.shape)}, "
              f"params={sum(p.numel() for p in m.parameters()):,}")
    print("OK.")
