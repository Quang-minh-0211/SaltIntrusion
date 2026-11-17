"""
TimesNet.py — bản dựng lại TimesNet (Wu et al., ICLR 2023) bám sát mã nguồn chính thức
=========================================================================================
Nguồn đối chiếu: thuml/Time-Series-Library
    models/TimesNet.py, layers/Embed.py, layers/Conv_Blocks.py

Bản này giữ nguyên từng phép tính của tác giả (task long_term_forecast), chỉ đổi interface
cho khớp pipeline 03_scenario3:
    model = TimesNet(input_size=C, context_len=168, horizon=H, target_idx=tgt_idx)
    out   = model(Xb)                      # Xb: (B, context_len, C) -> (B, H)

Khác biệt duy nhất so với thí nghiệm của tác giả: pipeline không có x_mark (đặc trưng giờ,
ngày, tháng), nên embedding = value + position, đúng nhánh `x_mark is None` trong
DataEmbedding gốc. Module temporal_embedding vì vậy không được tạo.

Luồng dữ liệu (B batch, L = context_len, H = horizon, C kênh, d = d_model):
    chuẩn hóa từng mẫu (mean, std theo thời gian)          (B, L, C)
    -> DataEmbedding: Conv1d k=3 circular + vị trí sin      (B, L, d)
    -> predict_linear: Linear(L -> L+H) dọc trục thời gian  (B, L+H, d)
    -> e_layers x [TimesBlock -> LayerNorm (dùng chung)]    (B, L+H, d)
    -> projection: Linear(d -> C)                            (B, L+H, C)
    -> khử chuẩn hóa, lấy H bước cuối, kênh target_idx      (B, H)

Siêu tham số mặc định = script ETTh1 chính thức: d_model=16, d_ff=32, e_layers=2,
top_k=5, num_kernels=6, dropout=0.1 (lr 1e-4, batch 32, 10 epoch, patience 3).

Cờ khôi phục mức nền (giống file PatchTST_IntraTCN):
  revin_mode  : "all" (mặc định, đúng bản gốc) — chuẩn hóa mọi kênh
                "target_only" — chỉ chuẩn hóa kênh mục tiêu, các kênh phụ giữ giá trị gốc
  stats_token : False (mặc định, đúng bản gốc)
                True — Linear(2C -> d) trên [μ, log σ] của mọi kênh, cộng vào MỌI vị trí
                (L+H) ngay sau predict_linear, trước các TimesBlock.
                Vì cộng một hằng số theo thời gian, stats token chỉ đổi thành phần f = 0
                của FFT — vốn bị TimesNet gán 0 — nên KHÔNG làm thay đổi việc chọn chu kỳ
                ở khối đầu tiên.
  Với revin_mode="all", stats_token=False, model cho ra đúng từng số như mã gốc.

Cờ chọn chu kỳ và gập (mục 2: lưới rời rạc + làm tròn):
  peak_mode : "bin"  (mặc định, đúng bản gốc) — top-k cột phổ cao nhất, kể cả cột cạnh nhau
              "peak" — chỉ nhận ĐỈNH CỤC BỘ (cao hơn hai cột bên cạnh), ước lượng tần số thực
                       f* = f + δ bằng parabol trên log chiều cao 3 cột quanh đỉnh, lấy tối đa
                       k đỉnh cao nhất; KHÔNG ép đủ k (ít đỉnh thì ít nhánh).
  fold_mode : "floor" (mặc định, đúng bản gốc) — gập theo p = floor(T/f), pad 0 phía sau
              "frac"  — nội suy tuyến tính để mỗi chu kỳ thực p dài đúng W = ceil(p) cột,
                        gập, conv, rồi nội suy ngược về T (cột thẳng hàng tuyệt đối)
              "leap"  — gập kiểu năm nhuận: hàng r bắt đầu tại round(r * p), hàng dài
                        floor(p) hoặc ceil(p), pad 0 cho đủ cột; mọi ô là dữ liệu thật,
                        độ lệch cột tối đa ±0,5 bước và không cộng dồn
              "phase" — như "frac" nhưng MỌI nhánh dùng cùng W = phase_width cột, nên
                        "xuống 1 hàng" = lùi 1 chu kỳ, "sang 1 cột" = 1/W chu kỳ ở mọi nhánh
  min_rows  : số hàng tối thiểu của ma trận gập (= f*). 1 (mặc định) = không lọc như bản
              gốc; 3 = bỏ các nhánh gập suy biến chỉ có 1–2 hàng. Áp dụng cho cả hai peak_mode.
  phase_width : W cho fold_mode="phase" (mặc định 16).
  peak_rel  : chỉ cho peak_mode="peak" — bỏ đỉnh thấp hơn peak_rel x đỉnh cao nhất (mặc
              định 0.05), để búp phụ của rò rỉ phổ và gợn nhiễu không chiếm nhánh.
  Chi tiết kỹ thuật của "peak": tìm đỉnh trên phổ đã nhân cửa sổ Hann (búp phụ nhỏ hơn
  ~10 lần, parabol-log sai ~0,004 bước thay vì ~0,04); trọng số nhánh vẫn lấy từ phổ KHÔNG
  cửa sổ như bản gốc. "frac"/"phase" dùng nội suy bậc ba Catmull-Rom.
  Với p nguyên, "frac" và "leap" cho ra đúng như "floor".

Ghi chu kỳ (không đổi phép tính):  model.record_periods = True
    -> model.period_log: list theo từng lần forward, mỗi phần tử là list theo từng khối
       {"periods": ndarray(k_eff,) chu kỳ thực, "bins": ndarray(k_eff,),
        "folds": list khóa hình dạng gập, "weights": Tensor(B, k_eff)} (biên độ thô)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# --------------------------------------------------------------------------
# Embedding — layers/Embed.py (PositionalEmbedding, TokenEmbedding, DataEmbedding)
# --------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
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
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3,
                                   padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in",
                                nonlinearity="leaky_relu")

    def forward(self, x):                           # (B, L, C) -> (B, L, d)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """Nhánh x_mark is None của DataEmbedding gốc: value + position, rồi dropout."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# --------------------------------------------------------------------------
# Inception_Block_V1 — layers/Conv_Blocks.py
# --------------------------------------------------------------------------
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels))
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


# --------------------------------------------------------------------------
# FFT_for_Period + TimesBlock — models/TimesNet.py
# --------------------------------------------------------------------------
def FFT_for_Period(x, k=2):
    """Bản gốc của tác giả (giữ để đối chiếu; TimesBlock dùng select_periods bên dưới)."""
    xf = torch.fft.rfft(x, dim=1)                   # (B, T//2+1, d)
    frequency_list = abs(xf).mean(0).mean(-1)       # trung bình trên batch và chiều ẩn
    frequency_list[0] = 0                           # bỏ thành phần f = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list                 # làm tròn xuống
    return period, abs(xf).mean(-1)[:, top_list]    # (k,), (B, k)


def select_periods(freq, k, T, peak_mode="bin", min_rows=1, peak_rel=0.0):
    """Chọn chu kỳ từ phổ biên độ 1 chiều `freq` (F,), freq[0] đã gán 0.

    peak_mode="peak": `freq` nên là phổ đã nhân cửa sổ Hann (TimesBlock tự làm), vì
    cửa sổ Hann làm búp phụ của rò rỉ phổ nhỏ đi nhiều và làm parabol-log gần chính xác.
    peak_rel: bỏ đỉnh thấp hơn peak_rel x đỉnh cao nhất (chỉ dùng cho "peak").

    Trả về (bins: ndarray int (k_eff,), fstar: ndarray float (k_eff,)).
    Chu kỳ thực = T / fstar; số hàng của ma trận gập ≈ fstar.
    """
    if peak_mode == "bin":
        if min_rows > 1:                            # lọc nhánh suy biến: cần f >= min_rows
            freq = freq.clone()
            freq[:min_rows] = 0
        _, top = torch.topk(freq, k)
        bins = top.detach().cpu().numpy()
        return bins, bins.astype(np.float64)

    a = freq.detach().double().cpu().numpy()
    F_ = len(a)
    la = np.log(np.maximum(a, 1e-12))
    cand = []
    for f in range(1, F_):
        right = a[f + 1] if f + 1 < F_ else -np.inf
        if not (a[f] > a[f - 1] and a[f] >= right):
            continue                                # chỉ nhận đỉnh cục bộ
        delta = 0.0
        if 1 < f < F_ - 1:                          # parabol trên log của 3 cột quanh đỉnh
            den = la[f - 1] - 2.0 * la[f] + la[f + 1]
            if den < 0:
                delta = float(np.clip(0.5 * (la[f - 1] - la[f + 1]) / den, -0.5, 0.5))
        fs = f + delta
        if fs >= min_rows:
            cand.append((a[f], f, fs))
    if cand and peak_rel > 0:                       # bỏ đỉnh không đáng kể (búp phụ, nhiễu)
        top = max(c[0] for c in cand)
        cand = [c for c in cand if c[0] >= peak_rel * top]
    if not cand:                                    # dự phòng: cột cao nhất (f >= 1)
        f = int(np.argmax(a[1:]) + 1)
        cand = [(a[f], f, float(f))]
    cand.sort(key=lambda c: -c[0])
    cand = cand[:k]                                 # KHÔNG ép đủ k
    return (np.array([c[1] for c in cand], dtype=np.int64),
            np.array([c[2] for c in cand], dtype=np.float64))


def _interp_time(x, pos):
    """Nội suy bậc ba Catmull-Rom x (B, L, N) tại các vị trí thực pos (M,) trên trục thời gian.
    Khứ hồi (lên lưới mới rồi về lại) sai khoảng 1,5% biên độ, so với khoảng 11% của
    nội suy tuyến tính ở chu kỳ ~7 bước. Khả vi theo x."""
    L = x.shape[1]
    i = torch.floor(pos).long()
    u = (pos - i.to(pos.dtype)).to(x.dtype).view(1, -1, 1)
    g = lambda o: x[:, (i + o).clamp(0, L - 1), :]
    p0, p1, p2, p3 = g(-1), g(0), g(1), g(2)
    return p1 + 0.5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3
                                          + u * (3 * (p1 - p2) + p3 - p0)))


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, top_k, num_kernels,
                 peak_mode="bin", fold_mode="floor", min_rows=1, phase_width=16,
                 peak_rel=0.05):
        super().__init__()
        self.seq_len, self.pred_len, self.k = seq_len, pred_len, top_k
        self.peak_mode, self.fold_mode = peak_mode, fold_mode
        self.min_rows, self.phase_width, self.peak_rel = min_rows, phase_width, peak_rel
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels))
        self.last_periods = None                    # để ghi log, không ảnh hưởng tính toán
        self._cache = {}

    # ---- các cách gập: x (B, T, N) -> conv 2D -> (B, T, N) ---------------------
    def _fold_floor(self, x, period):               # bản gốc
        B, total, N = x.shape
        if total % period != 0:                     # pad 0 phía sau cho chia hết
            length = (total // period + 1) * period
            padding = torch.zeros([B, length - total, N], device=x.device, dtype=x.dtype)
            out = torch.cat([x, padding], dim=1)
        else:
            length, out = total, x
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        out = self.conv(out)                        # (B, d, hàng, cột)
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        return out[:, :total, :]

    def _leap_index(self, T, p, device):
        key = ("leap", T, round(p, 6), str(device))
        if key not in self._cache:
            starts = []
            r = 0
            while True:
                s = int(math.floor(r * p + 0.5))
                if s >= T:
                    break
                starts.append(s)
                r += 1
            ends = starts[1:] + [T]
            W = max(e - s for s, e in zip(starts, ends))
            idx = torch.full((len(starts), W), T, dtype=torch.long)   # T = ô pad (vector 0)
            inv = torch.empty(T, dtype=torch.long)
            for r, (s, e) in enumerate(zip(starts, ends)):
                idx[r, :e - s] = torch.arange(s, e)
                inv[s:e] = r * W + torch.arange(e - s)
            self._cache[key] = (idx.to(device), inv.to(device), len(starts), W)
        return self._cache[key]

    def _fold_leap(self, x, p):
        B, T, N = x.shape
        idx, inv, R, W = self._leap_index(T, p, x.device)
        xp = torch.cat([x, x.new_zeros(B, 1, N)], dim=1)
        out = xp[:, idx.reshape(-1), :].reshape(B, R, W, N).permute(0, 3, 1, 2).contiguous()
        out = self.conv(out)
        out = out.permute(0, 2, 3, 1).reshape(B, R * W, N)
        return out[:, inv, :]

    def _fold_resample(self, x, p, W):
        B, T, N = x.shape
        key = ("rs", T, round(p, 6), W, str(x.device))
        if key not in self._cache:
            # lưới mới phủ TRỌN tới t = T-1 (vùng bản nháp tương lai), nên nội suy ngược
            # không bao giờ phải ngoại suy ở cuối chuỗi
            Tn = int(math.ceil((T - 1) * W / p - 1e-9)) + 1
            fwd = torch.arange(Tn, dtype=torch.float64) * (p / W)     # vị trí gốc của lưới mới
            back = torch.arange(T, dtype=torch.float64) * (W / p)     # vị trí lưới mới của từng t
            self._cache[key] = (fwd.to(x.device), back.to(x.device), Tn)
        fwd, back, Tn = self._cache[key]
        xr = _interp_time(x, fwd)                   # (B, Tn, N): mỗi chu kỳ = W điểm
        rows = -(-Tn // W)
        if rows * W > Tn:
            xr = torch.cat([xr, xr.new_zeros(B, rows * W - Tn, N)], dim=1)
        out = xr.reshape(B, rows, W, N).permute(0, 3, 1, 2).contiguous()
        out = self.conv(out)
        out = out.permute(0, 2, 3, 1).reshape(B, rows * W, N)[:, :Tn, :]
        return _interp_time(out, back)              # nội suy ngược về T

    def _fold(self, x, bin_, fstar):
        T = x.shape[1]
        if self.fold_mode == "floor":
            period = T // int(bin_) if self.peak_mode == "bin" else max(1, int(math.floor(T / fstar)))
            return self._fold_floor(x, period), ("floor", period)
        p = T / fstar
        if self.fold_mode == "leap":
            return self._fold_leap(x, p), ("leap", round(p, 4))
        W = int(math.ceil(p - 1e-9)) if self.fold_mode == "frac" else self.phase_width
        return self._fold_resample(x, p, W), (self.fold_mode, round(p, 4), W)

    def forward(self, x):
        B, T, N = x.size()
        xf = torch.fft.rfft(x, dim=1)
        amp = abs(xf)                               # (B, F, d)
        if self.peak_mode == "bin":
            freq = amp.mean(0).mean(-1)
            freq[0] = 0
        else:                                       # tìm đỉnh trên phổ đã nhân cửa sổ Hann
            win = torch.hann_window(T, periodic=False, device=x.device, dtype=x.dtype)
            freq = abs(torch.fft.rfft(x * win.view(1, T, 1), dim=1)).mean(0).mean(-1)
            freq[0] = 0
        bins, fstar = select_periods(freq, self.k, T, self.peak_mode, self.min_rows,
                                     self.peak_rel)
        # trọng số luôn lấy từ phổ KHÔNG cửa sổ (cùng thang với bản gốc), theo từng mẫu
        period_weight = amp.mean(-1)[:, bins]       # (B, k_eff)

        res, folds = [], []
        for b, fs in zip(bins, fstar):
            out, key = self._fold(x, b, fs)
            res.append(out)
            folds.append(key)
        self.last_periods = (T / fstar, bins, folds, period_weight.detach())

        res = torch.stack(res, dim=-1)              # (B, T, d, k_eff)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                              # residual


# --------------------------------------------------------------------------
# Model — models/TimesNet.py (forecast)
# --------------------------------------------------------------------------
class TimesNet(nn.Module):
    def __init__(self, input_size, context_len, horizon,
                 d_model=16, d_ff=32, n_layers=2, top_k=5, num_kernels=6,
                 dropout=0.1, target_idx=0, use_future=False,
                 revin_mode="all", stats_token=False,
                 peak_mode="bin", fold_mode="floor", min_rows=1, phase_width=16,
                 peak_rel=0.05, **kwargs):
        super().__init__()
        if kwargs:   # print thay vì warnings (notebook đang tắt warnings)
            print(f"[CẢNH BÁO] TimesNet bỏ qua tham số không dùng: {sorted(kwargs)}")
        if top_k > (context_len + horizon) // 2:
            raise ValueError("top_k vượt quá số tần số khả dụng")
        if revin_mode not in ("all", "target_only"):
            raise ValueError(f"revin_mode không hợp lệ: {revin_mode}")
        if peak_mode not in ("bin", "peak"):
            raise ValueError(f"peak_mode không hợp lệ: {peak_mode}")
        if fold_mode not in ("floor", "frac", "leap", "phase"):
            raise ValueError(f"fold_mode không hợp lệ: {fold_mode}")
        if min_rows < 1 or phase_width < 2:
            raise ValueError("min_rows phải >= 1 và phase_width phải >= 2")
        self.revin_mode = revin_mode
        self.stats_proj = nn.Linear(2 * input_size, d_model) if stats_token else None
        self.seq_len, self.pred_len = context_len, horizon
        self.target_idx, self.use_future, self.aux_loss = target_idx, use_future, None
        self.layer = n_layers

        self.model = nn.ModuleList(
            TimesBlock(context_len, horizon, d_model, d_ff, top_k, num_kernels,
                       peak_mode=peak_mode, fold_mode=fold_mode,
                       min_rows=min_rows, phase_width=phase_width, peak_rel=peak_rel)
            for _ in range(n_layers))
        self.enc_embedding = DataEmbedding(input_size, d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)     # một LayerNorm dùng chung mọi khối
        self.predict_linear = nn.Linear(context_len, horizon + context_len)
        self.projection = nn.Linear(d_model, input_size, bias=True)   # c_out = C

        self.record_periods = False
        self.period_log = []

    def forecast(self, x_enc):                      # (B, L, C) -> (B, L+H, C)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_enc.div(stdev)
        if self.revin_mode == "target_only":        # kênh phụ trở lại giá trị gốc
            mask = torch.zeros(1, 1, x_norm.shape[-1], device=x_norm.device, dtype=x_norm.dtype)
            mask[..., self.target_idx] = 1.0
            x_norm = x_norm * mask + (x_enc + means) * (1.0 - mask)

        enc_out = self.enc_embedding(x_norm)                                 # (B, L, d)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L+H, d)
        if self.stats_proj is not None:             # stats token: [μ ; log σ] -> d
            stats = torch.cat([means[:, 0, :], torch.log(stdev[:, 0, :])], dim=-1)  # (B, 2C)
            enc_out = enc_out + self.stats_proj(stats).unsqueeze(1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)                                   # (B, L+H, C)

        T = self.pred_len + self.seq_len
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, T, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, T, 1))
        return dec_out

    def forward(self, x, future=None):              # (B, L, C) -> (B, H)
        dec_out = self.forecast(x)
        if self.record_periods:
            self.period_log.append([{"periods": blk.last_periods[0].copy(),
                                     "bins": blk.last_periods[1].copy(),
                                     "folds": list(blk.last_periods[2]),
                                     "weights": blk.last_periods[3].cpu()}
                                    for blk in self.model])
        return dec_out[:, -self.pred_len:, self.target_idx]


# ==========================================================================
# Smoke test.
# ==========================================================================
if __name__ == "__main__":
    B, L, C = 4, 168, 3
    for H in (6, 24):
        for pm in ("bin", "peak"):
            for fm in ("floor", "frac", "leap", "phase"):
                m = TimesNet(input_size=C, context_len=L, horizon=H, target_idx=1,
                             revin_mode="target_only", stats_token=True,
                             peak_mode=pm, fold_mode=fm, min_rows=3)
                x = torch.randn(B, L, C)
                y = m(x)
                y.sum().backward()
                ng = [n for n, p in m.named_parameters() if p.grad is None]
                assert y.shape == (B, H) and not ng, (H, pm, fm, ng)
                print(f"H={H:>2} peak_mode={pm:<4} fold_mode={fm:<5} "
                      f"params={sum(p.numel() for p in m.parameters()):>9,}")
    print("OK.")