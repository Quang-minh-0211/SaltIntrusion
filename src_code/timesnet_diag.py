"""
timesnet_diag.py — chẩn đoán việc chọn chu kỳ của TimesNet trên model ĐÃ TRAIN
================================================================================
Không train lại, không sửa model. Dùng với TimesNet.py (mọi cấu hình cờ).

1) period_diversity(model, X)   — mục 1 + mục 2
   Với mỗi TimesBlock, so chu kỳ CHUNG của batch (cái model thực sự dùng) với chu kỳ
   RIÊNG mà từng mẫu sẽ chọn nếu được tự chọn (cùng công thức FFT, không lấy trung
   bình trên batch).
     top1_outside : % mẫu có chu kỳ mạnh nhất của RIÊNG nó không nằm trong k chu kỳ chung
     jaccard      : độ trùng trung bình giữa tập chu kỳ riêng và tập chu kỳ chung (1 = trùng hết)
     dup_rate     : % batch có ít nhất hai nhánh gập GIỐNG HỆT nhau (cùng hình dạng gập)
     branches/batch : số nhánh trung bình (peak_mode="peak" có thể < top_k)
     period_freq  : chu kỳ THỰC (T/f*) xuất hiện nhiều nhất

2) batch_dependence(model, X, Y) — hệ quả của mục 1
   Dự báo cùng một tập với batch = 1 (mỗi mẫu tự chọn chu kỳ), 32 tuần tự, 32 xáo trộn, 256.
   Model, dữ liệu, trọng số giữ nguyên; chỉ cách ghép batch thay đổi.
     mean|Δŷ| so với batch = 1, và R²/MSE của từng cấu hình (nếu có Y)

Cách dùng (trong notebook, sau khi train xong model TimesNet):
    from timesnet_diag import period_diversity, batch_dependence
    Xte = torch.as_tensor(ds["test"][0], dtype=torch.float32)
    Yte = ds["test"][1]                         # (N, H) cùng thang với đầu ra model
    print(period_diversity(model, Xte))
    print(batch_dependence(model, Xte, Yte))
"""

from collections import Counter

import numpy as np
import torch


@torch.no_grad()
def period_diversity(model, X, batch_size=32, device=None):
    """Đọc đúng các nhánh model thực sự dùng (blk.last_periods), mọi peak_mode/fold_mode."""
    from TimesNet import select_periods
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    blocks = list(model.model)
    captured = {}
    hooks = [blk.register_forward_pre_hook(
        lambda mod, inp, i=i: captured.__setitem__(i, inp[0].detach()))
        for i, blk in enumerate(blocks)]

    stats = [dict(outside=0, n=0, jac=0.0, dup=0, nb=0, nbr=0, freq=Counter(), fold=Counter())
             for _ in blocks]
    try:
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size].to(device)
            model(xb)
            for i, blk in enumerate(blocks):
                periods, bins, folds, _ = blk.last_periods        # cái model thực sự dùng
                x = captured[i]                                    # (B, T, d)
                B, T, _ = x.shape
                if blk.peak_mode == "peak":
                    win = torch.hann_window(T, periodic=False, device=x.device, dtype=x.dtype)
                    x = x * win.view(1, T, 1)
                own_amp = torch.fft.rfft(x, dim=1).abs().mean(-1)  # (B, F) phổ riêng từng mẫu
                own_amp[:, 0] = 0
                S = set(int(b) for b in bins)
                st = stats[i]
                for b in range(B):
                    ob, _ = select_periods(own_amp[b], blk.k, T, blk.peak_mode,
                                           blk.min_rows, getattr(blk, "peak_rel", 0.0))
                    O = set(int(v) for v in ob)
                    st["outside"] += int(int(ob[0]) not in S)
                    st["jac"] += len(S & O) / len(S | O)
                st["n"] += B
                st["dup"] += int(len(set(folds)) < len(folds))   # hai nhánh gập giống hệt
                st["nb"] += 1
                st["nbr"] += len(folds)
                st["freq"].update(round(float(v), 2) for v in periods)
                st["fold"].update(folds)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    rows = []
    for i, st in enumerate(stats):
        rows.append({
            "block": i + 1,
            "top1_outside_%": round(100 * st["outside"] / st["n"], 1),
            "jaccard": round(st["jac"] / st["n"], 3),
            "dup_rate_%": round(100 * st["dup"] / st["nb"], 1),
            "branches/batch": round(st["nbr"] / st["nb"], 2),
            "period_freq": st["freq"].most_common(8),
        })
    return rows


@torch.no_grad()
def _predict(model, X, batch_size, order, device):
    out = np.empty((len(X), model.pred_len), dtype=np.float64)
    for s in range(0, len(X), batch_size):
        idx = order[s:s + batch_size]
        out[idx] = model(X[idx].to(device)).cpu().numpy()
    return out


def _r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


@torch.no_grad()
def batch_dependence(model, X, Y=None, batch_sizes=(1, 32, 256), shuffle_bs=32,
                     seed=0, device=None):
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    N = len(X)
    seq = np.arange(N)
    shuf = np.random.default_rng(seed).permutation(N)
    try:
        preds = {f"bs={b}": _predict(model, X, b, seq, device) for b in batch_sizes}
        preds[f"bs={shuffle_bs} xáo"] = _predict(model, X, shuffle_bs, shuf, device)
    finally:
        model.train(was_training)

    ref_key = f"bs={batch_sizes[0]}"
    ref = preds[ref_key]
    Yn = None if Y is None else np.asarray(Y, dtype=np.float64).reshape(ref.shape)
    rows = []
    for key, p in preds.items():
        row = {"config": key,
               f"mean|Δŷ| vs {ref_key}": float(np.abs(p - ref).mean()),
               f"max|Δŷ| vs {ref_key}": float(np.abs(p - ref).max())}
        if Yn is not None:
            row["MSE"] = float(((p - Yn) ** 2).mean())
            row["R2"] = float(_r2(Yn, p))
            row["mean|Δŷ| / std(y)"] = row[f"mean|Δŷ| vs {ref_key}"] / float(Yn.std())
        rows.append(row)
    return rows


# ==========================================================================
# Tự kiểm tra trên dữ liệu giả (model chưa train) — chỉ để chắc code chạy.
# ==========================================================================
if __name__ == "__main__":
    from TimesNet import TimesNet
    torch.manual_seed(0)
    t = torch.arange(400 + 168 + 6).float()
    base = torch.sin(2 * np.pi * t / 12.4) + 0.3 * torch.sin(2 * np.pi * t / 24)
    X = torch.stack([torch.stack([base[i:i + 168] * (1 + 0.1 * c) for c in range(3)], -1)
                     for i in range(400)]) + 0.1 * torch.randn(400, 168, 3)
    Y = torch.stack([base[i + 168:i + 174] for i in range(400)]).numpy()
    for pm, fm in (("bin", "floor"), ("peak", "leap")):
        m = TimesNet(3, 168, 6, target_idx=0, revin_mode="target_only", stats_token=True,
                     peak_mode=pm, fold_mode=fm, min_rows=3)
        print(pm, fm)
        for r in period_diversity(m, X):
            print("  ", r)
    for r in batch_dependence(m, X, Y):
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})