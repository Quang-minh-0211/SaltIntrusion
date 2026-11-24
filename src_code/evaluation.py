"""
evaluation.py — chỉ số vận hành cho dự báo xâm nhập mặn
=======================================================
Biến preds/trues (đã về ‰ thật) thành các chỉ số QUYẾT ĐỊNH, không chỉ R²/MSE.
Logic thuần hàm, test được, import vào notebook để vẽ/ xuất bảng.

Quy ước dữ liệu:
  pred, true : mảng (N, H) — N cửa sổ gốc, H bước dự báo (bước = hours_per_step giờ).
               pred[i, h] = độ mặn dự báo cho thời điểm (gốc_i + h + 1).
  Với nhóm lead-time cần MA TRẬN cuốn chiếu: các gốc LIÊN TIẾP (cửa sổ trượt bước 1),
  và một chuỗi thật liên tục true_series để định vị các lần vượt ngưỡng.

Ngưỡng mặc định 4‰ (lúa). hours_per_step mặc định 2 (lấy mẫu 2h).
"""

import numpy as np


# --------------------------------------------------------------------------
# Nhóm 1 — Phát hiện vượt ngưỡng (bắt buộc)
# --------------------------------------------------------------------------
def threshold_detection(pred, true, thr=4.0):
    """Precision/Recall/F1/CSI của việc báo 'nước không an toàn' (>= thr)."""
    p = np.asarray(pred).ravel() >= thr
    t = np.asarray(true).ravel() >= thr
    TP = int(np.sum(p & t)); FP = int(np.sum(p & ~t)); FN = int(np.sum(~p & t))
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")          # = POD
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    csi = TP / (TP + FP + FN) if (TP + FP + FN) else float("nan")
    far = FP / (TP + FP) if TP + FP else float("nan")          # tỉ lệ báo động nhầm
    return {"precision": prec, "recall": rec, "f1": f1, "csi": csi,
            "far": far, "n_event": int(np.sum(t)), "TP": TP, "FP": FP, "FN": FN}


# --------------------------------------------------------------------------
# Nhóm 2 — Sai số tại đỉnh (có dấu: âm = đánh giá thấp = nguy hiểm)
# --------------------------------------------------------------------------
def peak_error(pred, true, thr=4.0, under_tol=0.5):
    """Sai số CHỈ tại các điểm mà THẬT >= thr (giai đoạn nguy hiểm)."""
    p = np.asarray(pred).ravel(); t = np.asarray(true).ravel()
    m = t >= thr
    if not np.any(m):
        return {"peak_bias": float("nan"), "peak_rmse": float("nan"),
                "frac_under": float("nan"), "n_peak": 0}
    err = p[m] - t[m]
    return {"peak_bias": float(err.mean()),                    # <0: đánh giá thấp đỉnh
            "peak_rmse": float(np.sqrt((err ** 2).mean())),
            "frac_under": float(np.mean(err < -under_tol)),     # % đỉnh bị hụt > under_tol ‰
            "n_peak": int(m.sum())}


def _reconstruct_series(mat):
    """Từ ma trận cửa sổ (N,H) gốc liên tiếp -> chuỗi liên tục ts[m]=series[gốc0+1+m]."""
    mat = np.asarray(mat)
    return np.concatenate([mat[:, 0], mat[-1, 1:]])           # dài N+H-1


def daily_max_error(pred, true, steps_per_day=12):
    """Sai số ĐỘ MẶN LỚN NHẤT MỖI NGÀY. Dùng dự báo 1 bước (cột 0) khớp chuỗi thật."""
    p = np.asarray(pred)[:, 0]
    t = _reconstruct_series(true)[:len(p)]
    n = (len(t) // steps_per_day) * steps_per_day
    pmax = p[:n].reshape(-1, steps_per_day).max(1)
    tmax = t[:n].reshape(-1, steps_per_day).max(1)
    d = pmax - tmax
    return {"daily_max_bias": float(d.mean()), "daily_max_mae": float(np.abs(d).mean()),
            "n_day": len(tmax)}


# --------------------------------------------------------------------------
# Nhóm 3 — Thời gian báo trước (cần ma trận cuốn chiếu + chuỗi thật)
# --------------------------------------------------------------------------
def lead_time(pred, true, thr=4.0, hours_per_step=2, persistence=True):
    """Chỉ cần pred,true dạng (N,H) gốc liên tiếp — tự dựng chuỗi thật bên trong.
    pred[i,h] dự báo thời điểm ts[i+h], với ts = chuỗi thật dựng lại từ `true`.

    Trả về:
      warning_lead_* : mỗi lần vượt ngưỡng, lead LỚN NHẤT (giờ) mà dự báo còn báo đúng
                       (persistence: đúng ở lead đó VÀ mọi lead ngắn hơn).
      detection_by_lead : recall các đợt theo từng lead (1..H) — báo xa thì bắt kém đi.
      onset_timing_* : sai số (giờ) model dự báo bắt đầu vượt so với thật. Âm = báo sớm.
    """
    P = np.asarray(pred); N, H = P.shape
    ts = _reconstruct_series(true)
    events = np.where((ts[1:] >= thr) & (ts[:-1] < thr))[0] + 1   # chỉ số ts bắt đầu vượt
    leads_h, onset_h, detected = [], [], np.zeros(H)
    n_cov = np.zeros(H)
    for m in events:
        hit = np.zeros(H, dtype=bool)
        for l in range(1, H + 1):                             # lead l: gốc i=m-(l-1), cột l-1
            i = m - (l - 1)
            if 0 <= i < N:
                n_cov[l - 1] += 1
                hit[l - 1] = P[i, l - 1] >= thr
        detected += hit
        if persistence:
            good = 0
            for l in range(1, H + 1):
                if hit[l - 1]:
                    good = l
                else:
                    break
            maxlead = good
        else:
            idx = np.where(hit)[0]
            maxlead = (idx[-1] + 1) if len(idx) else 0
        if maxlead > 0:
            leads_h.append(maxlead * hours_per_step)
        i0 = m - (H - 1)                                       # gốc xa nhất còn phủ m
        if 0 <= i0 < N:
            fh = np.where(P[i0] >= thr)[0]
            if len(fh):
                onset_h.append((i0 + fh[0] - m) * hours_per_step)
    det_by_lead = np.divide(detected, np.maximum(n_cov, 1))
    return {"warning_lead_mean_h": float(np.mean(leads_h)) if leads_h else float("nan"),
            "warning_lead_median_h": float(np.median(leads_h)) if leads_h else float("nan"),
            "detection_by_lead": det_by_lead,                 # mảng (H,), lead 1..H
            "lead_hours_axis": np.arange(1, H + 1) * hours_per_step,
            "onset_timing_mean_h": float(np.mean(onset_h)) if onset_h else float("nan"),
            "n_event": int(len(events))}


# --------------------------------------------------------------------------
# Nhóm 4 — Cửa sổ nước an toàn (giá trị của độ phân giải 2h)
# --------------------------------------------------------------------------
def safe_window(pred, true, thr=4.0, window_hours=24, hours_per_step=2):
    """Số GIỜ an toàn (mặn < thr) trong window_hours tới: sai số và độ lệch có dấu.

    bias > 0: model nói CÓ NHIỀU giờ an toàn hơn thực tế -> hướng nguy hiểm.
    """
    p = np.asarray(pred); t = np.asarray(true)
    W = min(window_hours // hours_per_step, p.shape[1])
    safe_p = (p[:, :W] < thr).sum(1) * hours_per_step
    safe_t = (t[:, :W] < thr).sum(1) * hours_per_step
    d = safe_p - safe_t
    return {"safe_hours_mae": float(np.abs(d).mean()),
            "safe_hours_bias": float(d.mean()),               # >0 = tự tin thái quá
            "window_h": int(W * hours_per_step)}


# --------------------------------------------------------------------------
# Tiện ích: gom tất cả cho một (model, horizon)
# --------------------------------------------------------------------------
def evaluate_all(pred, true, thr=4.0, hours_per_step=2,
                 steps_per_day=12, safe_windows=(24, 48)):
    out = {"detection": threshold_detection(pred, true, thr),
           "peak": peak_error(pred, true, thr),
           "lead": lead_time(pred, true, thr, hours_per_step),
           "daily_max": daily_max_error(pred, true, steps_per_day)}
    for w in safe_windows:
        out[f"safe_{w}h"] = safe_window(pred, true, thr, w, hours_per_step)
    return out

def metrics_by_lead(pred, true, thr=4.0, hours_per_step=2):
    """Precision/Recall/F1/FAR theo TỪNG bước lead (từng cột của (N,H))."""
    pred = np.asarray(pred); true = np.asarray(true)
    H = pred.shape[1]
    hrs  = np.arange(1, H+1) * hours_per_step
    prec = np.full(H, np.nan); rec = np.full(H, np.nan)
    f1   = np.full(H, np.nan); far = np.full(H, np.nan)
    for j in range(H):
        p = pred[:, j] >= thr; t = true[:, j] >= thr
        tp = int((p &  t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
        if tp + fp:          prec[j] = tp/(tp+fp); far[j] = fp/(tp+fp)
        if tp + fn:          rec[j]  = tp/(tp+fn)
        if 2*tp + fp + fn:   f1[j]   = 2*tp/(2*tp+fp+fn)
    return {"lead_hours": hrs, "precision": prec, "recall": rec, "f1": f1, "far": far}
# ==========================================================================
# Smoke test trên dữ liệu giả có đỉnh đã biết.
# ==========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 2000
    t = np.arange(T)
    series = 2.5 + 2.0 * np.sin(2 * np.pi * t / 60) + 1.5 * np.sin(2 * np.pi * t / 360)
    series = np.clip(series + 0.2 * rng.standard_normal(T), 0, None)   # chuỗi thật
    H = 24
    # dự báo giả = thật + nhiễu + hụt đỉnh có chủ đích (mô phỏng MSE làm mượt)
    P = np.empty((T - H, H)); Y = np.empty((T - H, H))
    for i in range(T - H):
        seg = series[i + 1:i + 1 + H]
        Y[i] = seg
        pr = seg - 0.3 * np.maximum(seg - 4, 0) + 0.25 * rng.standard_normal(H)
        P[i] = np.clip(pr, 0, None)
    res = evaluate_all(P, Y, thr=4.0, hours_per_step=2)
    import pprint
    for k, v in res.items():
        if k == "lead":
            print("lead:", {kk: (np.round(vv, 3) if np.isscalar(vv) else "…")
                            for kk, vv in v.items()})
            print("  detection_by_lead[:6]:", np.round(v["detection_by_lead"][:6], 2))
        else:
            print(f"{k}:", {kk: round(vv, 3) if isinstance(vv, float) else vv
                            for kk, vv in v.items()})
    print("OK.")
