"""
utils.py — Module dùng chung cho dự án dự báo xâm nhập mặn ĐBSCL
================================================================
Cung cấp các tiện ích:
- Đọc & ghép dữ liệu các trạm (BenLuc, CauNoi, TanAn)
- Chia train/val/test theo năm (Train: 2020-2022, Val: 2023, Test: 2025)
- Tạo sliding-window sequences có lọc gap (sampling 2 giờ/lần)
- Chuẩn hóa (MinMax / Standard) tránh data leakage
- Metrics: RMSE, MAE, MAPE, R^2, NSE
- Vẽ kết quả & lưu CSV/PNG
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Cấu hình mặc định
# ---------------------------------------------------------------------------

DATA_ROOT_DEFAULT = "../data/processed_data"  # tương đối từ thư mục notebooks/

STATION_FILES = {
    "BenLuc":  "BenLuc_data/BenLuc_Weather_2020_2025.csv",
    "CauNoi":  "CauNoi_data/CauNoi_Weather_2020_2025.csv",
    "TanAn":   "TanAn_data/TanAn_Weather_2020_2025.csv",
}

SALINITY_COL = {
    "BenLuc": "Salinity_BenLuc",
    "CauNoi": "Salinity_CauNoi",
    "TanAn":  "Salinity_TanAn",
}

WEATHER_COLS = ["wind_speed", "temp", "total_precipitation"]

# Bước thời gian danh nghĩa = 2 giờ ⇒ 1 ngày = 12 step, 24h = 12 step, 48h = 24 step
STEPS_PER_HOUR = 0.5  # 1 step = 2 giờ
HORIZONS_HOURS = [12, 24, 48]   # giờ
HORIZON_STEPS = {12: 6, 24: 12, 48: 24}

DEFAULT_LOOKBACK_STEPS = 48   # 48 step × 2h = 96h ≈ 4 ngày quá khứ

TRAIN_YEARS = [2020, 2021, 2022]
VAL_YEARS   = [2023]
TEST_YEARS  = [2025]


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Sửa các typo phổ biến (".9.5", "4..2", "13,8") rồi ép sang float."""
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)         # "13,8" -> "13.8"
    s = s.str.replace(r"^\.(?=\d)", "", regex=True)   # ".9.5" -> "9.5"
    s = s.str.replace(r"\.\.+", ".", regex=True)      # "4..2" -> "4.2"
    return pd.to_numeric(s, errors="coerce")


def load_station(station: str, data_root: str = DATA_ROOT_DEFAULT) -> pd.DataFrame:
    """Đọc dữ liệu của 1 trạm, parse Time, sort, loại trùng & sửa lỗi định dạng số."""
    if station not in STATION_FILES:
        raise ValueError(f"Trạm '{station}' không hợp lệ. Chọn: {list(STATION_FILES)}")
    path = os.path.join(data_root, STATION_FILES[station])
    df = pd.read_csv(path)
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Time"]).drop_duplicates("Time").sort_values("Time")

    # Coerce numeric columns (xử lý typo trong CSV)
    sal_col = SALINITY_COL[station]
    for col in [sal_col] + WEATHER_COLS:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Bỏ các hàng có NaN ở target/feature (rất ít)
    df = df.dropna(subset=[sal_col] + [c for c in WEATHER_COLS if c in df.columns])
    df = df.reset_index(drop=True)
    return df


def load_all_stations(data_root: str = DATA_ROOT_DEFAULT) -> pd.DataFrame:
    """Đọc cả 3 trạm rồi merge theo Time (inner join trên Time chung).

    Trả về DataFrame có các cột:
        Time, Salinity_BenLuc, Salinity_CauNoi, Salinity_TanAn,
        wind_speed, temp, total_precipitation
    """
    frames = []
    for st in STATION_FILES:
        df = load_station(st, data_root)
        sal = SALINITY_COL[st]
        keep = ["Time", sal] + (WEATHER_COLS if st == "BenLuc" else [])
        frames.append(df[keep])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Time", how="inner")
    return out.sort_values("Time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chia train / val / test theo năm
# ---------------------------------------------------------------------------

def split_by_year(
    df: pd.DataFrame,
    train_years=TRAIN_YEARS,
    val_years=VAL_YEARS,
    test_years=TEST_YEARS,
):
    y = df["Time"].dt.year
    train = df[y.isin(train_years)].reset_index(drop=True)
    val   = df[y.isin(val_years)].reset_index(drop=True)
    test  = df[y.isin(test_years)].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# Tạo sequences (sliding window) có lọc gap
# ---------------------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    feature_cols,
    target_cols,
    lookback: int = DEFAULT_LOOKBACK_STEPS,
    horizon_steps: int = 12,
    max_gap_hours: float = 2.5,
):
    """Tạo (X, y) từ DataFrame đã sort theo Time.

    - lookback: số bước quá khứ làm input
    - horizon_steps: dự báo đúng 1 giá trị tại thời điểm t + horizon_steps
    - max_gap_hours: bước thời gian giữa 2 hàng phải <= max_gap_hours
      thì sequence mới được giữ (tránh nối qua gap mùa mưa).

    Trả về:
        X: shape (N, lookback, n_features)
        y: shape (N, n_targets)
        idx: chỉ số cuối cùng (target) trong df gốc - dùng để align thời gian
    """
    times = df["Time"].values
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targs = df[target_cols].to_numpy(dtype=np.float32)

    n = len(df)
    # Khoảng cách giữa hàng i và i-1 (giờ). NaT cho hàng đầu.
    dt_h = np.empty(n, dtype=np.float32)
    dt_h[0] = 0.0
    dt_h[1:] = (times[1:] - times[:-1]).astype("timedelta64[m]").astype(np.float32) / 60.0

    Xs, ys, idxs = [], [], []
    for i in range(lookback, n - horizon_steps):
        # Kiểm tra continuity: từ i-lookback+1 đến i+horizon_steps đều cách nhau <= max_gap_hours
        seg = dt_h[i - lookback + 1: i + horizon_steps + 1]
        if seg.max() > max_gap_hours:
            continue
        Xs.append(feats[i - lookback + 1: i + 1])  # gồm thời điểm i
        ys.append(targs[i + horizon_steps])
        idxs.append(i + horizon_steps)

    if not Xs:
        return (np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
                np.empty((0, len(target_cols)), dtype=np.float32),
                np.array([], dtype=np.int64))

    return np.stack(Xs), np.stack(ys), np.array(idxs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Scaler tránh data leakage (fit trên train, transform trên val/test)
# ---------------------------------------------------------------------------

def fit_scalers(train_df: pd.DataFrame, feature_cols, target_cols, kind: str = "minmax"):
    """Tạo 2 scaler: 1 cho features, 1 cho targets. Fit trên train_df."""
    cls = MinMaxScaler if kind == "minmax" else StandardScaler
    sx = cls(); sx.fit(train_df[feature_cols].to_numpy(dtype=np.float32))
    sy = cls(); sy.fit(train_df[target_cols].to_numpy(dtype=np.float32))
    return sx, sy


def apply_scalers(df: pd.DataFrame, sx, sy, feature_cols, target_cols) -> pd.DataFrame:
    """Trả về bản sao df với feature_cols & target_cols đã được scale."""
    out = df.copy()
    out[feature_cols] = sx.transform(df[feature_cols].to_numpy(dtype=np.float32))
    out[target_cols] = sy.transform(df[target_cols].to_numpy(dtype=np.float32))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency (1 = perfect, 0 = bằng dự báo trung bình)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def safe_mape(y_true, y_pred, eps: float = 1e-2) -> float:
    """MAPE bỏ qua các giá trị quá nhỏ (eps) để tránh chia 0."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred) -> dict:
    """Tính 5 metrics cho 1 cặp (y_true, y_pred). Hỗ trợ multi-output."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan")
    return {
        "RMSE": rmse,
        "MAE":  mae,
        "MAPE": safe_mape(y_true, y_pred),
        "R2":   r2,
        "NSE":  nse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_history(history, title: str = "Training history", save_path: str | None = None):
    """Vẽ loss train/val theo epoch."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=140)
    plt.show()


def plot_pred_vs_true(y_true, y_pred, time_index=None, title: str = "Predicted vs True",
                      save_path: str | None = None, max_points: int = 1000):
    """Vẽ chuỗi dự báo vs thực tế (cắt bớt nếu quá dài)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), max_points)
    plt.figure(figsize=(11, 4))
    x = time_index[:n] if time_index is not None else np.arange(n)
    plt.plot(x, y_true[:n], label="Thực tế", linewidth=1.5)
    plt.plot(x, y_pred[:n], label="Dự báo", linewidth=1.2, alpha=0.85)
    plt.title(title); plt.xlabel("Thời gian"); plt.ylabel("Độ mặn (g/L)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=140)
    plt.show()


def save_results_table(rows: list[dict], save_path: str) -> pd.DataFrame:
    """Lưu danh sách kết quả thành CSV để tổng hợp."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Tiện ích cao cấp - chuẩn bị toàn bộ dữ liệu cho 1 cấu hình
# ---------------------------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
    feature_cols,
    target_cols,
    lookback: int,
    horizon_steps: int,
    scaler_kind: str = "minmax",
):
    """One-shot: split year → fit scaler trên train → tạo sequences cho cả 3 tập.

    Trả về dict gồm X_train, y_train, X_val, y_val, X_test, y_test, sx, sy,
    test_time (timestamp tương ứng các y_test).
    """
    train_df, val_df, test_df = split_by_year(df)
    sx, sy = fit_scalers(train_df, feature_cols, target_cols, kind=scaler_kind)

    train_s = apply_scalers(train_df, sx, sy, feature_cols, target_cols)
    val_s   = apply_scalers(val_df,   sx, sy, feature_cols, target_cols)
    test_s  = apply_scalers(test_df,  sx, sy, feature_cols, target_cols)

    Xtr, ytr, _ = make_sequences(train_s, feature_cols, target_cols, lookback, horizon_steps)
    Xva, yva, _ = make_sequences(val_s,   feature_cols, target_cols, lookback, horizon_steps)
    Xte, yte, idx_te = make_sequences(test_s, feature_cols, target_cols, lookback, horizon_steps)
    test_time = test_df["Time"].iloc[idx_te].reset_index(drop=True) if len(idx_te) else pd.Series([], dtype="datetime64[ns]")

    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val":   Xva, "y_val":   yva,
        "X_test":  Xte, "y_test":  yte,
        "sx": sx, "sy": sy,
        "test_time": test_time,
        "n_features": Xtr.shape[2] if Xtr.size else len(feature_cols),
        "n_targets":  ytr.shape[1] if ytr.size else len(target_cols),
    }


def inverse_target(scaler, y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform giá trị target (1D hoặc 2D)."""
    arr = np.asarray(y_scaled, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr)
