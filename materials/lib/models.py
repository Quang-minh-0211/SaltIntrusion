"""
models.py — 4 kiến trúc deep learning cho dự báo xâm nhập mặn
=============================================================
Mỗi hàm trả về một keras.Model đã compile sẵn.

Cấu hình input:
    input_shape  = (lookback, n_features)
    n_outputs    = số target (1 nếu dự báo 1 trạm; 3 nếu Kịch bản 3)

Tất cả mô hình đều output 1 vector (n_outputs,) tương ứng với giá trị dự báo
tại horizon đã chọn (single-step ở horizon = h).
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ---------------------------------------------------------------------------
# Chung
# ---------------------------------------------------------------------------

def _compile(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(patience: int = 15, save_path: str | None = None):
    """EarlyStopping + ReduceLROnPlateau + (tuỳ chọn) ModelCheckpoint."""
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5
        ),
    ]
    if save_path:
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_loss", save_best_only=True, save_weights_only=False
        ))
    return cbs


# ---------------------------------------------------------------------------
# 1. Vanilla RNN (SimpleRNN)
# ---------------------------------------------------------------------------

def build_rnn(input_shape, n_outputs: int = 1, units: int = 64,
              dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.SimpleRNN(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.SimpleRNN(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_outputs, name="output")(x)
    return _compile(models.Model(inp, out, name="RNN"), lr)


# ---------------------------------------------------------------------------
# 2. LSTM
# ---------------------------------------------------------------------------

def build_lstm(input_shape, n_outputs: int = 1, units: int = 64,
               dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.LSTM(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_outputs, name="output")(x)
    return _compile(models.Model(inp, out, name="LSTM"), lr)


# ---------------------------------------------------------------------------
# 3. GRU
# ---------------------------------------------------------------------------

def build_gru(input_shape, n_outputs: int = 1, units: int = 64,
              dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.GRU(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_outputs, name="output")(x)
    return _compile(models.Model(inp, out, name="GRU"), lr)


# ---------------------------------------------------------------------------
# 4. CNN-LSTM
# ---------------------------------------------------------------------------

def build_cnn_lstm(input_shape, n_outputs: int = 1, filters: int = 64,
                   kernel_size: int = 3, lstm_units: int = 64,
                   dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    """Conv1D trích đặc trưng cục bộ → LSTM học phụ thuộc thời gian."""
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                      padding="causal", activation="relu")(inp)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                      padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units // 2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_outputs, name="output")(x)
    return _compile(models.Model(inp, out, name="CNN_LSTM"), lr)


# ---------------------------------------------------------------------------
# Builder dispatcher
# ---------------------------------------------------------------------------

BUILDERS = {
    "RNN":      build_rnn,
    "LSTM":     build_lstm,
    "GRU":      build_gru,
    "CNN-LSTM": build_cnn_lstm,
    "CNN_LSTM": build_cnn_lstm,
}


def build_model(name: str, input_shape, n_outputs: int = 1, **kwargs):
    if name not in BUILDERS:
        raise ValueError(f"Mô hình '{name}' không hỗ trợ. Chọn: {list(BUILDERS)}")
    return BUILDERS[name](input_shape, n_outputs=n_outputs, **kwargs)
