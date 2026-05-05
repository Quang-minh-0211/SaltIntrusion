"""
models.py — Định nghĩa 4 mô hình học sâu dùng chung cho tất cả kịch bản.
Import: from models import build_model
"""

import torch
import torch.nn as nn

# ─── Hyperparameters mặc định (có thể override khi gọi build_model) ───
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
CNN_FILTERS = 64
KERNEL_SIZE = 3


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon,
                 cnn_filters=CNN_FILTERS, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            cnn_filters, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)          # → (batch, features, seq)
        x = self.relu(self.conv(x))      # → (batch, filters, seq)
        x = x.permute(0, 2, 1)          # → (batch, seq, filters)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_model(
    model_name: str,
    input_size: int,
    horizon: int,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int  = NUM_LAYERS,
) -> nn.Module:
    """
    Tạo model theo tên.

    Parameters
    ----------
    model_name  : 'RNN' | 'LSTM' | 'GRU' | 'CNN-LSTM'
    input_size  : số features đầu vào
    horizon     : số bước dự báo
    hidden_size : số units ẩn
    num_layers  : số lớp RNN/LSTM/GRU

    Returns
    -------
    nn.Module chưa được đưa lên device
    """
    kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
    )
    models = {
        "RNN":      RNNModel,
        "LSTM":     LSTMModel,
        "GRU":      GRUModel,
        "CNN-LSTM": CNNLSTMModel,
    }
    if model_name not in models:
        raise ValueError(f"Tên model không hợp lệ: '{model_name}'. "
                         f"Chọn trong: {list(models.keys())}")
    return models[model_name](**kwargs)
