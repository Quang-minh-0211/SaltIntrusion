import torch
import torch.nn as nn

hidden_size = 64
num_layers = 2
cnn_filters = 128
kernel_size = 3

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
            if num_layers>1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self,x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1,:])
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self,x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1,:])
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self,x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1,:])
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, cnn_filters=cnn_filters, kernel_size=kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            cnn_filters, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
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
    hidden_size: int = hidden_size,
    num_layers: int = num_layers,
    cnn_filters: int = cnn_filters,
    kernel_size: int = kernel_size,
) -> nn.Module:
    kwargs = dict(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        horizon     = horizon,
    )
    models = {
        "RNN":      RNNModel,
        "LSTM":     LSTMModel,
        "GRU":      GRUModel,
        "CNN_LSTM": CNNLSTMModel,
    }
    if model_name not in models:
        raise ValueError(f"Model name not available: '{model_name}'.")
    
    # Chỉ truyền cnn_filters và kernel_size cho CNN_LSTM
    if model_name == "CNN_LSTM":
        kwargs["cnn_filters"] = cnn_filters
        kwargs["kernel_size"] = kernel_size

    return models[model_name](**kwargs)