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
    
class parallelCNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, cnn_filters=32, kernel_size=3, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2,
        )
        self.relu = nn.ReLU()
        self.cnn_pooling = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size+cnn_filters, horizon)
    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]

        c = x.permute(0,2,1)
        c = self.relu(self.conv(c))
        c = self.cnn_pooling(c)
        cnn_feat = c.squeeze(-1)

        combined = torch.cat([lstm_feat, cnn_feat], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)
class TidePhysicsLoss(nn.Module):
    def __init__(self, lambda_tv=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_tv = lambda_tv
    def forward(self, y_pred, y_true):
        loss_mse = self.mse(y_pred,y_true)

        diff = torch.abs(y_pred[:,1:] - y_pred[:, :-1])

        loss_tv = torch.mean(torch.sum(diff, dim=1))

        return loss_mse + (self.lambda_tv * loss_tv)
# ============================================================
# UDF-LSTM (Utility-Driven Forgetting LSTM)
# Dán toàn bộ khối này vào models.py (trước hàm build_model)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class UDFLSTMCell(nn.Module):
    """Mot lop UDF-LSTM: cong quen nhan them tin hieu loi ich u_bar,
    cell state duoc nhan them cong reset rho (tinh o cap model)."""

    def __init__(self, input_size, hidden_size, beta=0.85):
        super().__init__()
        self.hidden_size = hidden_size
        self.beta = beta                                   # EMA cua loi ich
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.gamma_raw = nn.Parameter(torch.zeros(hidden_size))  # khuech dai tin hieu loi ich
        self.w_p = nn.Parameter(torch.randn(hidden_size) * 0.01)  # dau doc phu
        self.b_p = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[hidden_size:2 * hidden_size] = 1.0   # forget bias = 1

    def forward(self, x, h, c, u_bar, rho):
        H = self.hidden_size
        eps = 1e-8

        # (1) Loi ich tung chieu bo nho: dong gop vao dau doc phu
        contrib = (self.w_p * c).abs()
        u = contrib / (contrib.sum(dim=-1, keepdim=True) + eps)
        u_bar = self.beta * u_bar + (1.0 - self.beta) * u

        z = self.gates(torch.cat([x, h], dim=-1))
        i, f_pre, g, o = z.chunk(4, dim=-1)

        # (2) Cong quen moi: chieu co loi ich tren trung binh -> nho,
        #     duoi trung binh -> quen
        util = F.softplus(self.gamma_raw) * (u_bar - 1.0 / H) * H
        f = torch.sigmoid(f_pre + util)

        c = rho * f * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)

        # Dau doc phu: du bao do man scaled o BUOC KE TIEP tu cell state
        y_aux = (c * self.w_p).sum(dim=-1, keepdim=True) + self.b_p
        return h, c, u_bar, y_aux


class UDFLSTMModel(nn.Module):
    """Giao dien giong het LSTMModel: forward(x) -> (B, horizon).
    Loss phu duoc tinh trong forward va gan vao self.aux_loss
    (train_model cong them; evaluate bo qua -> khong can sua evaluate).

    target_idx: vi tri cot Salinity cua tram target trong feature_cols.
                Voi SC3 (input = BenLuc+CauNoi+TanAn, target = BenLuc)
                thi Salinity_BenLuc la cot 0.
    """

    def __init__(self, input_size, hidden_size, num_layers, horizon,
                 target_idx=0, beta=0.85, alpha=0.3, dropout=0.2):
        super().__init__()
        self.target_idx = target_idx
        self.alpha = alpha
        self.cells = nn.ModuleList([
            UDFLSTMCell(input_size if l == 0 else hidden_size,
                        hidden_size, beta=beta)
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else nn.Identity()
        self.log_kappa = nn.Parameter(torch.zeros(1))   # do nhay reset (hoc duoc)
        self.fc = nn.Linear(hidden_size, horizon)
        self.aux_loss = None
        # Luu de ve heatmap dien giai (khong anh huong training)
        self.last_f_seq = None
        self.last_rho_seq = None

    def forward(self, x):
        B, T, _ = x.shape
        L = len(self.cells)
        H = self.cells[0].hidden_size
        dev = x.device

        # Chuoi man quan sat (scaled) cua tram target ngay trong X
        y_seq = x[:, :, self.target_idx]                       # (B, T)

        h = [x.new_zeros(B, H) for _ in range(L)]
        c = [x.new_zeros(B, H) for _ in range(L)]
        u_bar = [x.new_full((B, H), 1.0 / H) for _ in range(L)]

        y_aux_prev = None
        aux_preds, rho_list, f_dummy = [], [], None

        for t in range(T):
            # (3) Cong reset: so du bao phu (lam o buoc t-1 cho buoc t)
            #     voi quan sat thuc te tai t -> phat hien doi che do
            if y_aux_prev is None:
                rho = x.new_ones(B, 1)
            else:
                kappa = torch.exp(self.log_kappa)
                err = (y_aux_prev.squeeze(-1) - y_seq[:, t]).abs()
                rho = torch.exp(-kappa * err).unsqueeze(-1)
            rho_list.append(rho)

            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l], u_bar[l], y_aux = cell(inp, h[l], c[l], u_bar[l], rho)
                inp = self.dropout(h[l]) if l < L - 1 else h[l]

            aux_preds.append(y_aux)     # (B,1): du bao man tai t+1
            y_aux_prev = y_aux

        y_hat = self.fc(h[-1])                                 # (B, horizon)

        # Loss phu: aux_preds[t] du bao y_seq[t+1]  (t = 0..T-2)
        if T > 1:
            aux_stack = torch.cat(aux_preds[:-1], dim=-1)      # (B, T-1)
            self.aux_loss = self.alpha * F.mse_loss(aux_stack, y_seq[:, 1:])
        else:
            self.aux_loss = None

        self.last_rho_seq = torch.cat(rho_list, dim=-1).detach()  # (B, T)
        return y_hat                 
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
        "PARALLEL_CNN_LSTM": parallelCNNLSTMModel,
        "UDF_LSTM": UDFLSTMModel,
    }
    if model_name not in models:
        raise ValueError(f"Model name not available: '{model_name}'.")
    
    # Chỉ truyền cnn_filters và kernel_size cho CNN_LSTM
    if model_name in ("CNN_LSTM","PARALLEL_CNN_LSTM","PARALLEL_CNN_LSTM_ATTEN"):
        kwargs["cnn_filters"] = cnn_filters
        kwargs["kernel_size"] = kernel_size

    return models[model_name](**kwargs)