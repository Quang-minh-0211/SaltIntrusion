# ============================================================
# patchTST_CT_LSTM.py — CT-PatchTST + JOINT attention + LSTM Head
#
# Kiến trúc lai (Hybrid Architecture) nhằm tối ưu hóa khối đặc trưng
# khổng lồ từ Joint Channel-Time Attention. Thay vì dùng một lớp 
# Linear (Flatten) làm lãng phí trật tự không gian - thời gian, 
# đầu ra của Transformer Encoder sẽ được đưa qua một mạng LSTM 
# để xâu chuỗi các phụ thuộc tuần tự trước khi đưa ra dự báo cuối.
# ============================================================

import torch
import torch.nn as nn

from patchTST_CT import CT_PatchTST, ChannelAttentionBlock

class JointAttentionBlock(nn.Module):
    """Pre-norm transformer block over the flattened (M*N) token grid."""
    def __init__(self, d_model: int, n_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, N, D = x.shape
        h = x.reshape(B, M * N, D)

        z = self.norm1(h)
        out, attn_w = self.attn(z, z, z, need_weights=True,
                                average_attn_weights=True)
        self.last_attn = attn_w.detach()                  
        h = h + self.drop(out)
        h = h + self.drop(self.ffn(self.norm2(h)))

        return h.reshape(B, M, N, D)

class CT_PatchTST_LSTM(CT_PatchTST):
    """
    CT-PatchTST tích hợp Joint Attention và LSTM Head.
    """
    def __init__(self, *args, n_joint_layers: int = 1,
                 joint_heads: int = 4, joint_dropout: float = 0.1,
                 lstm_hidden_size: int = 128, lstm_layers: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.embed.out_features
        d_ff = self.encoder.layers[0].linear1.out_features
        
        # 1. Loại bỏ per-patch channel block, dùng Joint Block
        self.channel_blocks = nn.ModuleList()             
        self.joint_blocks = nn.ModuleList([
            JointAttentionBlock(d_model, joint_heads, d_ff, joint_dropout)
            for _ in range(n_joint_layers)
        ])

        # 2. Thêm khối LSTM để xử lý tuần tự ma trận đặc trưng
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=joint_dropout if lstm_layers > 1 else 0.0
        )
        
        # 3. Định nghĩa lại Head dự báo
        # Thay vì: nn.Linear(n_patches * d_model, horizon)
        # Ta dùng đầu ra của LSTM: nn.Linear(n_patches * lstm_hidden_size, horizon)
        self.lstm_head = nn.Linear(self.n_patches * lstm_hidden_size, self.horizon)

    def forward(self, x: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        B, L, M = x.shape

        # (0) RevIN Normalization
        if self.revin is not None:                        
            x = self.revin.normalize(x)

        # (1 & 2 & 3) Tiền xử lý, Patchify và Embedding
        x = x.permute(0, 2, 1).reshape(B * M, L)          
        x = self._patchify(x)                             
        x = self.embed_dropout(self.embed(x) + self.pos)  

        # --- (4) JOINT Channel-Time Attention ---
        x = x.view(B, M, self.n_patches, -1)
        x = x + self.channel_emb.unsqueeze(0)             
        for blk in self.joint_blocks:
            x = blk(x)                                    
        x = x.reshape(B * M, self.n_patches, -1)
        
        # --- (5) Time Attention (PatchTST Encoder) ---
        x = self.encoder_norm(self.encoder(x))            # output: (B*M, N, d_model)

        # --- (6) Hybrid LSTM Processing ---
        # Đưa chuỗi N patches qua LSTM để nắm bắt động lực học
        lstm_out, (h_n, c_n) = self.lstm(x)               # lstm_out: (B*M, N, lstm_hidden_size)

        # --- (7) Dự báo qua LSTM Head ---
        # Flatten toàn bộ các bước thời gian của LSTM (N * lstm_hidden_size)
        lstm_out_flat = lstm_out.reshape(B * M, -1)
        
        y = self.lstm_head(lstm_out_flat).view(B, M, self.horizon)
        y = y[:, self.target_idx, :]
        
        # Xử lý biến tương lai (nếu có)
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
            
        # (8) RevIN Denormalization
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
            
        return y

if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("— Test 1: Khởi tạo khối lai CT-PatchTST-LSTM —")
    # Khởi tạo mô hình thử nghiệm với hidden_size của LSTM là 128
    model = CT_PatchTST_LSTM(input_size=12, context_len=96, horizon=24, lstm_hidden_size=128)
    
    # Batch size = 4, context = 96, channels = 12
    x_dummy = torch.randn(4, 96, 12) 
    y_pred = model(x_dummy)
    
    print(f"Kích thước đầu vào: {x_dummy.shape}")
    print(f"Kích thước đầu ra: {y_pred.shape} (Kỳ vọng: [4, 1, 24] hoặc [4, số target, 24])")
    print("Mạng truyền thẳng (Forward pass) chạy thành công!")