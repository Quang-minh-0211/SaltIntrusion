# ============================================================
# patchTST_Parallel.py — Parallel Two-Stream Hybrid Architecture
# (CT-PatchTST || RNN) -> Concat -> Fully Connected
#
# Kiến trúc luồng song song:
# - Nhánh 1 (CT-PatchTST): Bắt các chu kỳ dài hạn và tương quan chéo.
# - Nhánh 2 (LSTM/GRU): Bắt các biến động tuần tự ngắn hạn, nhân quả.
# - Lớp Gộp (Fusion): Ghép 2 vector đặc trưng và đi qua mạng FC 2 lớp.
# ============================================================

import torch
import torch.nn as nn

# Tùy thuộc vào việc bạn muốn lai bản gốc CT_PatchTST hay bản Joint,
# bạn có thể import tương ứng. Ở đây dùng CT_PatchTST làm base.
from patchTST_CT import CT_PatchTST

class CT_PatchTST_Parallel(CT_PatchTST):
    """
    Hai nhánh chạy song song:
    1. CT_PatchTST Encoder -> trích xuất features (B*M, N * d_model)
    2. LSTM/GRU -> trích xuất features (B*M, rnn_hidden_size)
    Ghép 2 vector lại -> MLP -> Output
    """
    def __init__(self, *args, rnn_type='LSTM', rnn_hidden_size=64, rnn_layers=1, **kwargs):
        # Khởi tạo nhánh PatchTST như bình thường
        super().__init__(*args, **kwargs)
        
        self.rnn_type = rnn_type.upper()
        
        # --- Kích thước đặc trưng từ nhánh PatchTST ---
        d_model = self.embed.out_features
        self.patch_feature_dim = self.n_patches * d_model
        
        # --- NHÁNH 2: RNN (LSTM/GRU) ---
        # Do PatchTST áp dụng Channel Independence (tách riêng từng kênh), 
        # ta cũng cho RNN chạy trên từng kênh độc lập để đồng bộ không gian mẫu.
        # Input của RNN sẽ có số chiều (B*M, L, 1)
        if self.rnn_type == "RNN":    
            RNNClass = nn.RNN
        elif self.rnn_type == 'LSTM':
            RNNClass = nn.LSTM
        elif self.rnn_type == "GRU":
            RNNClass = nn.GRU
        self.rnn = RNNClass(
            input_size=1, 
            hidden_size=rnn_hidden_size, 
            num_layers=rnn_layers, 
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0.0
        )
        self.rnn_feature_dim = rnn_hidden_size
        
        # --- LỚP FUSION (Fully Connected / MLP) ---
        self.combined_dim = self.patch_feature_dim + self.rnn_feature_dim
        
        # Thay thế Linear Head cũ bằng một MLP 2 lớp để trộn đặc trưng hiệu quả hơn
        # nn.Sequential(Linear -> GELU -> Dropout -> Linear)
        self.fusion_head = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.combined_dim // 2, self.horizon)
        )
        
        # Vô hiệu hóa head gốc của CT_PatchTST để tiết kiệm bộ nhớ (nếu có)
        self.head = nn.Identity()

    def forward(self, x: torch.Tensor, x_future: torch.Tensor = None) -> torch.Tensor:
        B, L, M = x.shape
        
        # (0) Chuẩn hóa RevIN (áp dụng chung cho cả dữ liệu thô đầu vào)
        if self.revin is not None:
            x = self.revin.normalize(x)
            
        # Tách channel (Channel Independence): (B, L, M) -> (B*M, L)
        x_ci = x.permute(0, 2, 1).reshape(B * M, L)
        
        # ============================================================
        # NHÁNH 1: CT-PatchTST ENCODER
        # ============================================================
        x_patch = self._patchify(x_ci)                             # (B*M, N, P)
        x_patch = self.embed_dropout(self.embed(x_patch) + self.pos) # (B*M, N, d_model)
        
        # Nếu đang kế thừa bản CT_PatchTST (có channel_blocks)
        if hasattr(self, 'channel_blocks') and len(self.channel_blocks) > 0:
            x_patch = x_patch.view(B, M, self.n_patches, -1)
            x_patch = x_patch + self.channel_emb.unsqueeze(0)
            for blk in self.channel_blocks:
                x_patch = blk(x_patch)
            x_patch = x_patch.reshape(B * M, self.n_patches, -1)
            
        # Nếu đang kế thừa bản Joint (có joint_blocks)
        elif hasattr(self, 'joint_blocks') and len(self.joint_blocks) > 0:
            x_patch = x_patch.view(B, M, self.n_patches, -1)
            x_patch = x_patch + self.channel_emb.unsqueeze(0)
            for blk in self.joint_blocks:
                x_patch = blk(x_patch)
            x_patch = x_patch.reshape(B * M, self.n_patches, -1)
            
        x_patch = self.encoder_norm(self.encoder(x_patch))         # (B*M, N, d_model)
        patch_features = x_patch.reshape(B * M, -1)                # (B*M, N * d_model)
        
        # ============================================================
        # NHÁNH 2: LSTM/GRU ENCODER
        # ============================================================
        # Input của RNN cần có định dạng (Batch, Sequence, Feature)
        x_rnn_in = x_ci.unsqueeze(-1)                              # (B*M, L, 1)
        rnn_out, _ = self.rnn(x_rnn_in)                            # rnn_out: (B*M, L, hidden_size)
        
        # Lấy trạng thái ẩn cuối cùng (last hidden state) của chuỗi thời gian
        rnn_features = rnn_out[:, -1, :]                           # (B*M, rnn_hidden_size)
        
        # ============================================================
        # FUSION & PREDICTION
        # ============================================================
        # Ghép đặc trưng (Concatenate)
        combined_features = torch.cat([patch_features, rnn_features], dim=-1) # (B*M, combined_dim)
        
        # Cho qua mạng MLP để dự báo
        y = self.fusion_head(combined_features)                    # (B*M, horizon)
        y = y.view(B, M, self.horizon)                             # (B, M, horizon)
        
        # Lấy channel mục tiêu
        y = y[:, self.target_idx, :]                               # (B, horizon)
        
        # Cộng thêm thông tin tương lai (nếu dùng)
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
            
        # Giải chuẩn hóa RevIN
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
            
        return y


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(42)

    print("— Test: Khởi tạo mô hình luồng song song Parallel_CT_PatchTST_RNN —")
    # Khởi tạo mô hình (giả sử kế thừa thành công)
    try:
        model = CT_PatchTST_Parallel(
            input_size=12, context_len=96, horizon=24, 
            rnn_type='LSTM', rnn_hidden_size=64, rnn_layers=1
        )
        x_dummy = torch.randn(4, 96, 12)
        y_pred = model(x_dummy)
        print(f"  Đầu vào x: {x_dummy.shape}")
        print(f"  Đầu ra y : {y_pred.shape} (Kỳ vọng: [4, 24])")
        print("✅ Khởi tạo và chạy truyền thẳng (forward pass) thành công!")
    except Exception as e:
        print("Lưu ý: Test nội bộ có thể báo lỗi nếu không import được file patchTST_CT (do môi trường test).")
        print(f"Error: {e}")
