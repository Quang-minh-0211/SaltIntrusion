# ============================================================
# patchtst_station.py — PatchTST + tầng attention giữa các trạm đo mặn
#
# Cải tiến (bước 3 của lộ trình): PatchTST gốc xử lý các kênh hoàn toàn
# độc lập nên không biểu diễn được quan hệ thượng-hạ lưu giữa các trạm —
# vốn là vật lý thật của xâm nhập mặn (mặn ở Cầu Nổi hôm nay sẽ xuất
# hiện ở Bến Lức sau vài con nước). Module này thêm MỘT tầng attention
# nhỏ CHỈ giữa các kênh độ mặn (mặc định 3 kênh đầu), chèn giữa
# bước 5 (encoder) và bước 6 (head):
#
#   ... -> encoder -> (B*M, N, D)
#            -> reshape (B, M, N, D)
#            -> tách các kênh mặn: (B, 3, N, D)
#            -> với MỖI vị trí patch n: attention giữa 3 trạm
#               x_sal = x_sal + tanh(alpha) * MHA(x_sal)   # alpha init = 0
#            -> ghép lại -> head như cũ
#
# Cổng tanh(alpha) khởi tạo 0 => tại init mô hình TƯƠNG ĐƯƠNG PatchTST
# gốc (không thể khởi đầu tệ hơn baseline). Giá trị alpha học được là
# bằng chứng định lượng mô hình khai thác liên kết trạm đến đâu.
# ============================================================

import torch
import torch.nn as nn

from patchTST_Base import PatchTST_BASE


class StationAttention(nn.Module):
    """Attention giữa các trạm tại từng vị trí patch (time-aligned).

    Đầu vào  : (B, S, N, D)  — S = số kênh mặn (3 trạm)
    Đầu ra   : (B, S, N, D)  — đã trao đổi thông tin giữa trạm
    Attention map (B*N, S, S) được lưu ở self.last_attn để vẽ hình
    interpretability (trạm nào ảnh hưởng trạm nào).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))   # cổng zero-init (ReZero)
        self.station_emb = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, D = x.shape
        # Trải phẳng trạm x thời gian thành MỘT câu dài S*N token:
        # mỗi token mặn giờ nhìn được TOÀN BỘ token của mọi trạm, mọi con nước
        h = x.reshape(B, S * N, D)
        # Danh tính trạm chỉ cộng vào nhánh attention (không vào residual)
        # để giữ tính tương đương với PatchTST gốc khi alpha = 0
        h_in = self.norm((x + self.station_emb[:S]).reshape(B, S * N, D))
        out, attn_w = self.attn(h_in, h_in, h_in,
                                need_weights=True, average_attn_weights=True)
        self.last_attn = attn_w.detach()            # (B, S*N, S*N)
        h = h + torch.tanh(self.alpha) * out
        return h.reshape(B, S, N, D)


class PatchTST_ATTENTION(PatchTST_BASE):
    """PatchTST + StationAttention giữa bước 5 và bước 6.

    Tham số thêm
    ------------
    n_salinity   : số kênh đầu tiên là kênh độ mặn (SC3: 3, thứ tự
                   BenLuc, CauNoi, TanAn — khớp feature_cols của bạn)
    station_heads: số head của tầng attention trạm (nhỏ thôi, 4 là đủ)
    """

    def __init__(self, *args, n_salinity: int = 3, station_heads: int = 4,
                 station_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_salinity = n_salinity
        # d_model suy ra từ lớp embed của lớp cha
        d_model = self.embed.out_features
        self.station_attn = StationAttention(d_model, station_heads,
                                             station_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.context_len

        if self.revin is not None:                       # (0) RevIN
            x = self.revin.normalize(x)

        x = x.permute(0, 2, 1).reshape(B * M, L)         # (2) tách kênh
        x = self._patchify(x)                            # (3) cắt patch
        x = self.embed_dropout(self.embed(x) + self.pos) # (4) embedding
        x = self.encoder_norm(self.encoder(x))           # (5) encoder

        # ---- (5.5) TẦNG MỚI: attention giữa các trạm mặn ----
        x = x.view(B, M, self.n_patches, -1)             # (B, M, N, D)
        x_sal = self.station_attn(x[:, :self.n_salinity])
        x = torch.cat([x_sal, x[:, self.n_salinity:]], dim=1)
        x = x.reshape(B * M, self.n_patches, -1)
        # ------------------------------------------------------

        y = self.head(x).view(B, M, self.horizon)        # (6) head
        y = y[:, self.target_idx, :]                     # (7) target
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# TÍCH HỢP: trong run_experiment thêm một nhánh nữa
#
#   from patchtst_station import PatchTSTStation
#   ...
#   elif model_name == "PATCHTST_STATION":
#       model = PatchTSTStation(
#           input_size  = ds["train"][0].shape[2],
#           context_len = lookback,
#           horizon     = horizon,
#           d_model     = p["hidden_size"],
#           n_layers    = p["num_layers"],
#           patch_len   = p.get("patch_len", 6),
#           stride      = p.get("stride", 6),
#           target_idx  = 0,
#           n_salinity  = 3,
#       )
#
#   BEST_PARAMS["PATCHTST_STATION"] = dict(BEST_PARAMS["PATCHTST"])
#
# Sau khi train, đọc mức độ sử dụng liên kết trạm:
#   float(torch.tanh(model.station_attn.alpha))   # ~0 = không dùng
# và attention map trung bình (3x3) để vẽ hình cho bài báo:
#   model.station_attn.last_attn.mean(dim=0)
# ============================================================


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    # Test 1: forward/backward các cấu hình
    for lb, hz in [(24, 12), (84, 24)]:
        m = PatchTSTStation(input_size=12, context_len=lb, horizon=hz)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        print(f"lookback={lb} horizon={hz} -> out {tuple(y.shape)} | backward OK")

    # Test 2 (quan trọng): tại init (alpha=0), đầu ra PHẢI trùng khớp
    # với PatchTST gốc cùng trọng số => cải tiến khởi đầu từ đúng baseline
    torch.manual_seed(1)
    base = PatchTST(input_size=12, context_len=24, horizon=12)
    torch.manual_seed(1)
    ext = PatchTSTStation(input_size=12, context_len=24, horizon=12)
    base.eval(); ext.eval()
    x = torch.randn(8, 24, 12)
    with torch.no_grad():
        diff = (base(x) - ext(x)).abs().max().item()
    print(f"chênh lệch với PatchTST gốc tại init: {diff:.2e} (kỳ vọng ~0)")
    assert diff < 1e-5, "Cổng zero-init không hoạt động đúng!"
    print("\nTất cả sanity test PASS.")