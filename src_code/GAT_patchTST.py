# ============================================================
# GAT_PatchTST_fixed.py — GAT (không gian) + PatchTST (thời gian)
#
# SỬA LẠI từ bản GAT_patchTST.py. Bốn lỗi đã khắc phục:
#
#  (1) MẤT THÔNG TIN GỐC: bản cũ thay chuỗi độ mặn bằng 16 chiều
#      trừu tượng của GAT trước khi patching -> PatchTST không còn
#      thấy tín hiệu mặn thô. Bản mới: GAT chỉ sinh ra SỐ HIỆU CHỈNH
#      cộng vào chuỗi mặn gốc qua cổng alpha:
#          sal_new = sal + alpha * GAT_delta(sal, exo)
#      alpha init nhỏ (0.05, theo Idea 4 của supervisor) => khởi đầu
#      gần như trùng baseline, không thể sụp đổ.
#
#  (2) CÁC NÚT GIỐNG NHAU: bản cũ cho mỗi nút vector [sal_i, 9 biến
#      khí tượng DÙNG CHUNG] -> 9/10 chiều identical, attention không
#      phân biệt nổi trạm. Bản mới: node identity embedding học được
#      + biến ngoại sinh chỉ đóng vai NGỮ CẢNH (điều biến attention
#      theo pha triều), không lấn át danh tính nút.
#
#  (3) MẤT RevIN: bản cũ viết lại từ đầu nên bỏ RevIN. Bản mới kế
#      thừa PatchTST_BASE -> có đủ RevIN, padding patch, use_future,
#      aux_loss, head.
#
#  (4) n_patches tính không padding -> crash khi (L-P) % stride != 0.
#      Nay dùng _patchify của lớp cha.
#
# Vị trí GAT: cấp BƯỚC THỜI GIAN THÔ, TRƯỚC patching. Đây là khác
# biệt thật so với PATCHTST_ATTENTION (attention ở cấp token/patch):
# ở đây mặn lan truyền giữa các trạm ở độ phân giải 2 giờ.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from patchTST_Base import PatchTST_BASE


class GraphAttentionLayer(nn.Module):
    """GAT một lớp trên đồ thị đầy đủ n_nodes trạm (multi-head trung bình).

    Đầu vào : h (B*L, n_nodes, d_in)
    Đầu ra  : (B*L, n_nodes, d_out), attention (B*L, n_nodes, n_nodes)
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4,
                 dropout: float = 0.1, neg_slope: float = 0.2):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_out // n_heads
        assert d_out % n_heads == 0, "d_out phải chia hết cho n_heads"
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(neg_slope)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        BL, M, _ = h.shape
        Wh = self.W(h).view(BL, M, self.n_heads, self.d_head)      # (BL,M,H,dh)

        # Additive attention kiểu GAT, tách thành 2 hạng để khỏi tạo
        # tensor (M,M,2*d) tốn bộ nhớ như bản cũ
        s = (Wh * self.a_src).sum(-1)                              # (BL,M,H)
        d = (Wh * self.a_dst).sum(-1)                              # (BL,M,H)
        e = self.leaky(s.unsqueeze(2) + d.unsqueeze(1))            # (BL,M,M,H)

        att = torch.softmax(e, dim=2)                              # chuẩn hóa theo nguồn
        att = self.drop(att)
        out = torch.einsum('bijh,bjhd->bihd', att, Wh)             # (BL,M,H,dh)
        return out.reshape(BL, M, -1), att.mean(-1)                # (BL,M,d_out), (BL,M,M)


class GAT_PatchTST(PatchTST_BASE):
    """PatchTST + GAT không gian có cổng, chạy trước patching.

    Tham số thêm
    ------------
    n_nodes      : số trạm mặn (3 kênh đầu của feature_cols)
    gat_dim      : chiều ẩn của GAT (nhỏ: 16-32 là đủ với 3 nút)
    gat_heads    : số head GAT
    alpha_init   : cổng residual (0.05 => khởi đầu ~ baseline)
    use_exo      : dùng biến ngoại sinh làm ngữ cảnh điều biến attention
    """

    def __init__(self, *args, n_nodes: int = 3, gat_dim: int = 32,
                 gat_heads: int = 4, gat_dropout: float = 0.1,
                 alpha_init: float = 0.05, use_exo: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = n_nodes
        self.use_exo = use_exo
        n_exo = self.input_size - n_nodes

        # (2) Danh tính nút: mỗi trạm có embedding riêng -> GAT phân biệt được
        self.node_emb = nn.Parameter(torch.randn(n_nodes, gat_dim) * 0.02)
        self.sal_proj = nn.Linear(1, gat_dim)
        # Ngoại sinh (gió/nhiệt/mưa/pha triều) là NGỮ CẢNH dùng chung:
        # cộng vào mọi nút -> điều biến attention theo pha triều,
        # nhưng không xóa nhòa danh tính nút như bản cũ
        self.exo_proj = nn.Linear(n_exo, gat_dim) if (use_exo and n_exo > 0) else None

        self.gat = GraphAttentionLayer(gat_dim, gat_dim, gat_heads, gat_dropout)
        self.gat_norm = nn.LayerNorm(gat_dim)
        self.out_proj = nn.Linear(gat_dim, 1)          # về lại thang độ mặn
        # KHỞI TẠO MỀM (Idea 4), KHÔNG zero-init: nếu out_proj = 0 thì
        # delta = 0 -> gradient về GAT và alpha đều = 0 -> GAT nằm chết
        # vĩnh viễn (deadlock). Thay vào đó dùng trọng số nhỏ + alpha nhỏ:
        # nhiễu ban đầu không đáng kể nhưng gradient vẫn chảy từ epoch 1.
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # (1) Cổng residual alpha nhỏ: khởi đầu GẦN baseline (không thể
        # sụp đổ như bản cũ), nhưng nhánh GAT vẫn được huấn luyện.
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.last_spatial_attn = None

    # --------------------------------------------------------
    def _spatial_update(self, x):
        """x: (B, L, M) đã chuẩn hóa RevIN -> trả x với 3 kênh mặn đã cập nhật."""
        B, L, M = x.shape
        n = self.n_nodes
        sal, exo = x[:, :, :n], x[:, :, n:]

        h = self.sal_proj(sal.reshape(B * L, n, 1))          # (B*L, n, d)
        h = h + self.node_emb.unsqueeze(0)                    # danh tính nút
        if self.exo_proj is not None:
            h = h + self.exo_proj(exo).reshape(B * L, 1, -1)  # ngữ cảnh chung

        g, att = self.gat(self.gat_norm(h))
        self.last_spatial_attn = att.view(B, L, n, n).detach()

        delta = self.out_proj(g).view(B, L, n)                # (B, L, n)
        sal = sal + self.alpha * delta                        # residual có cổng
        return torch.cat([sal, exo], dim=-1)

    def forward(self, x, x_future=None):
        B, L, M = x.shape
        assert L == self.context_len, (
            f"Model build với context_len={self.context_len}, nhận L={L}.")

        if self.revin is not None:                    # (0) RevIN
            x = self.revin.normalize(x)

        x = self._spatial_update(x)                   # (0.5) GAT không gian

        x = x.permute(0, 2, 1).reshape(B * M, L)      # (2) tách kênh
        x = self._patchify(x)                         # (3) patch (có padding)
        x = self.embed_dropout(self.embed(x) + self.pos)   # (4)
        x = self.encoder_norm(self.encoder(x))        # (5) time encoder

        y = self.head(x).view(B, M, self.horizon)     # (6)
        y = y[:, self.target_idx, :]                  # (7)
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# TÍCH HỢP trong run_experiment:
#
#   from GAT_PatchTST_fixed import GAT_PatchTST
#   ...
#   elif model_name == "GAT_PATCHTST":
#       model = GAT_PatchTST(
#           input_size  = ds["train"][0].shape[2],
#           context_len = lookback,
#           horizon     = horizon,
#           d_model     = p["hidden_size"],
#           n_layers    = p["num_layers"],
#           patch_len   = p.get("patch_len", 6),
#           stride      = p.get("stride", 6),
#           target_idx  = 0,
#           n_nodes     = 3,
#           gat_dim     = p.get("gat_dim", 32),
#           alpha_init  = p.get("alpha_init", 0.05),
#       )
#
# Sau khi train, đọc:
#   model.alpha.item()                                  # mức dùng GAT
#   model.last_spatial_attn.mean(dim=(0,1))             # (3,3) trạm-nhìn-trạm
#   # attention theo pha triều (điểm hay của GAT cấp bước thời gian):
#   #   nhóm last_spatial_attn theo giờ trong ngày -> xem hướng lan
#   #   truyền có đảo chiều theo triều lên/xuống không
# ============================================================


if __name__ == "__main__":
    torch.manual_seed(0)

    print("— Test 1: forward/backward nhiều cấu hình —")
    for lb, hz in [(24, 12), (60, 60), (50, 24)]:   # 50 -> kiểm tra padding
        m = GAT_PatchTST(input_size=12, context_len=lb, horizon=hz)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        print(f"  lookback={lb} horizon={hz} -> out {tuple(y.shape)} | "
              f"n_patches={m.n_patches} | backward OK")

    print("— Test 2: tại init, đầu ra GẦN PatchTST gốc (nhiễu nhỏ, có gradient) —")
    from patchTST_Base import PatchTST_BASE
    torch.manual_seed(1); base = PatchTST_BASE(input_size=12, context_len=24, horizon=12)
    torch.manual_seed(1); ext = GAT_PatchTST(input_size=12, context_len=24, horizon=12)
    base.eval(); ext.eval()
    x = torch.randn(8, 24, 12)
    with torch.no_grad():
        yb, ye = base(x), ext(x)
        diff = (yb - ye).abs().max().item()
        scale = yb.abs().mean().item()
    print(f"  lệch tối đa: {diff:.4f} | thang dự báo: {scale:.4f} "
          f"| tỉ lệ: {diff/scale:.1%} (nhỏ = khởi đầu gần baseline)")
    assert diff < 0.5 * scale, "Nhiễu khởi tạo quá lớn!"

    print("— Test 3: GAT thực sự nhận gradient (không chết) —")
    m = GAT_PatchTST(input_size=12, context_len=24, horizon=12)
    y = m(torch.randn(4, 24, 12))
    F.mse_loss(y, torch.randn(4, 12)).backward()
    g_gat = m.gat.W.weight.grad.abs().sum().item()
    g_alpha = m.alpha.grad.abs().item()
    print(f"  grad GAT.W = {g_gat:.4f} | grad alpha = {g_alpha:.4f} (đều phải > 0)")
    assert g_gat > 0 and g_alpha > 0

    print("— Test 4: 3 nút PHẢI phân biệt được (lỗi 2 đã sửa) —")
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 24, 12))
    att = m.last_spatial_attn.mean(dim=(0, 1))
    off_diag_var = (att - att.mean()).abs().sum().item()
    print(f"  ma trận attention 3x3:\n{att.numpy().round(3)}")
    print(f"  độ lệch khỏi đồng nhất: {off_diag_var:.4f} (>0 = nút phân biệt được)")
    assert off_diag_var > 1e-3

    print("— Test 5: tương thích use_future —")
    m5 = GAT_PatchTST(input_size=12, context_len=24, horizon=12, use_future=True)
    y5 = m5(torch.randn(4, 24, 12), torch.randn(4, 12, 6))
    F.mse_loss(y5, torch.randn(4, 12)).backward()
    print(f"  out {tuple(y5.shape)} | OK")

    print("\nTất cả sanity test PASS.")