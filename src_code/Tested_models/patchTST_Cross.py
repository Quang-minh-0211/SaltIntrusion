# ============================================================
# patchTST_Cross.py — PatchTST + Target-Query Cross-Attention có mặt nạ nhân quả
#
# Hiện thực hóa Idea 1 + Idea 2 (supervisor) + Idea 4 (alpha=0.05):
#
#   Idea 1 — Target-Query Cross-Attention:
#     Query  = token của TRẠM ĐÍCH (Bến Lức)
#     Key/Value = token của các trạm còn lại (Cầu Nổi — phía biển,
#                 Tân An — nhánh song song)
#     -> Trạm đích "hỏi" các trạm kia về khối mặn đang di chuyển tới;
#        không tốn tham số/đường attention cho chiều ngược lại vô nghĩa.
#
#   Idea 2 — Causal mask theo vị trí patch:
#     Token đích ở con nước i chỉ được nhìn token nguồn ở con nước j <= i
#     -> mã hóa ràng buộc vật lý: nước tới trạm đích lúc i phải rời trạm
#        nguồn TRƯỚC đó; chặn các liên kết phi vật lý (nhìn "tương lai
#        nội cửa sổ" của trạm nguồn).
#
#   Idea 4 — alpha khởi tạo 0.05 (không tanh): nhánh cross-attention nhận
#     gradient ngay từ epoch 1, tránh deadlock của zero-init.
#
# Vị trí chèn: giữa bước 5 (encoder) và bước 6 (head) — chỉ kênh đích
# được cập nhật, các kênh khác đi thẳng.
# ============================================================

import torch
import torch.nn as nn

from patchTST_Base import PatchTST_BASE


class TargetQueryCrossAttention(nn.Module):
    """Cross-attention: trạm đích (Query) hỏi các trạm nguồn (Key/Value),
    với mặt nạ nhân quả theo vị trí patch.

    Đầu vào : q  (B, N, D)            — token trạm đích
              kv (B, S_src, N, D)     — token các trạm nguồn
    Đầu ra  : (B, N, D)               — token trạm đích đã cập nhật

    self.last_attn: (B, N, S_src*N) — attention weight để phân tích độ trễ
    (cột được xếp theo [nguồn_0 patch 0..N-1 | nguồn_1 patch 0..N-1 | ...]).
    """

    def __init__(self, d_model: int, n_patches: int, n_sources: int,
                 n_heads: int = 4, dropout: float = 0.1,
                 alpha_init: float = 0.05, causal: bool = True):
        super().__init__()
        self.n_patches = n_patches
        self.n_sources = n_sources
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))   # Idea 4
        # Embedding danh tính trạm nguồn (để phân biệt Cầu Nổi vs Tân An)
        self.src_emb = nn.Parameter(torch.randn(n_sources, 1, d_model) * 0.02)
        self.last_attn = None

        # ----- Idea 2: mặt nạ nhân quả (True = CẤM nhìn) -----
        # mask[i, s*N + j] = True nếu j > i (nguồn ở "tương lai nội cửa sổ")
        if causal:
            i = torch.arange(n_patches).unsqueeze(1)            # (N, 1)
            j = torch.arange(n_patches).unsqueeze(0)            # (1, N)
            block = (j > i)                                     # (N, N)
            mask = block.repeat(1, n_sources)                   # (N, S_src*N)
        else:
            mask = torch.zeros(n_patches, n_sources * n_patches,
                               dtype=torch.bool)
        self.register_buffer("causal_mask", mask)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, S_src, N, D = kv.shape
        assert N == self.n_patches and S_src == self.n_sources

        kv = kv + self.src_emb[:S_src].unsqueeze(0)             # danh tính nguồn
        kv = kv.reshape(B, S_src * N, D)

        out, attn_w = self.attn(self.q_norm(q), self.kv_norm(kv),
                                self.kv_norm(kv),
                                attn_mask=self.causal_mask,
                                need_weights=True,
                                average_attn_weights=True)
        self.last_attn = attn_w.detach()                        # (B, N, S_src*N)
        return q + self.alpha * out                             # residual có cổng


class PatchTST_CROSS(PatchTST_BASE):
    """PatchTST + TargetQueryCrossAttention giữa bước 5 và bước 6.

    Tham số thêm
    ------------
    n_salinity  : số kênh đầu tiên là kênh độ mặn (SC3: 3 — thứ tự
                  BenLuc, CauNoi, TanAn, khớp feature_cols)
    cross_heads : số head của cross-attention (4 là đủ)
    causal      : bật/tắt mặt nạ nhân quả (tắt để ablate Idea 2)
    alpha_init  : giá trị khởi tạo cổng (0.05 theo Idea 4;
                  đặt 0.0 để ablate lại chế độ zero-init)

    Lưu ý: kênh ĐÍCH trong nhóm mặn = target_idx của lớp cha (mặc định 0
    = Salinity_BenLuc); các kênh mặn còn lại tự động thành trạm nguồn.
    """

    def __init__(self, *args, n_salinity: int = 3, cross_heads: int = 4,
                 cross_dropout: float = 0.1, causal: bool = True,
                 alpha_init: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0 <= self.target_idx < n_salinity):
            raise ValueError("target_idx phải nằm trong nhóm kênh mặn "
                             f"[0, {n_salinity}) — hiện là {self.target_idx}.")
        self.n_salinity = n_salinity
        d_model = self.embed.out_features
        self.cross_attn = TargetQueryCrossAttention(
            d_model=d_model, n_patches=self.n_patches,
            n_sources=n_salinity - 1, n_heads=cross_heads,
            dropout=cross_dropout, alpha_init=alpha_init, causal=causal,
        )

    def forward(self, x: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.context_len

        if self.revin is not None:                        # (0) RevIN
            x = self.revin.normalize(x)

        x = x.permute(0, 2, 1).reshape(B * M, L)          # (2) tách kênh
        x = self._patchify(x)                             # (3) cắt patch
        x = self.embed_dropout(self.embed(x) + self.pos)  # (4) embedding
        x = self.encoder_norm(self.encoder(x))            # (5) encoder

        # ---- (5.5) Target-Query Cross-Attention (Idea 1 + 2) ----
        x = x.view(B, M, self.n_patches, -1)              # (B, M, N, D)
        t = self.target_idx
        src_ids = [s for s in range(self.n_salinity) if s != t]
        q_new = self.cross_attn(x[:, t],                  # (B, N, D)
                                x[:, src_ids])            # (B, S-1, N, D)
        # Chỉ thay kênh đích, mọi kênh khác giữ nguyên (tránh in-place
        # trên tensor cần gradient -> dùng cat thay vì gán chỉ số)
        x = torch.cat([x[:, :t], q_new.unsqueeze(1), x[:, t + 1:]], dim=1)
        x = x.reshape(B * M, self.n_patches, -1)
        # ----------------------------------------------------------

        y = self.head(x).view(B, M, self.horizon)         # (6) head
        y = y[:, self.target_idx, :]                      # (7) target
        if self.use_future and x_future is not None:      # (6.5) nếu bật
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# TÍCH HỢP: trong run_experiment thêm nhánh
#
#   from patchTST_Cross import PatchTST_CROSS
#   ...
#   elif model_name == "PATCHTST_CROSS":
#       model = PatchTST_CROSS(
#           input_size  = ds["train"][0].shape[2],
#           context_len = lookback,
#           horizon     = horizon,
#           d_model     = p["hidden_size"],
#           n_layers    = p["num_layers"],
#           patch_len   = p.get("patch_len", 6),
#           stride      = p.get("stride", 6),
#           target_idx  = 0,
#           n_salinity  = 3,
#           causal      = p.get("causal", True),
#           alpha_init  = p.get("alpha_init", 0.05),
#       )
#
#   BEST_PARAMS["PATCHTST_CROSS"] = dict(BEST_PARAMS["PATCHTST"])
#
# Bảng ablation nên chạy (cùng seed, cùng grid):
#   1. PATCHTST (base)
#   2. PATCHTST_ATTENTION (self-attn, alpha=0.05)   <- đang tốt nhất
#   3. PATCHTST_CROSS, causal=False                 <- chỉ Idea 1
#   4. PATCHTST_CROSS, causal=True                  <- Idea 1 + 2
#
# Phân tích sau train:
#   alpha:  model.cross_attn.alpha.item()
#   độ trễ: w = model.cross_attn.last_attn            # (B, N, 2N)
#           N = model.n_patches
#           w_cn = w[:, :, :N].mean(0)                # đích nhìn Cầu Nổi
#           lag_k = [w_cn.diagonal(offset=-k).mean().item()
#                    for k in range(N)]               # k = 0..N-1 con nước
# ============================================================


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("— Test 1: forward/backward các cấu hình —")
    for lb, hz in [(24, 12), (60, 60), (84, 24)]:
        m = PatchTST_CROSS(input_size=12, context_len=lb, horizon=hz)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        print(f"  lookback={lb} horizon={hz} -> out {tuple(y.shape)} | "
              f"n_patches={m.n_patches} | backward OK")

    print("— Test 2: mặt nạ nhân quả hoạt động đúng —")
    m = PatchTST_CROSS(input_size=12, context_len=36, horizon=12)
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 36, 12))
    w = m.cross_attn.last_attn                     # (B, N, 2N)
    N = m.n_patches
    leak = 0.0
    for i in range(N):
        for s in range(2):
            leak += w[:, i, s * N + i + 1: (s + 1) * N].abs().sum().item()
    print(f"  tổng attention rò rỉ vào j > i: {leak:.2e} (kỳ vọng = 0)")
    assert leak == 0.0, "Mặt nạ nhân quả bị rò rỉ!"

    print("— Test 3: hàng attention chuẩn hóa đúng (tổng = 1) —")
    row_sums = w.sum(dim=-1)
    print(f"  min/max tổng hàng: {row_sums.min():.4f}/{row_sums.max():.4f}")

    print("— Test 4: causal=False cho phép nhìn mọi vị trí —")
    m2 = PatchTST_CROSS(input_size=12, context_len=36, horizon=12, causal=False)
    m2.eval()
    with torch.no_grad():
        m2(torch.randn(2, 36, 12))
    w2 = m2.cross_attn.last_attn
    fut = sum(w2[:, i, s * N + i + 1: (s + 1) * N].abs().sum().item()
              for i in range(N) for s in range(2))
    print(f"  attention vào j > i khi tắt mask: {fut:.4f} (kỳ vọng > 0)")
    assert fut > 0

    print("\nTất cả sanity test PASS.")