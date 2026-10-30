# ============================================================
# patchTST_CTJoint.py — CT-PatchTST + JOINT channel-time attention
#
# Implements the "future work" stated in the CT-PatchTST paper
# (arXiv:2501.08620, Sec. V): their channel attention only relates
# channels WITHIN the same patch position; they propose capturing
# inter-channel AND inter-patch dependencies simultaneously.
#
# This variant replaces the per-patch channel attention with JOINT
# attention over the flattened (M x N) token grid: every channel at
# every patch position can attend to every other channel at every
# other position. Time attention (standard PatchTST encoder) follows,
# as in CT-PatchTST.
#
#   RevIN -> patchify + projection + pos + channel embedding
#     -> [Joint channel-time attention]  tokens = M*N   (B, M*N, D)
#     -> [Time attention] PatchTST encoder, per channel (B*M, N, D)
#     -> flatten head -> target -> denorm
#
# NOTE / honest expectation: with M=12 and small data this is the
# UNRESTRICTED version of what PATCHTST_ATTENTION (3 salinity channels,
# flattened S*N) and PATCHTST_CROSS (causal-masked) already do. Its
# role in the paper is the "naive joint attention" baseline: if the
# physics-restricted variants match or beat it, that is evidence that
# domain-guided restriction is what makes joint attention work on
# small hydrological datasets.
# ============================================================

import torch
import torch.nn as nn

from patchTST_CT import CT_PatchTST, ChannelAttentionBlock


class JointAttentionBlock(nn.Module):
    """Pre-norm transformer block over the flattened (M*N) token grid.

    Input/output: (B, M, N, D).
    self.last_attn: (B, M*N, M*N) — joint attention map. Token order is
    [channel_0 patch 0..N-1 | channel_1 patch 0..N-1 | ...], so block
    (i*N:(i+1)*N, j*N:(j+1)*N) = channel i attending to channel j.
    """

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
        self.last_attn = attn_w.detach()                  # (B, M*N, M*N)
        h = h + self.drop(out)
        h = h + self.drop(self.ffn(self.norm2(h)))

        return h.reshape(B, M, N, D)


class CT_PatchTST_Joint(CT_PatchTST):
    """CT-PatchTST with joint channel-time attention (their future work).

    Reuses everything from CT_PatchTST (RevIN, patchify, channel
    embedding, time encoder, head, use_future) and swaps the per-patch
    channel blocks for joint blocks over the M*N grid.
    """

    def __init__(self, *args, n_joint_layers: int = 1,
                 joint_heads: int = 4, joint_dropout: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.embed.out_features
        d_ff = self.encoder.layers[0].linear1.out_features
        # Replace per-patch channel blocks with joint blocks
        self.channel_blocks = nn.ModuleList()             # unused now
        self.joint_blocks = nn.ModuleList([
            JointAttentionBlock(d_model, joint_heads, d_ff, joint_dropout)
            for _ in range(n_joint_layers)
        ])

    def forward(self, x: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        B, L, M = x.shape
        assert L == self.context_len, (
            f"Model build với context_len={self.context_len}, nhận L={L}.")

        if self.revin is not None:                        # (0) RevIN
            x = self.revin.normalize(x)

        x = x.permute(0, 2, 1).reshape(B * M, L)          # (2) split channels
        x = self._patchify(x)                             # (3) (B*M, N, P)
        x = self.embed_dropout(self.embed(x) + self.pos)  # (4) (B*M, N, D)

        # ---- (4.5) JOINT channel-time attention over M*N tokens ----
        x = x.view(B, M, self.n_patches, -1)
        x = x + self.channel_emb.unsqueeze(0)             # channel identity
        for blk in self.joint_blocks:
            x = blk(x)                                    # (B, M, N, D)
        x = x.reshape(B * M, self.n_patches, -1)
        # ------------------------------------------------------------

        x = self.encoder_norm(self.encoder(x))            # (5) time attention

        y = self.head(x).view(B, M, self.horizon)         # (6)(7)
        y = y[:, self.target_idx, :]
        if self.use_future and x_future is not None:
            y = y + self.future_head(x_future.reshape(x_future.size(0), -1))
        if self.revin is not None:
            y = self.revin.denormalize_target(y, self.target_idx)
        return y


# ============================================================
# INTEGRATION in run_experiment:
#
#   from patchTST_CTJoint import CT_PatchTST_Joint
#   ...
#   elif model_name == "CT_PATCHTST_JOINT":
#       model = CT_PatchTST_Joint(
#           input_size  = ds["train"][0].shape[2],
#           context_len = lookback,
#           horizon     = horizon,
#           d_model     = p["hidden_size"],
#           n_layers    = p["num_layers"],
#           patch_len   = p.get("patch_len", 6),
#           stride      = p.get("stride", 6),
#           target_idx  = 0,
#           n_joint_layers = p.get("n_joint_layers", 1),
#       )
#
#   BEST_PARAMS["CT_PATCHTST_JOINT"] = dict(BEST_PARAMS["PATCHTST"])
#
# Full comparison table this completes (same seeds, same grid):
#   1. PATCHTST            — channel independent (no cross-channel)
#   2. CT_PATCHTST         — per-patch channel attention (published)
#   3. CT_PATCHTST_JOINT   — naive joint channel-time attn (their
#                            future work, unrestricted)
#   4. PATCHTST_ATTENTION  — joint attn RESTRICTED to salinity channels
#   5. PATCHTST_CROSS      — restricted + target-query + causal mask
#
# Channel-level summary of the joint map for figures:
#   w = model.joint_blocks[0].last_attn.mean(0)        # (M*N, M*N)
#   M, N = model.input_size, model.n_patches
#   w_ch = w.view(M, N, M, N).mean(dim=(1, 3))         # (M, M)
# ============================================================


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("— Test 1: forward/backward —")
    for lb, hz in [(24, 12), (60, 60)]:
        m = CT_PatchTST_Joint(input_size=12, context_len=lb, horizon=hz)
        x = torch.randn(4, lb, 12)
        y = m(x)
        F.mse_loss(y, torch.randn(4, hz)).backward()
        n_tok = 12 * m.n_patches
        print(f"  lookback={lb} horizon={hz} -> out {tuple(y.shape)} | "
              f"joint tokens={n_tok} | backward OK")

    print("— Test 2: joint attention map shape & normalization —")
    m = CT_PatchTST_Joint(input_size=12, context_len=36, horizon=12)
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 36, 12))
    w = m.joint_blocks[0].last_attn
    MN = 12 * m.n_patches
    print(f"  attn shape: {tuple(w.shape)} (expected (2, {MN}, {MN})) | "
          f"row sums: {w.sum(-1).min():.4f}/{w.sum(-1).max():.4f}")
    assert w.shape == (2, MN, MN)

    print("— Test 3: use_future compatibility —")
    m3 = CT_PatchTST_Joint(input_size=12, context_len=24, horizon=12,
                           use_future=True)
    y3 = m3(torch.randn(4, 24, 12), torch.randn(4, 12, 6))
    F.mse_loss(y3, torch.randn(4, 12)).backward()
    print(f"  out {tuple(y3.shape)} | OK")

    print("\nAll sanity tests PASS.")