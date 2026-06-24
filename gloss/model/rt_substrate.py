"""Phase 0 (DOC-RT substrate) — RT's cell-token transformer with relational attention.

Faithful port of the attention core of the Relational Transformer (RT, Ranjan et al., ICLR 2026,
arXiv:2510.06377; reference code snap-stanford/relational-transformer, CC-BY-4.0). We reimplement it
in-stack (PyTorch + SDPA) rather than adopting RT's Rust sampler / pixi pipeline.

A ``RelationalBlock`` runs four masked attentions over the flat ``[B, S]`` cell sequence, then a SwiGLU
FFN, all pre-norm (RMSNorm):
  * ``col``  — same column of the same table (across rows).
  * ``feat`` — same row, or a row this cell's row references via a foreign key (forward FK).
  * ``nbr``  — a row that references this cell's row (reverse FK).
  * ``full`` — global (all non-pad cells).

The four boolean ``[B, S, S]`` masks are derived from the collate's index tensors
(``node_idxs``/``col_idxs``/``f2p_nbr_idxs``) and reused across all blocks. The identity is OR-ed into
every mask so no query row is fully masked (which would NaN softmax); real cells never attend pad cells.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.collate import CellBatch

REL_ORDER = ("col", "feat", "nbr", "full")


def build_relational_masks(cb: CellBatch) -> dict[str, Tensor]:
    """Build the four boolean ``[B, S, S]`` attention masks (True = attend) from a :class:`CellBatch`."""
    real = (~cb.is_padding)                              # [B, S]
    pad_pair = real.unsqueeze(2) & real.unsqueeze(1)     # [B, S, S]
    node = cb.node_idxs                                  # [B, S]
    col = cb.col_idxs

    same_node = (node.unsqueeze(2) == node.unsqueeze(1)) & pad_pair
    same_col = (col.unsqueeze(2) == col.unsqueeze(1)) & pad_pair

    B, S = node.shape
    kv_in_f2p = torch.zeros(B, S, S, dtype=torch.bool, device=node.device)
    for k in range(cb.max_fk):
        fk = cb.f2p_nbr_idxs[:, :, k]                    # [B, S]  parent local idx for q's slot k (-1 none)
        match = (fk.unsqueeze(2) == node.unsqueeze(1)) & (fk.unsqueeze(2) >= 0)
        kv_in_f2p |= match
    kv_in_f2p &= pad_pair
    q_in_f2p = kv_in_f2p.transpose(1, 2)                 # i is among j's f2p neighbors

    eye = torch.eye(S, dtype=torch.bool, device=node.device).unsqueeze(0)
    return {
        "col": same_col | eye,
        "feat": (same_node | kv_in_f2p) | eye,
        "nbr": q_in_f2p | eye,
        "full": pad_pair | eye,
    }


class MaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, S, _ = x.shape
        H, hd = self.num_heads, self.head_dim
        q = self.wq(x).view(B, S, H, hd).transpose(1, 2)   # [B, H, S, hd]
        k = self.wk(x).view(B, S, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, S, H, hd).transpose(1, 2)
        attn_mask = mask.unsqueeze(1)                      # [B, 1, S, S] bool (True = attend)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H * hd)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in (*REL_ORDER, "ffn")})
        self.attns = nn.ModuleDict({l: MaskedAttention(d_model, num_heads) for l in REL_ORDER})
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, masks: dict[str, Tensor]) -> Tensor:
        for l in REL_ORDER:
            x = x + self.attns[l](self.norms[l](x), masks[l])
        x = x + self.ffn(self.norms["ffn"](x))
        return x


class RTSubstrate(nn.Module):
    """Stack of :class:`RelationalBlock`s over the flat cell sequence."""

    def __init__(self, *, d_model: int = 256, n_blocks: int = 8, n_heads: int = 8, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.blocks = nn.ModuleList(RelationalBlock(d_model, n_heads, d_ff) for _ in range(n_blocks))
        self.norm_out = nn.RMSNorm(d_model)

    def forward(self, x: Tensor, cb: CellBatch) -> Tensor:
        masks = build_relational_masks(cb)
        for block in self.blocks:
            x = block(x, masks)
        x = self.norm_out(x)
        return x * (~cb.is_padding).unsqueeze(-1)
