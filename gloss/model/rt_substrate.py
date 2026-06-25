"""Phase 0 (DOC-RT substrate) — RT's cell-token transformer with relational attention.

Faithful port of the attention core of the Relational Transformer (RT, Ranjan et al., ICLR 2026,
arXiv:2510.06377; reference code snap-stanford/relational-transformer, CC-BY-4.0). We reimplement it
in-stack (PyTorch + SDPA) rather than adopting RT's Rust sampler / pixi pipeline.

A ``RelationalBlock`` runs four masked attentions over the flat ``[B, S]`` cell sequence, then an FFN
(a ``SwiGLU``, or a Mixture-of-Experts FFN routed on the relational signature — MoRE's only new
mechanism), all pre-norm (RMSNorm):
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
from .moe import MoEFFN, SwiGLU

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


class RelationalBlock(nn.Module):
    """col→feat→nbr→full masked attention + an FFN (SwiGLU, or a MoE FFN when ``moe``).

    A MoE block returns the router-orthogonality loss as its second output; a dense block returns 0.
    ``route_on`` selects what the router reads: ``signature`` (``z``), ``hidden`` (the block's own
    normed hidden), ``value`` (the cell's value component), or ``identity`` (a learned per-column id).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        moe: bool = False,
        d_route: int | None = None,
        num_experts: int = 4,
        k: int = 2,
        route_on: str = "dense",
    ):
        super().__init__()
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in (*REL_ORDER, "ffn")})
        self.attns = nn.ModuleDict({l: MaskedAttention(d_model, num_heads) for l in REL_ORDER})
        self.moe = moe
        self.route_on = route_on
        if moe:
            self.ffn = MoEFFN(d_model, d_ff, int(d_route), num_experts=num_experts, k=k)
        else:
            self.ffn = SwiGLU(d_model, d_ff)

    def _route_feat(self, h, z, value_feat, id_emb):
        return {"signature": z, "hidden": h, "value": value_feat, "identity": id_emb}[self.route_on]

    def forward(self, x, masks, *, z=None, value_feat=None, id_emb=None):
        for l in REL_ORDER:
            x = x + self.attns[l](self.norms[l](x), masks[l])
        h = self.norms["ffn"](x)
        if self.moe:
            y, _gates = self.ffn(h, self._route_feat(h, z, value_feat, id_emb))
            return x + y, self.ffn.ortho_loss()
        return x + self.ffn(h), x.new_zeros(())


class RTSubstrate(nn.Module):
    """Stack of :class:`RelationalBlock`s over the flat cell sequence.

    ``route_on`` configures the whole stack. ``dense`` / ``dense_wide`` are pure RT (no router;
    ``dense_wide`` widens the FFN by ``k`` as a param-matched control). The MoE arms (signature /
    hidden / value / identity) put a :class:`MoEFFN` in the placed blocks; ``forward`` threads the
    routing tensors and returns ``(states, aux)`` where ``aux`` is the summed router-orthogonality loss.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_blocks: int = 8,
        n_heads: int = 8,
        d_ff: int | None = None,
        route_on: str = "dense",
        d_sig: int = 128,
        num_experts: int = 4,
        k: int = 2,
        moe_placement: str = "all",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        is_moe_arm = route_on not in ("dense", "dense_wide")
        d_route = d_sig if route_on in ("signature", "identity") else d_model
        moe_idx = self._placement(n_blocks, moe_placement) if is_moe_arm else set()
        blocks = []
        for i in range(n_blocks):
            block_is_moe = i in moe_idx
            block_d_ff = d_ff * k if route_on == "dense_wide" else d_ff
            blocks.append(RelationalBlock(
                d_model, n_heads, d_ff if block_is_moe else block_d_ff,
                moe=block_is_moe, d_route=d_route, num_experts=num_experts, k=k,
                route_on=route_on if block_is_moe else "dense",
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.RMSNorm(d_model)
        self.route_on = route_on

    @staticmethod
    def _placement(n_blocks: int, placement: str) -> set[int]:
        if placement == "upper_half":
            return set(range(n_blocks // 2, n_blocks))
        return set(range(n_blocks))                              # "all"

    def forward(self, x: Tensor, cb: CellBatch, *, z=None, value_feat=None, id_emb=None):
        masks = build_relational_masks(cb)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, a = block(x, masks, z=z, value_feat=value_feat, id_emb=id_emb)
            aux = aux + a
        x = self.norm_out(x)
        return x * (~cb.is_padding).unsqueeze(-1), aux
