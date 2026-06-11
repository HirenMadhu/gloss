"""Phase 2 — documentation-conditioned node encoder.

Per cell (column ``c``, value ``v``): encode the value with a pytorch-frame stype encoder, then
**FiLM-modulate by the column's grounded documentation** ``d_c``:

    x_c = γ(d_c) ⊙ W_v Enc_dtype(v_c) + β(d_c)

Pool cells to a node vector with a **doc-keyed** attention pool (documentation decides which columns
matter), then add a node-type embedding and the dimensionless node-time term:

    h_u = AttnPool_c(x_c; keys = d_c) + E_type(t) + W_t φ(τ_u)

Ungrounded columns (below the grounding threshold, or the ``null`` regime) fall back to a learned
``d_null`` — so the encoder degrades gracefully toward name-only instead of breaking. The doc regime is
already baked into the ``GroundingResult`` passed in, which is how switching full↔null changes features.

Per-cell embeddings come from ``torch_frame.nn.StypeWiseFeatureEncoder`` (the building block relbench's
``HeteroEncoder`` wraps), so HALOS reuses pytorch-frame's dtype encoders rather than reinventing them.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import GlossBatch
from ..data.graph import GraphBundle
from ..docs.grounding import GroundingResult
from .time_encoding import BochnerTime, node_tau


def _default_stype_encoders(col_names_dict):
    import torch_frame
    from torch_frame.nn import (
        EmbeddingEncoder,
        LinearEmbeddingEncoder,
        LinearEncoder,
        MultiCategoricalEmbeddingEncoder,
        TimestampEncoder,
    )

    defaults = {
        torch_frame.categorical: EmbeddingEncoder,
        torch_frame.numerical: LinearEncoder,
        torch_frame.multicategorical: MultiCategoricalEmbeddingEncoder,
        torch_frame.embedding: LinearEmbeddingEncoder,
        torch_frame.timestamp: TimestampEncoder,
    }
    return {st: defaults[st]() for st in col_names_dict if st in defaults}


class ColumnEncoder(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        *,
        d_model: int = 256,
        d_text: int = 2560,
        enc_channels: int | None = None,
        n_freq: int = 16,
    ):
        super().__init__()
        from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

        self.d_model = d_model
        self.d_text = d_text
        enc_channels = enc_channels or d_model
        self.enc_channels = enc_channels

        # one per-cell stype encoder per node type, plus the ordered (col_name -> grounding key) map
        self.cell_encoders = nn.ModuleDict()
        self._col_keys: dict[str, list[str]] = {}
        for nt in bundle.node_types:
            tf = bundle.data[nt].tf
            col_names_dict = tf.col_names_dict
            stype_enc = _default_stype_encoders(col_names_dict)
            if not stype_enc:
                self._col_keys[nt] = []
                continue
            self.cell_encoders[nt] = StypeWiseFeatureEncoder(
                out_channels=enc_channels,
                col_stats=bundle.col_stats_dict[nt],
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_enc,
            )
            ordered = [c for st in col_names_dict for c in col_names_dict[st]]
            self._col_keys[nt] = [f"col::{nt}::{c}" for c in ordered]

        # FiLM from doc embedding; value projection; doc-keyed attention pool
        self.gamma = nn.Linear(d_text, d_model)
        self.beta = nn.Linear(d_text, d_model)
        self.w_v = nn.Linear(enc_channels, d_model)
        self.key_proj = nn.Linear(d_text, d_model)
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.d_null = nn.Parameter(torch.zeros(d_text))

        self.e_type = nn.Embedding(bundle.num_node_types, d_model)
        self.bochner = BochnerTime(n_freq)
        self.w_t = nn.Linear(n_freq, d_model)
        self.node_type_id = dict(bundle.node_type_id)

    def _doc_for_columns(self, keys: list[str], grounding: GroundingResult) -> Tensor:
        """Gather d_c per column; substitute the learned d_null where ungrounded. -> [C, d_text]."""
        emb, _rel, grounded = grounding.gather(keys)
        emb = emb.to(self.d_null.dtype).to(self.d_null.device)
        grounded = grounded.to(self.d_null.device)
        return torch.where(grounded.unsqueeze(-1), emb, self.d_null.unsqueeze(0))

    def encode_type(self, nt: str, tf, grounding: GroundingResult) -> Tensor:
        """Encode all rows of one node type -> [n, d_model]."""
        n = tf.num_rows
        if nt not in self.cell_encoders:
            return self.pool_query.new_zeros(n, self.d_model)
        x, col_names = self.cell_encoders[nt](tf)            # x: [n, C, enc_channels]
        keys = [f"col::{nt}::{c}" for c in col_names]
        d_c = self._doc_for_columns(keys, grounding)         # [C, d_text]
        gamma = self.gamma(d_c).unsqueeze(0)                 # [1, C, d_model]
        beta = self.beta(d_c).unsqueeze(0)
        x_c = gamma * self.w_v(x) + beta                     # [n, C, d_model]  (FiLM per column)
        # doc-keyed attention pool over columns (weights shared across rows)
        col_keys = self.key_proj(d_c)                        # [C, d_model]
        scores = (col_keys @ self.pool_query) / (self.d_model ** 0.5)   # [C]
        weights = torch.softmax(scores, dim=0).view(1, -1, 1)          # [1, C, 1]
        return (weights * x_c).sum(dim=1)                    # [n, d_model]

    def forward(self, gb: GlossBatch, grounding: GroundingResult) -> Tensor:
        """-> node states ``[B, N_max, d_model]`` (zeros at pad positions)."""
        B, N = gb.num_seeds, gb.n_max
        dev = self.pool_query.device
        h = torch.zeros(B, N, self.d_model, device=dev)
        for nt, tf in gb.tf_dict.items():
            h_nt = self.encode_type(nt, tf, grounding)       # [n_t, d_model]
            seg, pos = gb.placement[nt]
            h[seg.to(dev), pos.to(dev)] = h_nt.to(dev)

        # + node-type embedding (real nodes only)
        ntid = gb.node_type_id.clamp_min(0).to(dev)
        h = h + self.e_type(ntid) * gb.pad_mask.unsqueeze(-1).to(dev)

        # + node-time term W_t φ(τ_u) for timed nodes
        tau_u, valid = node_tau(gb.seed_time, gb.row_time, gb.t_ctx, gb.is_timed)
        phi = self.bochner(tau_u.to(dev))                    # [B, N, n_freq]
        h = h + self.w_t(phi) * valid.to(dev).unsqueeze(-1)
        return h * gb.pad_mask.unsqueeze(-1).to(dev)
