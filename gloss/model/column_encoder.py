"""Phase 2 (DOC-RT core) — the documentation-conditioned **cell** encoder.

Per cell (row ``u``, feature column ``c``, value ``v``): encode the value with a pytorch-frame stype
encoder, FiLM-modulate it by the column's grounded documentation ``d_c``, and add an RT-style
column-**name** token:

    e_{u,c} = W_v Enc_dtype(v_{u,c})                      # pytorch-frame dtype cell embedding
    x_{u,c} = γ(d_c) ⊙ e_{u,c} + β(d_c)  +  W_name name_c # FiLM by docs + RT name-as-string token

``d_c`` comes from the regime-dependent grounding ``emb`` (full=docs, null=d_null, shuffled=placebo,
name_only=name). Ungrounded columns fall back to a learned ``d_null`` — so the encoder degrades
gracefully to RT's name-only regime instead of breaking. The RT name token (``name_c``) is the same in
every regime, so it never confounds the docs comparison.

Cells stay as **tokens** (no pooling here): the encoder scatters per-cell vectors into the dense
``[B, S, d_model]`` grid using the collate's ``cell_placement`` map; the RT substrate
(``rt_substrate.py``) contextualizes them, and a head pools afterward.

Value encoders are ``torch_frame.nn.StypeWiseFeatureEncoder`` (the block relbench's ``HeteroEncoder``
wraps), so DOC-RT reuses pytorch-frame's dtype encoders rather than reinventing them.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import CellBatch, feature_col_names
from ..data.graph import GraphBundle
from ..docs.grounding import GroundingResult


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


class CellEncoder(nn.Module):
    def __init__(
        self,
        bundle: GraphBundle,
        *,
        d_model: int = 256,
        d_text: int = 2560,
        enc_channels: int | None = None,
    ):
        super().__init__()
        from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

        self.d_model = d_model
        self.d_text = d_text
        enc_channels = enc_channels or d_model
        self.enc_channels = enc_channels

        self.cell_encoders = nn.ModuleDict()
        self._sorted_cols: dict[str, list[str]] = {}
        for nt in bundle.node_types:
            tf = bundle.data[nt].tf
            col_names_dict = tf.col_names_dict
            stype_enc = _default_stype_encoders(col_names_dict)
            sorted_cols = feature_col_names(tf)
            self._sorted_cols[nt] = sorted_cols
            if not stype_enc or not sorted_cols:
                continue
            self.cell_encoders[nt] = StypeWiseFeatureEncoder(
                out_channels=enc_channels,
                col_stats=bundle.col_stats_dict[nt],
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_enc,
            )

        # FiLM from doc embedding; value projection; RT name token; null fallback
        self.gamma = nn.Linear(d_text, d_model)
        self.beta = nn.Linear(d_text, d_model)
        self.w_v = nn.Linear(enc_channels, d_model)
        self.name_proj = nn.Linear(d_text, d_model)
        self.d_null = nn.Parameter(torch.zeros(d_text))

    def _doc_name_for_columns(self, keys: list[str], grounding: GroundingResult) -> tuple[Tensor, Tensor]:
        """-> (d_c [C, d_text] with d_null where ungrounded, name [C, d_text])."""
        emb, name, _rel, grounded = grounding.gather(keys)
        emb = emb.to(self.d_null)
        name = name.to(self.d_null)
        grounded = grounded.to(self.d_null.device)
        d_c = torch.where(grounded.unsqueeze(-1), emb, self.d_null.unsqueeze(0))
        return d_c, name

    def encode_type(self, nt: str, tf, grounding: GroundingResult) -> Tensor:
        """Encode all rows of one node type -> per-cell ``[n, C, d_model]`` (columns in sorted order)."""
        x, col_names = self.cell_encoders[nt](tf)                 # [n, C0, enc_channels]
        sorted_cols = self._sorted_cols[nt]
        perm = [col_names.index(c) for c in sorted_cols]
        x = x[:, perm, :]                                          # [n, C, enc_channels]
        keys = [f"col::{nt}::{c}" for c in sorted_cols]
        d_c, name = self._doc_name_for_columns(keys, grounding)   # [C, d_text]
        gamma = self.gamma(d_c).unsqueeze(0)                      # [1, C, d_model]
        beta = self.beta(d_c).unsqueeze(0)
        film = gamma * self.w_v(x) + beta                         # [n, C, d_model]  FiLM by docs
        film = film + self.name_proj(name).unsqueeze(0)           # + RT name token (per column)
        return film

    def forward(self, cb: CellBatch, grounding: GroundingResult) -> Tensor:
        """-> cell states ``[B, S, d_model]`` (zeros at pad positions)."""
        dev = self.d_null.device
        h = torch.zeros(cb.num_seeds, cb.seq_len, self.d_model, device=dev)
        for nt in cb.tf_dict:
            if nt not in self.cell_encoders or nt not in cb.cell_placement:
                continue
            x = self.encode_type(nt, cb.tf_dict[nt], grounding)   # [n, C, d_model]
            b_idx, s_idx, row_idx, col_idx = cb.cell_placement[nt]
            h[b_idx.to(dev), s_idx.to(dev)] = x[row_idx.to(dev), col_idx.to(dev)]
        return h * (~cb.is_padding).to(dev).unsqueeze(-1)
