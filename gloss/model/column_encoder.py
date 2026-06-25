"""The RT **cell** encoder (MoRE substrate, FiLM removed).

Per cell (row ``u``, feature column ``c``, value ``v``): encode the value with a pytorch-frame stype
encoder and add the frozen-LM column-**name** token — RT's additive cell token, unchanged:

    e_{u,c} = W_v Enc_dtype(v_{u,c})                       # pytorch-frame dtype cell embedding (value)
    x_{u,c} = e_{u,c} + W_name name_c                      # + RT name-as-string token (schema part)

``name_c`` is gathered from the frozen per-column ``name_emb`` table (``gloss.text.schema``) by the
cell's global column id. There is **no documentation / FiLM** here anymore — MoRE's only new signal is
the Mixture-of-Experts FFN in the substrate, routed on a value-free signature built from this same
``name_emb`` (plus modality + recency); see ``model/signature.py`` and ``model/moe.py``.

Cells stay as **tokens** (no pooling here): the encoder scatters per-cell vectors into the dense
``[B, S, d_model]`` grid using the collate's ``cell_placement`` map; the RT substrate
(``rt_substrate.py``) contextualizes them, and a head pools afterward.

Value encoders are ``torch_frame.nn.StypeWiseFeatureEncoder`` (the block relbench's ``HeteroEncoder``
wraps), so we reuse pytorch-frame's dtype encoders rather than reinventing them.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import CellBatch, column_vocab, feature_col_names
from ..data.graph import GraphBundle


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
        name_emb: Tensor,
        *,
        d_model: int = 256,
        enc_channels: int | None = None,
    ):
        super().__init__()
        from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

        self.d_model = d_model
        self.d_text = int(name_emb.shape[1])
        enc_channels = enc_channels or d_model
        self.enc_channels = enc_channels

        vocab = column_vocab(bundle)
        self.cell_encoders = nn.ModuleDict()
        self._sorted_cols: dict[str, list[str]] = {}
        self._col_gids: dict[str, Tensor] = {}
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
            self._col_gids[nt] = torch.tensor([vocab[(nt, c)] for c in sorted_cols], dtype=torch.long)

        # frozen per-column name table (RT schema component); value projection; RT name token.
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.w_v = nn.Linear(enc_channels, d_model)
        self.name_proj = nn.Linear(self.d_text, d_model)

    def encode_type(self, nt: str, tf) -> tuple[Tensor, Tensor]:
        """Encode all rows of one node type -> (cell, value) each ``[n, C, d_model]`` (sorted cols)."""
        x, col_names = self.cell_encoders[nt](tf)                 # [n, C0, enc_channels]
        sorted_cols = self._sorted_cols[nt]
        perm = [col_names.index(c) for c in sorted_cols]
        x = x[:, perm, :]                                          # [n, C, enc_channels]
        gids = self._col_gids[nt].to(self.name_emb.device)
        name = self.name_emb.index_select(0, gids)                # [C, d_text]
        value = self.w_v(x)                                       # [n, C, d_model]  (value component)
        cell = value + self.name_proj(name).unsqueeze(0)          # + RT name token
        return cell, value

    def forward(self, cb: CellBatch, return_value: bool = False):
        """-> cell states ``[B, S, d_model]`` (zeros at pad). If ``return_value``, also the value
        component ``[B, S, d_model]`` (the input to the ``value`` routing arm)."""
        dev = self.name_emb.device
        h = torch.zeros(cb.num_seeds, cb.seq_len, self.d_model, device=dev)
        hv = torch.zeros_like(h) if return_value else None
        for nt in cb.tf_dict:
            if nt not in self.cell_encoders or nt not in cb.cell_placement:
                continue
            cell, value = self.encode_type(nt, cb.tf_dict[nt])   # [n, C, d_model]
            b_idx, s_idx, row_idx, col_idx = cb.cell_placement[nt]
            b_idx, s_idx = b_idx.to(dev), s_idx.to(dev)
            row_idx, col_idx = row_idx.to(dev), col_idx.to(dev)
            h[b_idx, s_idx] = cell[row_idx, col_idx]
            if return_value:
                hv[b_idx, s_idx] = value[row_idx, col_idx]
        mask = (~cb.is_padding).to(dev).unsqueeze(-1)
        if return_value:
            return h * mask, hv * mask
        return h * mask
