"""Phase 3 — task heads. A seed-node readout for entity tasks (binary/regression).

The masked-attribute / temporal-autocomplete pretraining heads belong to Phase 4; only the supervised
entity head is needed for the Phase-3 forward DoD.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import GlossBatch


class EntityHead(nn.Module):
    """Read out the seed node of each subgraph and project to ``out_dim`` logits."""

    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, node_states: Tensor, gb: GlossBatch) -> Tensor:
        """node_states ``[B, N, d]`` -> logits ``[B, out_dim]`` taken at each seed node."""
        B, N, d = node_states.shape
        seed = gb.is_seed.to(node_states.device)
        # exactly one seed per segment; if (defensively) more, average them
        w = seed.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (node_states * w.unsqueeze(-1)).sum(dim=1)      # [B, d]
        return self.mlp(pooled)
