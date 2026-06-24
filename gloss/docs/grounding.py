"""Phase 1 — grounding: map each schema element to (a) its column-**name** embedding and (b) a pooled
**documentation** embedding, for the four DOC-RT regimes.

For element ``e`` with descriptor/name query ``q_e`` and corpus spans ``s_k``:
  name_e    = encode(q_e, 'query')                       # the column-name embedding (RT name token);
                                                          # ALWAYS real, regime-independent.
  sims      = cos(name_e, encode(s_k, 'document'))
  d_e       = softmax(top-K(sims)/temp) · s_topk         if max sims >= threshold   (grounded)
            = 0  (-> the cell encoder substitutes a learned d_null)                 otherwise

The cell encoder FiLM-conditions on ``emb`` (= d_e) and adds an RT name token from ``name_emb``. The
regime decides what goes into ``emb``:
  full       — grounded documentation d_e (the method).
  null       — every element ungrounded (FiLM falls back to d_null): RT names-only baseline (docs OFF).
  shuffled   — PLACEBO: the element->doc assignment is permuted (derangement) so each element receives
               another element's pooled doc — coverage/length matched, meaning decorrelated.
  name_only  — FiLM-condition on the column-name embedding itself (RELATE-style control: docs must beat
               *names*, not just beat nothing).

``name_emb`` (the RT name token) is held identical across all four regimes, so it never confounds the
docs comparison (same role as the self-labels).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .corpus import SchemaElement

REGIMES = ("full", "null", "shuffled", "name_only")


@dataclass
class GroundingConfig:
    chunk_sentences: int = 3
    top_k: int = 4
    sim_threshold: float = 0.3
    temp: float = 0.1


@dataclass
class GroundingResult:
    d_text: int
    regime: str
    keys: list[str]                 # element order
    emb: Tensor                     # [E, d_text]  FiLM conditioning (d_e; regime-dependent)
    name_emb: Tensor                # [E, d_text]  column-name embedding (RT name token; regime-indep.)
    rel: Tensor                     # [E]          (max cosine; 0 where ungrounded)
    grounded: Tensor                # [E] bool     (True => use emb; False => cell encoder uses d_null)
    key_to_row: dict                # key -> row index

    def grounded_by_key(self) -> dict[str, bool]:
        return {k: bool(self.grounded[i]) for k, i in self.key_to_row.items()}

    def gather(self, keys: list[str]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gather (emb, name_emb, rel, grounded) for ``keys`` (unknown keys -> ungrounded zeros)."""
        rows = [self.key_to_row.get(k, -1) for k in keys]
        idx = torch.tensor(rows)
        ok = idx >= 0
        emb = torch.zeros(len(keys), self.d_text)
        name = torch.zeros(len(keys), self.d_text)
        rel = torch.zeros(len(keys))
        grd = torch.zeros(len(keys), dtype=torch.bool)
        if ok.any():
            emb[ok] = self.emb[idx[ok]]
            name[ok] = self.name_emb[idx[ok]]
            rel[ok] = self.rel[idx[ok]]
            grd[ok] = self.grounded[idx[ok]]
        return emb, name, rel, grd


def ground(
    elements: list[SchemaElement],
    spans: list[str],
    encode,
    cfg: GroundingConfig | None = None,
    *,
    regime: str = "full",
    seed: int = 0,
) -> GroundingResult:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    cfg = cfg or GroundingConfig()
    keys = [e.key for e in elements]
    key_to_row = {k: i for i, k in enumerate(keys)}
    E = len(elements)

    # column-name embeddings: ALWAYS real (the RT name token + the name_only control).
    name_emb = encode([e.query for e in elements], kind="query")          # [E, d]
    d_text = name_emb.shape[1]

    # pooled documentation d_e (full/shuffled). null/name_only never use the doc pooling.
    doc = torch.zeros(E, d_text)
    rel = torch.zeros(E)
    grounded = torch.zeros(E, dtype=torch.bool)
    if spans and regime in ("full", "shuffled"):
        span_emb = encode(spans, kind="document")                         # [M, d]
        sims = name_emb @ span_emb.T                                      # [E, M] (unit-norm rows)
        k = min(cfg.top_k, span_emb.shape[0])
        top_sims, top_idx = sims.topk(k, dim=1)
        rel = top_sims[:, 0]
        grounded = rel >= cfg.sim_threshold
        weights = torch.softmax(top_sims / cfg.temp, dim=1).unsqueeze(-1)  # [E, k, 1]
        pooled = (weights * span_emb[top_idx]).sum(dim=1)                 # [E, d]
        doc = torch.where(grounded.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        rel = torch.where(grounded, rel, torch.zeros_like(rel))

    if regime == "full":
        emb = doc
    elif regime == "null":
        emb = torch.zeros(E, d_text)
        grounded = torch.zeros(E, dtype=torch.bool)
    elif regime == "shuffled":
        perm = _derangement(E, seed)
        emb, rel, grounded = doc[perm], rel[perm], grounded[perm]
    else:  # name_only
        emb = name_emb.clone()
        grounded = torch.ones(E, dtype=torch.bool)
        rel = torch.ones(E)

    return GroundingResult(
        d_text=d_text, regime=regime, keys=keys, emb=emb, name_emb=name_emb,
        rel=rel, grounded=grounded, key_to_row=key_to_row,
    )


def _derangement(n: int, seed: int) -> Tensor:
    if n <= 1:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    for _ in range(32):
        perm = torch.randperm(n, generator=g)
        if (perm == torch.arange(n)).sum() == 0:
            return perm
    return torch.roll(torch.arange(n), 1)  # fallback: guaranteed fixed-point-free
