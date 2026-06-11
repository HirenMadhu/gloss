"""Phase 1 — grounding: map each schema element to a pooled documentation embedding.

For element ``e`` with descriptor query ``q_e`` and corpus spans ``s_k``:
  sims      = cos(encode(q_e, 'query'), encode(s_k, 'document'))
  d_e       = softmax(top-K(sims)/temp) · s_topk      if max sims >= threshold
            = 0 (ungrounded -> the model substitutes a learned d_null)   otherwise
  rel_e     = max_k sims                               (kept as a relevance feature)

Regimes:
  full            — as above.
  null            — every element ungrounded (name-only / RT-like floor).
  shuffled_spans  — PLACEBO: retrieval runs normally, then the per-element results are permuted across
                    elements (each element gets *another* element's pooled doc). Length/-distribution
                    matched and same coverage stats, but the meaning routed to each element is wrong —
                    the audit's negative control. (Permuting the span matrix itself is a no-op, since
                    cosine is recomputed against the same vectors; the correspondence must be broken at
                    the element level.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .corpus import SchemaElement

REGIMES = ("full", "null", "shuffled_spans")


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
    emb: Tensor                     # [E, d_text]  (zeros where ungrounded)
    rel: Tensor                     # [E]          (max cosine; 0 where ungrounded)
    grounded: Tensor                # [E] bool
    key_to_row: dict                # key -> row index

    def grounded_by_key(self) -> dict[str, bool]:
        return {k: bool(self.grounded[i]) for k, i in self.key_to_row.items()}

    def gather(self, keys: list[str]) -> tuple[Tensor, Tensor, Tensor]:
        """Gather (emb, rel, grounded) for ``keys`` (unknown keys -> ungrounded zeros)."""
        rows, miss = [], []
        for k in keys:
            rows.append(self.key_to_row.get(k, -1))
        idx = torch.tensor(rows)
        ok = idx >= 0
        emb = torch.zeros(len(keys), self.d_text)
        rel = torch.zeros(len(keys))
        grd = torch.zeros(len(keys), dtype=torch.bool)
        if ok.any():
            emb[ok] = self.emb[idx[ok]]
            rel[ok] = self.rel[idx[ok]]
            grd[ok] = self.grounded[idx[ok]]
        return emb, rel, grd


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

    span_emb = encode(spans, kind="document") if spans else torch.zeros(0, 0)
    d_text = span_emb.shape[1] if span_emb.numel() else int(getattr(encode, "dim", 0))
    E = len(elements)

    if regime == "null" or span_emb.numel() == 0:
        return GroundingResult(
            d_text=d_text, regime=regime, keys=keys,
            emb=torch.zeros(E, d_text), rel=torch.zeros(E),
            grounded=torch.zeros(E, dtype=torch.bool), key_to_row=key_to_row,
        )

    q_emb = encode([e.query for e in elements], kind="query")          # [E, d]
    sims = q_emb @ span_emb.T                                          # [E, S] (rows are unit-norm)
    k = min(cfg.top_k, span_emb.shape[0])
    top_sims, top_idx = sims.topk(k, dim=1)                           # [E, k]
    rel = top_sims[:, 0]
    grounded = rel >= cfg.sim_threshold

    weights = torch.softmax(top_sims / cfg.temp, dim=1).unsqueeze(-1)  # [E, k, 1]
    pooled = (weights * span_emb[top_idx]).sum(dim=1)                  # [E, d]
    emb = torch.where(grounded.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    rel = torch.where(grounded, rel, torch.zeros_like(rel))

    if regime == "shuffled_spans":
        # PLACEBO: permute the element->doc assignment (a derangement when E>1) so each element
        # receives another element's pooled doc. Same multiset of docs (coverage/length matched),
        # decorrelated from the element's own meaning.
        perm = _derangement(E, seed)
        emb, rel, grounded = emb[perm], rel[perm], grounded[perm]

    return GroundingResult(
        d_text=d_text, regime=regime, keys=keys, emb=emb, rel=rel,
        grounded=grounded, key_to_row=key_to_row,
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
