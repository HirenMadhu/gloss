"""The masked-cell decode head — the piece ``model/heads.py`` said "does not exist yet".

Two decoders, one per maskable modality, and **neither has a parameter whose shape depends on the
database**:

numerical
    ``RMSNorm -> Linear(d_model, 1)``, zero-initialised, trained with Huber on the column-z-scored
    value. This is RT's head verbatim (``dec_dict['number'] = Linear(d_model, 1)``, zero-init,
    ``F.huber_loss``). Huber rather than MSE because RelBench numerics are heavy-tailed — rel-trial's
    ``study-adverse`` target has train skew 39 — and a z-score of 500 under MSE would own the batch.

categorical
    ``RMSNorm -> Linear(d_model, enc_channels)``, scored **against the column's own slice of the
    cell encoder's category-embedding table**. torch_frame keeps one ``nn.Embedding`` per table for
    all its categorical columns, so column ``j``'s classes are rows
    ``cat_base[j] .. cat_base[j] + n_cat[j]``; the logits are that slice times the projected cell
    state. RT has no categorical head at all (it routes low-cardinality strings through the frozen
    text encoder instead), so this is ours — and tying it to the encoder table is what keeps it legal
    under changes.md §0: the *only* vocabulary-shaped tensor involved already belongs to the per-DB
    encoder, and the head itself transfers to an unseen schema unchanged.

Losses are reported split three ways — by modality, and within that by whether the cell was the seed
row's designated target (the RT-comparable number) or one of the extra randomly masked cells. Keeping
them apart matters: they have different counts by an order of magnitude, so a single average would be
the random term wearing the seed term's name.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.masking import CATEGORICAL_ID, NUMERICAL_ID
from .row_level import RMSNorm


def _mean_or_zero(x: Tensor, like: Tensor) -> Tensor:
    """Mean of ``x``, or a graph-connected zero when empty (the ``losses.py`` idiom, for DDP)."""
    return x.mean() if x.numel() else like.sum() * 0.0


class MaskedCellHead(nn.Module):
    """Reconstruct masked cells from their contextualized cell states.

    ``forward`` takes the substrate's ``[B, S, d_model]`` cell states plus the flat
    :class:`~gloss.data.masking.MaskedTargets`, and returns ``(loss, stats)``.
    """

    def __init__(self, d_model: int, enc_channels: int, *, lambda_cat: float = 1.0,
                 huber_delta: float = 1.0):
        super().__init__()
        self.lambda_cat = lambda_cat
        self.huber_delta = huber_delta
        self.norm = RMSNorm(d_model)
        self.num_head = nn.Linear(d_model, 1)
        self.cat_proj = nn.Linear(d_model, enc_channels)
        # Zero-init the numerical decoder, as RT does: the first prediction is then exactly the
        # column mean (0 in z-space), so early training cannot be dominated by a random readout of a
        # heavy-tailed column.
        nn.init.zeros_(self.num_head.weight)
        nn.init.zeros_(self.num_head.bias)

    def category_logits(self, hidden: Tensor, gid: Tensor, spec, cat_table: Tensor) -> Tensor:
        """``[M, max_cat]`` logits, ``-inf`` past each column's own class count.

        Columns in one batch have different vocabularies (2 to 42 across the corpus), so the rows are
        padded to the batch's widest and the surplus is masked out before the cross-entropy. Gathering
        a ``[M, max_cat, d]`` slice of the shared table is the cost of doing this without a Python
        loop over columns; ``max_cat`` is small enough that it is the right trade.
        """
        dev = hidden.device
        n_cat = spec.n_cat.to(dev).index_select(0, gid)                      # [M]
        base = spec.cat_base.to(dev).index_select(0, gid)                    # [M]
        width = int(n_cat.max())
        ar = torch.arange(width, device=dev).unsqueeze(0)                    # [1, K]
        valid = ar < n_cat.unsqueeze(1)                                      # [M, K]
        rows = (base.unsqueeze(1) + ar).clamp(0, cat_table.shape[0] - 1)     # [M, K]
        emb = cat_table.index_select(0, rows.reshape(-1)).view(*rows.shape, -1)
        logits = torch.einsum("md,mkd->mk", self.cat_proj(hidden), emb)
        return logits.masked_fill(~valid, float("-inf"))

    def forward(self, cell_states: Tensor, targets, spec,
                cat_table: Tensor | None = None) -> tuple[Tensor, dict]:
        """-> ``(loss, stats)``. ``cat_table`` is the per-DB category embedding matrix to tie against.

        Returns a graph-connected zero when nothing was masked, so a degenerate batch cannot desync
        DDP ranks or produce a ``None`` gradient.
        """
        zero = cell_states.sum() * 0.0
        stats: dict[str, float] = {}
        if len(targets) == 0:
            return zero, {"n_num": 0.0, "n_cat": 0.0, "n_seed": 0.0,
                          "n_seeds_without_target": float(targets.n_seeds_without_target)}

        h = self.norm(cell_states[targets.b, targets.s])                     # [M, d]
        is_num = targets.stype == NUMERICAL_ID
        is_cat = targets.stype == CATEGORICAL_ID

        loss_num = zero
        if bool(is_num.any()):
            pred = self.num_head(h[is_num]).squeeze(-1)
            per = F.huber_loss(pred, targets.y_num[is_num], reduction="none",
                               delta=self.huber_delta)
            loss_num = per.mean()
            seed = targets.is_seed[is_num]
            stats["loss_num_seed"] = float(_mean_or_zero(per[seed], zero).detach())
            stats["loss_num_random"] = float(_mean_or_zero(per[~seed], zero).detach())

        loss_cat = zero
        if bool(is_cat.any()) and cat_table is not None:
            logits = self.category_logits(h[is_cat], targets.gid[is_cat], spec, cat_table)
            per = F.cross_entropy(logits, targets.y_cat[is_cat], reduction="none")
            loss_cat = per.mean()
            seed = targets.is_seed[is_cat]
            stats["loss_cat_seed"] = float(_mean_or_zero(per[seed], zero).detach())
            stats["loss_cat_random"] = float(_mean_or_zero(per[~seed], zero).detach())
            stats["cat_acc"] = float((logits.argmax(-1) == targets.y_cat[is_cat]).float().mean())

        stats.update(
            loss_num=float(loss_num.detach()), loss_cat=float(loss_cat.detach()),
            n_num=float(int(is_num.sum())), n_cat=float(int(is_cat.sum())),
            n_seed=float(int(targets.is_seed.sum())), n_masked=float(len(targets)),
            n_seeds_without_target=float(targets.n_seeds_without_target),
        )
        return loss_num + self.lambda_cat * loss_cat, stats
