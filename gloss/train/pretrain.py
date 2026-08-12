"""Masked-cell pretraining: the LightningModule, the LR schedule, and checkpointing.

Three things here that the supervised path (`train/loop.py`, `train/finetune.py`) does not have, and
each exists for a stated reason:

**An LR schedule.** There is none anywhere else in this repo — `configure_optimizers` returns a bare
AdamW. Fine for a 10-epoch finetune, not for a 50k-step pretrain. RT uses linear warmup over the first
20% then linear decay; that is reproduced here explicitly rather than through `OneCycleLR`, whose
defaults quietly start at `max_lr/25`, end at `max_lr/2.5e8`, and cycle Adam's beta1 from 0.95 to 0.85
and back — none of which is in the paper's description of it.

**Disk checkpoints.** Also absent everywhere else, and for a real reason: `MoRELitModule.__init__`
takes the unserializable `GraphBundle`, so `enable_checkpointing=False` is set throughout and
`_BestValState` keeps weights in memory. Pretraining has to survive a 16 h SLURM wall, so
:class:`PretrainCheckpoint` writes plain `state_dict`s split into a **portable trunk** and **per-DB
adapters**, plus optimizer/scheduler/step for resume. The split is the point: `cell_encoders` are
sized by one database's columns and categories, everything else is not, so the trunk is what LODO and
finetuning load.

**Aux applied once.** `loop.py:86` multiplies the whole `aux` by `lambda_ortho` even though `RowMoE`
has already weighted its own ortho and balance terms internally (`row_level.py:431`), so the row terms
effectively get lambda squared. That is left alone there — changing it would move every reported
finetune baseline — but it is not reproduced here. This module weights each level once, explicitly,
and logs the raw and weighted terms separately.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytorch_lightning as pl
import torch

from ..data.collate import to_cell_batch
from ..data.masking import build_column_target_spec, gather_masked_targets, sample_cell_mask
from ..model.mlm_head import MaskedCellHead
from ..model.more import MoRE

#: The one sub-tree of the model that is sized by a particular database's schema. Matched as a
#: SUBSTRING, not a prefix: this function is called on a LightningModule's state_dict, whose keys are
#: already prefixed (`model.encoder.cell_encoders...`), and a `startswith` test silently classified
#: every per-DB encoder as portable — i.e. wrote them into the trunk and defeated the whole split.
ADAPTER_MARKERS = ("encoder.cell_encoders.",)


def is_adapter_key(key: str) -> bool:
    return any(m in key for m in ADAPTER_MARKERS)


def split_state_dict(state: dict) -> tuple[dict, dict]:
    """-> ``(trunk, adapter)``. Trunk is everything not sized by a particular database.

    Verified by diffing a model built on rel-f1 (9 tables / 13 roles / 45 columns) against one built
    on rel-trial (15 / 15 / 103): of the keys present in both, **zero** differ in shape, and every
    non-shared key is under ``encoder.cell_encoders``. So the split is exhaustive, not hopeful.

    Two things make that true and are easy to break. The frozen name tables are registered
    ``persistent=False`` precisely so no ``state_dict`` entry has a `C`-, `K`- or `n_tables`-shaped
    row count (`row_level._frozen`), and `RowAttention` derives `K` from the frozen table at forward
    time rather than owning an `R^{2K+2}` parameter. What remains schema-bound is only the
    pytorch-frame stype encoders, whose category tables and numerical mean/std are per column.
    """
    trunk, adapter = {}, {}
    for k, v in state.items():
        (adapter if is_adapter_key(k) else trunk)[k] = v
    return trunk, adapter


def linear_warmup_decay(max_steps: int, warmup_frac: float = 0.2, min_frac: float = 0.0):
    """RT's schedule: linear 0 -> 1 over the first ``warmup_frac``, then linear 1 -> ``min_frac``."""
    warmup = max(int(round(warmup_frac * max_steps)), 1)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(max_steps - warmup, 1)
        return max(min_frac, 1.0 - prog)

    return fn


class MoREPretrainLitModule(pl.LightningModule):
    """MoRE + a masked-cell decode head, trained on a task-free stream of `(node_type, batch)`."""

    def __init__(
        self,
        bundle,
        name_emb,
        *,
        model_kwargs: dict | None = None,
        route_on: str = "signature",
        p_random: float = 0.15,
        seed_target: bool = True,
        lambda_cat: float = 1.0,
        lambda_ortho: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        max_steps: int = 50_000,
        warmup_frac: float = 0.2,
        seq_len: int = 1024,
        max_fk: int = 5,
        max_rows: int | None = None,
        mask_seed: int = 0,
    ):
        super().__init__()
        self.bundle = bundle
        self.p_random = p_random
        self.seed_target = seed_target
        self.lambda_ortho = lambda_ortho
        self.lr, self.weight_decay, self.betas = lr, weight_decay, betas
        self.max_steps_budget, self.warmup_frac = max_steps, warmup_frac
        self.seq_len, self.max_fk, self.max_rows = seq_len, max_fk, max_rows
        self.mask_seed = mask_seed

        mk = dict(model_kwargs or {})
        mk.pop("d_text", None)
        self.model = MoRE(bundle, name_emb, route_on=route_on, **mk)
        self.spec = build_column_target_spec(bundle, self.model.encoder)
        # Tie the head to whatever category table the chosen encoder exposes: torch_frame's learned
        # per-DB embedding (width `enc_channels`) or the schema-free encoder's frozen LABEL table
        # (width `d_text`). Getting this wrong is a silent shape error only at the first masked
        # categorical cell, which may be many steps in.
        probe = self.model.category_table(next(iter(bundle.node_types), ""))
        self.head = MaskedCellHead(self.model.encoder.d_model, self.model.encoder.enc_channels,
                                   lambda_cat=lambda_cat,
                                   d_cat=None if probe is None else int(probe.shape[-1]))
        self._t0 = time.time()
        self._cells = 0

    # -- data ---------------------------------------------------------------------------------
    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        """``(node_type, HeteroData)`` -> a device-resident ``(node_type, CellBatch)``.

        The node type has to travel with the batch: `to_cell_batch` needs it as `entity_table`, and
        the pretrain stream deliberately draws each batch's seeds from one table (see
        `data/pretrain_loader.py`).
        """
        node_type, raw = batch
        cb = to_cell_batch(raw, self.bundle, node_type, seq_len=self.seq_len,
                           max_fk=self.max_fk, max_rows=self.max_rows)
        return node_type, cb.to(device)

    # -- objective ----------------------------------------------------------------------------
    def _masked_loss(self, node_type, cb, *, step_seed: int):
        g = torch.Generator(device=cb.col_idxs.device).manual_seed(step_seed)
        mask, seed = sample_cell_mask(cb, self.spec, p_random=self.p_random,
                                      seed_target=self.seed_target, generator=g)
        targets = gather_masked_targets(cb, self.spec, mask, seed)
        _logits, aux, cells = self.model(cb, cell_mask=mask, return_cells=True)
        loss, stats = self.head(cells, targets, self.spec, self.model.category_table(node_type))
        stats["frac_masked"] = float(mask.sum()) / max(int((~cb.is_padding).sum()), 1)
        stats["occupancy"] = float((~cb.is_padding).float().mean())
        return loss, aux, stats

    def training_step(self, batch, batch_idx):
        node_type, cb = batch
        # Seeded per global step so the mask is reproducible on resume and identical across ranks.
        loss, aux, stats = self._masked_loss(node_type, cb,
                                             step_seed=self.mask_seed * 1_000_003 + self.global_step)
        total = loss + self.lambda_ortho * aux
        self._cells += int((~cb.is_padding).sum())
        bs = int(cb.num_seeds)
        self.log_dict({"train/loss": total, "train/mlm": loss, "train/aux": aux},
                      prog_bar=True, batch_size=bs)
        self.log_dict({f"train/{k}": v for k, v in stats.items()}, batch_size=bs)
        self.log("train/cells_per_s", self._cells / max(time.time() - self._t0, 1e-9),
                 batch_size=bs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], batch_size=bs)
        self.log("train/table", float(hash(node_type) % 1000), batch_size=bs)
        return total

    def validation_step(self, batch, batch_idx):
        node_type, cb = batch
        # Fixed seed, so val masks the SAME cells every epoch: otherwise the metric moves because the
        # question changed, not because the model did.
        loss, aux, stats = self._masked_loss(node_type, cb, step_seed=7_919 + batch_idx)
        bs = int(cb.num_seeds)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in stats.items()}, batch_size=bs, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [*self.model.parameters(), *self.head.parameters()]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay,
                                betas=self.betas)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, linear_warmup_decay(self.max_steps_budget, self.warmup_frac))
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    def on_before_optimizer_step(self, optimizer):
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)


class PretrainCheckpoint(pl.Callback):
    """Writes resumable checkpoints, with the portable trunk kept separate from the per-DB adapters.

    Layout under ``root``::

        trunk.pt            # loads onto any schema -- this is what LODO / finetune read
        adapters/<db>.pt    # pytorch-frame stype encoders, sized by that database
        train_state.pt      # optimizer, scheduler, global step (resume)
        config.json         # so a checkpoint is self-describing months later
        best_trunk.pt       # lowest val/loss seen

    Files are written to a temporary name and renamed, because a 16 h wall clock landing mid-`save`
    is exactly how a resume-capable run acquires an unreadable checkpoint.
    """

    def __init__(self, root, dataset: str, config: dict, *, every_n_steps: int = 1000,
                 keep_last: int = 2):
        self.root = Path(root)
        self.dataset = dataset
        self.config = config
        self.every_n_steps = every_n_steps
        self.keep_last = keep_last
        self.best: float | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "adapters").mkdir(exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(config, indent=2, default=str))

    @staticmethod
    def _atomic(obj, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def _write(self, trainer, pl_module, *, tag: str = "") -> None:
        trunk, adapter = split_state_dict(pl_module.state_dict())
        self._atomic(trunk, self.root / (f"{tag}trunk.pt" if tag else "trunk.pt"))
        if not tag:
            self._atomic(adapter, self.root / "adapters" / f"{self.dataset}.pt")
            scheds = [getattr(c, "scheduler", None) for c in (trainer.lr_scheduler_configs or [])]
            state = {"global_step": int(trainer.global_step),
                     "optimizer": [o.state_dict() for o in trainer.optimizers],
                     "scheduler": [s.state_dict() for s in scheds if s is not None]}
            self._atomic(state, self.root / "train_state.pt")

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.every_n_steps and trainer.global_step and \
                trainer.global_step % self.every_n_steps == 0:
            self._write(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val/loss")
        if v is None:
            return
        v = float(v)
        if math.isfinite(v) and (self.best is None or v < self.best):
            self.best = v
            self._write(trainer, pl_module, tag="best_")

    def on_train_end(self, trainer, pl_module):
        self._write(trainer, pl_module)


def load_trunk(model, head, path, *, strict: bool = False) -> dict:
    """Load a portable trunk into a freshly built model+head; report what did and did not transfer.

    Returns a dict of counts rather than raising on a miss, because the *expected* use is loading a
    trunk onto a **different database**, where the adapter keys are legitimately absent. Silence
    would be the wrong behaviour in both directions — so the caller gets the numbers and logs them.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    own = {**{f"model.{k}": v for k, v in model.state_dict().items()},
           **{f"head.{k}": v for k, v in head.state_dict().items()}}
    ok, shape_mismatch, unknown = {}, [], []
    for k, v in state.items():
        if k not in own:
            unknown.append(k)
        elif own[k].shape != v.shape:
            shape_mismatch.append((k, tuple(own[k].shape), tuple(v.shape)))
        else:
            ok[k] = v
    missing = [k for k in own if k not in ok]
    # Split the misses. A trunk file NEVER contains adapter keys, so those are missing by design and
    # mean "this database needs its own encoders" — the expected state on a LODO target. A *portable*
    # key going missing is the actual failure, and conflating the two makes the report useless
    # precisely when it matters.
    missing_adapter = [k for k in missing if is_adapter_key(k)]
    missing_portable = [k for k in missing if not is_adapter_key(k)]
    if strict and (shape_mismatch or unknown or missing_portable):
        raise RuntimeError(f"strict trunk load failed: {len(missing_portable)} portable keys "
                           f"missing, {len(shape_mismatch)} shape-mismatched, {len(unknown)} unknown")
    for name, mod in {"model": model, "head": head}.items():
        sub = {k[len(name) + 1:]: v for k, v in ok.items() if k.startswith(name + ".")}
        mod.load_state_dict(sub, strict=False)
    return {"loaded": len(ok), "missing_adapter": len(missing_adapter),
            "missing_portable": missing_portable, "shape_mismatch": shape_mismatch,
            "unknown": len(unknown)}
