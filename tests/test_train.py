"""Phase 4 — training: a step decreases loss and the model can overfit a single batch (~0 loss)."""
from __future__ import annotations

import warnings

import torch

from tests.conftest import rel_f1_available

D_TEXT, D_MODEL = 32, 32


def _module_and_batch():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.train.finetune import docs_for_regime
    from gloss.train.loop import HALOSLitModule

    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    g, doc_mp = docs_for_regime(bundle, "rel-f1", "full", encoder="hash", d_text=D_TEXT)
    module = HALOSLitModule(
        bundle, g, doc_mp, task.entity_table,
        model_kwargs=dict(d_model=D_MODEL, n_heads=4, n_layers=2, d_text=D_TEXT, n_freq=8),
    )
    loader = make_loader(bundle, task, "train", num_neighbors=[8, 8], batch_size=128, shuffle=False)
    raw = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    gb = module.transfer_batch_to_device(raw, device)
    return module, gb


@rel_f1_available
def test_overfits_single_batch():
    from gloss.train.losses import masked_bce

    module, gb = _module_and_batch()
    module.train()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = masked_bce(module(gb), gb.target, gb.has_target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "loss did not decrease"
    assert losses[-1] < 0.1, f"failed to overfit (final loss {losses[-1]:.3f})"


@rel_f1_available
def test_val_metrics_finite():
    from gloss.eval.metrics import binary_metrics

    module, gb = _module_and_batch()
    module.eval()
    with torch.no_grad():
        logits = module(gb)
    m = binary_metrics(logits[gb.has_target], gb.target[gb.has_target])
    assert torch.isfinite(torch.tensor(m["logloss"]))
    assert 0.0 <= m["auroc"] <= 1.0
