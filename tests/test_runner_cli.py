"""Every CLI knob the array runners parse must actually reach the function that runs the job.

This exists because of `amendments.md` §9.3: `run_gridsearch.py` parsed `--arch`, `--phase` and
`--encoder` and then called `run_index(...)` **without them**, so every task fell back to
`arch="rt", encoder="qwen"` regardless of the command line — while `--list` in the same `main()` DID
honour `args.arch`. The array was therefore sized from the two-level grid (72) but every task ran the
864-entry RT grid, and a `--encoder harrier` array quietly ran qwen. It cost 96 completed GPU-jobs of
the wrong experiment, twice, before a result JSON was read back.

A dropped kwarg is invisible at runtime — the job succeeds, it just answers a different question — so
it needs a test rather than a code review. Assert on the *whole* forwarded set, not one flag, so the
next knob added to the parser is covered by the same failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(mod_name: str):
    """Import a runner script without executing a job (its heavy imports are function-local)."""
    return pytest.importorskip(mod_name)


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--index", "0", "--arch", "two_level", "--phase", "full", "--encoder", "qwen"],
         {"arch": "two_level", "phase": "full", "encoder": "qwen"}),
        (["--index", "5", "--arch", "two_level", "--phase", "phase0a", "--encoder", "harrier"],
         {"arch": "two_level", "phase": "phase0a", "encoder": "harrier"}),
        (["--index", "5", "--encoder", "harrier"],            # the flag that silently ran qwen
         {"arch": "rt", "phase": "full", "encoder": "harrier"}),
    ],
)
def test_gridsearch_main_forwards_arch_phase_encoder(monkeypatch, tmp_path, argv, expected):
    rg = _load("run_gridsearch")
    seen = {}

    def fake_run_index(index, **kw):
        seen["index"] = index
        seen.update(kw)
        return {"config_idx": index, **kw}

    monkeypatch.setattr(rg, "run_index", fake_run_index)
    monkeypatch.setattr(sys, "argv", ["run_gridsearch.py", *argv, "--out-dir", str(tmp_path)])
    assert rg.main() == 0

    for k, v in expected.items():
        assert seen.get(k) == v, f"--{k} was parsed but not forwarded to run_index (got {seen.get(k)!r})"


def test_gridsearch_forwards_every_run_index_knob(monkeypatch, tmp_path):
    """`--list` and the run path must not disagree about the grid, and no parsed knob may be dropped.

    The §9.3 bug was exactly a disagreement: `--list` read `args.arch`, the run path did not.
    """
    rg = _load("run_gridsearch")
    seen = {}
    monkeypatch.setattr(rg, "run_index", lambda index, **kw: (seen.update(kw), {})[1])
    monkeypatch.setattr(sys, "argv", [
        "run_gridsearch.py", "--index", "0", "--arch", "two_level", "--phase", "full",
        "--encoder", "harrier", "--seeds", "2", "--epochs", "7", "--num-workers", "3",
        "--seq-len", "256", "--max-fk", "4", "--tasks", "regression", "--reg-loss", "l1",
        "--bin-loss", "auc", "--out-dir", str(tmp_path),
    ])
    assert rg.main() == 0
    assert seen == {
        "seeds": 2, "epochs": 7, "num_workers": 3, "seq_len": 256, "max_fk": 4,
        "out_dir": tmp_path, "arch": "two_level", "phase": "full", "encoder": "harrier",
        "task_set": "regression", "regression_loss": "l1", "binary_loss": "auc",
    }
    # the grid `--list` reports must be the grid the run path indexes
    assert len(rg.jobs(2, "two_level")) == len(rg.two_level_grid()) * len(rg.TASKS) * 2


def test_task_set_changes_the_index_mapping_consistently(monkeypatch, tmp_path):
    """`--tasks` re-maps index->job, so `--list` and the run path must agree on it.

    This is the same failure shape as §9.3: a knob that `--list` honours and the run path ignores
    sizes the array from one grid while the jobs index another. Here it would silently run binary
    tasks under an `--reg-loss l1` array and report them as the regression sweep.
    """
    rg = _load("run_gridsearch")
    reg = rg.TASK_SETS["regression"]
    assert len(reg) == 4 and set(reg) <= set(rg.TASKS)
    assert set(reg) & set(rg.TASK_SETS["binary"]) == set(), "regression/binary subsets must partition"
    assert len(rg.TASK_SETS["binary"]) + len(reg) == len(rg.TASKS)

    assert len(rg.jobs(1, "two_level", "regression")) == len(rg.two_level_grid()) * 4
    # index 0 must denote a REGRESSION job under this task set, not TASKS[0] (a binary task)
    _ci, _cfg, ds, tk, _s = rg.jobs(1, "two_level", "regression")[0]
    assert (ds, tk) in reg

    seen = {}
    monkeypatch.setattr(rg, "run_index", lambda index, **kw: (seen.update(kw), {})[1])
    monkeypatch.setattr(sys, "argv", ["run_gridsearch.py", "--list", "--arch", "two_level",
                                      "--tasks", "regression", "--seeds", "1",
                                      "--out-dir", str(tmp_path)])
    assert rg.main() == 0          # --list must not blow up on the subset

    with pytest.raises(ValueError, match="unknown task_set"):
        rg.jobs(1, "two_level", "nonsense")


def test_ablation_main_forwards_arch_and_phase(monkeypatch, tmp_path):
    """`run_ablation.py` got this right — pin it so it stays right (the headline arrays depend on it)."""
    ra = _load("run_ablation")
    seen = {}
    monkeypatch.setattr(ra.ablation, "run_config", lambda index, **kw: (seen.update(kw), {})[1])
    monkeypatch.setattr(sys, "argv", [
        "run_ablation.py", "--index", "0", "--arch", "two_level", "--phase", "full",
        "--encoder", "qwen", "--seeds", "3", "--datasets", "rel-f1",
        "--signals", "signature", "--out-dir", str(tmp_path),
    ])
    assert ra.main() == 0
    assert seen.get("arch") == "two_level"
    assert seen.get("encoder") == "qwen"
    assert seen.get("two_level"), "--phase full must resolve to a non-empty two_level switch dict"


def test_ablation_record_is_self_describing():
    """A finished result must state the config it ran, not just its scores.

    `run_config` used to record neither `encoder` nor the model shape, so `results/two_level_full/`
    can only be shown to be qwen by inference from the submit line, not by record. Trusting the
    submit line is exactly how two grid arrays finished on the wrong architecture AND the wrong
    encoder (amendments.md §9.3).
    """
    from gloss.eval.ablation import FINGERPRINT_KEYS, run_fingerprint

    fp = run_fingerprint(
        encoder="harrier", d_text=5376,
        model_kwargs={"d_model": 256, "n_blocks": 8, "n_heads": 8, "d_ff": 1024,
                      "enc_channels": 256, "arch": "two_level"},
        num_experts=4, k=2, lr=3e-4, max_epochs=10, batch_size=64, seq_len=512,
    )
    assert set(fp) == set(FINGERPRINT_KEYS)
    assert fp["encoder"] == "harrier" and fp["d_text"] == 5376
    assert fp["d_model"] == 256 and fp["n_blocks"] == 8 and fp["d_ff"] == 1024
    assert fp["lr"] == 3e-4 and fp["max_epochs"] == 10 and fp["seq_len"] == 512
    # a missing model kwarg must read as null, never silently vanish from the record
    sparse = run_fingerprint(encoder="qwen", d_text=2560, model_kwargs={}, num_experts=4, k=2,
                             lr=1e-3, max_epochs=5, batch_size=8, seq_len=128)
    assert set(sparse) == set(FINGERPRINT_KEYS)
    assert sparse["d_model"] is None


def test_ablation_run_config_merges_the_fingerprint():
    """The fingerprint must actually be merged into the record, not just exist as a helper."""
    import inspect

    from gloss.eval import ablation

    src = inspect.getsource(ablation.run_config)
    assert "run_fingerprint(" in src, "run_config no longer stamps its result with run_fingerprint()"


# --- the MultiEmbeddingTensor offset repair (amendments.md §9.2) -----------------------------------

def _met(values, offset):
    import torch
    from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor

    import gloss.data.graph  # noqa: F401  — installs the patch at import

    return MultiEmbeddingTensor(num_rows=values.shape[0], num_cols=max(offset.numel() - 1, 0),
                                values=values, offset=offset)


def test_met_zero_column_offset_is_rebased():
    """A ZERO-column MET with a non-zero offset base must be repaired, not crash.

    `validate()` requires `len(offset) == num_cols + 1`, so a 1-element offset means num_cols == 0 —
    no column embedding is addressable, and rebasing to [0] cannot lose data. This case fell through
    the old `numel() >= 2` guard into torch_frame's bare `assert self.offset[0] == 0` and killed
    29030571_{15,25,71}, three rel-event grid tasks on three different configs.
    """
    import torch

    met = _met(torch.zeros(4, 0), torch.tensor([7]))     # would previously raise AssertionError
    assert int(met.offset[0]) == 0 and met.offset.numel() == 1
    assert met.num_cols == 0


def test_met_repair_is_a_noop_on_healthy_tensors():
    """The patch must never touch a batch that already validates — it can only turn a crash into the
    correct result, never change a correct result."""
    import torch

    values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    met = _met(values.clone(), torch.tensor([0, 3, 6]))
    assert torch.equal(met.offset, torch.tensor([0, 3, 6]))
    assert torch.equal(met.values, values)


def test_met_unrecognised_layout_raises_a_diagnostic_not_a_bare_assert():
    """The un-normalisable branch must report the layout. Worker stdout is discarded, so a print
    could never surface it — the message has to ride on the exception."""
    import re

    import pytest
    import torch

    with pytest.raises(RuntimeError, match=r"could not be rebased") as exc:
        _met(torch.zeros(2, 5), torch.tensor([3, 6, 99]))   # w=5 matches neither T nor k+T
    msg = str(exc.value)
    assert re.search(r"n_cols=2", msg) and re.search(r"k=3", msg) and "col_dims=" in msg
    assert "--num-workers 0" in msg
