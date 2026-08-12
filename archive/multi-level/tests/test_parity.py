"""The §6 bit-for-bit parity guard — the regression guard for the whole two-level refactor.

``changes.md §6`` ("Parity"): *"A ``arch: rt`` run reproduces the pre-change numbers bit-for-bit given
the same seed. This is the regression guard for the whole refactor."*

``changes.md §9.7`` ("Parity vs P0.5 — ordering constraint"): pinning the stype enum changes
``stype_emb.num_embeddings``, which changes init RNG draw order, which breaks this guard — so the
baseline had to be captured **before** P0.5 lands. It was: ``tests/fixtures/parity_baseline.json``.

**This test is supposed to be brittle.** Every assertion below is exact, with no tolerance, because a
guard with a tolerance cannot see an init-order change. When it fails, read the failure message: it
tells you *which tier* moved (inputs / init / numerics) and therefore whether you broke something or
legitimately changed init order.

The fixture is frozen and **self-contained** (``tests/fixtures/parity_fixture.py``) — it deliberately
does not import ``tests/conftest.py``, so that unrelated edits to the shared synthetic fixtures cannot
masquerade as parity regressions here.
"""
from __future__ import annotations

import pytest

from .fixtures.parity_fixture import (
    ARTIFACT_PATH,
    CELL_BATCH_FIELDS,
    build_fixture,
    build_model,
    compute_fingerprint,
    init_fingerprint,
    input_fingerprint,
    load_baseline,
    numerics_fingerprint,
)

# ------------------------------------------------------------------------------------------------
# The message every failure carries. Its whole job is to stop someone "fixing" a real regression by
# silently re-capturing, and to stop someone debugging for an hour when the change was intended.
# ------------------------------------------------------------------------------------------------
_WHAT_NOW = """
------------------------------------------------------------------------------------------------
PARITY BASELINE MISMATCH — changes.md §6 (Parity) / §9.7 (Parity vs P0.5, ordering constraint)

The current `arch: rt` model no longer reproduces the captured pre-refactor baseline bit-for-bit.
Baseline: {artifact}

Decide WHICH of these happened before touching anything:

 (a) UNINTENDED REGRESSION. Something changed the model, its init, or its numerics by accident.
     -> Fix the code. Do NOT re-capture the baseline.

 (b) INTENDED init-order change. The canonical case is **P0.5** (pinning the pytorch-frame stype id
     space to a fixed enum). That resizes `RelationalSignature.stype_emb`, which changes how many RNG
     draws construction consumes, which changes every parameter initialised after it. §9.7 predicted
     exactly this. Every value below will differ, and that is correct.
     -> Re-capture DELIBERATELY, in the same commit as the change, saying why:
            .venv/bin/python scripts/capture_parity_baseline.py --force
     Note that re-capturing retires the "before" reference permanently: it can never be recovered
     once the pre-change code is gone. That is why §9.7 ordered this capture first.

 (c) FIXTURE / COLLATE DRIFT. If `test_input_fixture_is_unchanged` is also failing, the model's
     *inputs* moved, and every numeric mismatch downstream is a consequence, not a separate bug.
     Fix or explain the input drift first; the rest will follow.

Diff the whole fingerprint at once with:
    .venv/bin/python scripts/capture_parity_baseline.py --check
------------------------------------------------------------------------------------------------"""


def _explain(what: str) -> str:
    return f"{what}\n{_WHAT_NOW.format(artifact=ARTIFACT_PATH)}"


# ------------------------------------------------------------------------------------------------
# Fixtures — computed once per session (the model is tiny and CPU-only by construction).
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.fail(
            f"no parity baseline at {ARTIFACT_PATH}. changes.md §9.7 requires it to be captured "
            f"BEFORE P0.5 lands:\n    .venv/bin/python scripts/capture_parity_baseline.py"
        )
    return load_baseline()["fingerprint"]


@pytest.fixture(scope="module")
def current() -> dict:
    try:
        return compute_fingerprint()
    except TypeError as e:                                   # e.g. CellBatch gained required fields
        pytest.fail(
            f"could not even build the frozen parity fixture: {e!r}\n"
            "This is a broken input path, not a parity result — the collate/CellBatch contract "
            "changed under the fixture. Fix that first, then re-run this test."
        )


def _assert_same(name: str, base, cur, what: str):
    assert cur == base, _explain(f"{what}\n  field   : {name}\n  baseline: {base!r}\n  current : {cur!r}")


# ------------------------------------------------------------------------------------------------
# Tier 0 — the artifact itself
# ------------------------------------------------------------------------------------------------
def test_baseline_artifact_is_wellformed():
    """The committed baseline records what it is, when, from which commit, at which seed."""
    payload = load_baseline()
    for key in ("env", "fingerprint", "captured_utc", "_doc"):
        assert key in payload, f"parity baseline is missing {key!r}"
    env = payload["env"]
    assert env["git_commit"] != "unknown", "baseline must record the commit it was captured at (§9.7)"
    assert env["seed"] == 0
    fp = payload["fingerprint"]
    assert set(fp) == {"input", "init", "numerics"}
    assert fp["init"]["config"]["route_on"] == "signature", "the baseline arm must be `signature`"
    assert fp["init"]["config"]["route_on"] not in ("dense", "dense_wide"), "never run the dense arms"


# ------------------------------------------------------------------------------------------------
# Tier 2 — inputs. Checked FIRST so input drift is never misread as model drift.
# ------------------------------------------------------------------------------------------------
def test_input_fixture_is_unchanged(baseline, current):
    """The frozen fixture + the collate still produce byte-identical model inputs.

    A failure here means the *fixture or the collate* moved, not the model. Everything downstream is
    then a consequence. This tier is why the fixture is self-contained: an edit to
    ``tests/conftest.py`` must never be able to reach it.
    """
    b, c = baseline["input"], current["input"]
    _assert_same("fixture_version", b["fixture_version"], c["fixture_version"],
                 "The frozen parity fixture itself was edited.")
    _assert_same("column_vocab", b["column_vocab"], c["column_vocab"],
                 "The global column vocabulary changed.")
    _assert_same("column_name_texts", b["column_name_texts"], c["column_name_texts"],
                 "The column-name text template changed (gloss/text/schema.py).")
    _assert_same("name_emb", b["name_emb"], c["name_emb"],
                 "The frozen HashEncoder name table changed.")
    for f in CELL_BATCH_FIELDS:
        _assert_same(f"cell_batch.{f}", b["cell_batch"]["fields"][f], c["cell_batch"]["fields"][f],
                     "The collate now produces a different CellBatch from the same frozen graph.")
    for k in ("num_seeds", "seq_len", "max_fk", "n_real_cells"):
        _assert_same(f"cell_batch.{k}", b["cell_batch"][k], c["cell_batch"][k],
                     "The CellBatch geometry changed.")


# ------------------------------------------------------------------------------------------------
# Tier 1 — init. THIS is the tier §9.7 is about.
# ------------------------------------------------------------------------------------------------
def test_stype_id_space_matches_baseline(baseline, current):
    """The §9.7 tripwire, isolated so it names itself.

    **P0.5 has landed.** ``build_column_modality_ids`` now pins the id space to the fixed
    ``schema.STYPE_ORDER`` enum, so ``n_stypes`` is the constant ``N_STYPES = 10`` rather than
    however many stypes a bundle happened to contain. This assertion did exactly its job when that
    change went in — it tripped first, with ``2 -> 10``, ahead of the 6 downstream bitwise failures —
    and the baseline was then re-captured deliberately.

    It stays as a guard: from here on, a change in this number means something resized the modality
    embedding *again*, which is a real bug unless someone intentionally edited ``STYPE_ORDER``
    (appending is safe; reordering silently invalidates every checkpoint).

    That the id space is genuinely pinned — same ids across datasets, constant width — is asserted
    separately in ``tests/test_schema_pinning.py``; this test only compares against the fingerprint.
    """
    b = baseline["init"]["stype_emb_num_embeddings"]
    c = current["init"]["stype_emb_num_embeddings"]
    assert c == b, _explain(
        f"stype_emb.num_embeddings changed {b} -> {c}.\n"
        "This is the exact quantity changes.md §9.7 predicted would break parity: the stype id space "
        "is no longer sized the way the baseline was captured with.\n"
        "If you just landed P0.5 (pinning the stype enum to a fixed, bundle-independent order) this "
        "is EXPECTED and correct — re-capture deliberately. If you did not, something else resized "
        "the modality embedding and you have a real bug."
    )


def test_param_shapes_and_creation_order(baseline, current):
    """Parameter creation order and shapes — the direct signature of module construction order."""
    b, c = baseline["init"], current["init"]
    _assert_same("config", b["config"], c["config"], "The fingerprinted model config changed.")
    _assert_same("param_order", b["param_order"], c["param_order"],
                 "Parameters are created in a different order (modules added/removed/reordered).")
    _assert_same("state_dict_order", b["state_dict_order"], c["state_dict_order"],
                 "state_dict key order changed.")
    _assert_same("n_params", b["n_params"], c["n_params"], "Total parameter count changed.")
    for k in b["params"]:
        _assert_same(f"params[{k}].shape", b["params"][k]["shape"], current["init"]["params"][k]["shape"],
                     "A parameter changed shape.")
        _assert_same(f"params[{k}].dtype", b["params"][k]["dtype"], current["init"]["params"][k]["dtype"],
                     "A parameter changed dtype.")


def test_init_rng_draw_order(baseline, current):
    """The RNG draw-order signature: how many draws construction consumed, and from where.

    ``state_after_init`` and ``post_init_probe`` are the sharpest available probe of init order — they
    move if *any* module consumes a different number of random draws, even when every shape is
    unchanged (e.g. a re-ordered ``nn.Module`` attribute assignment).
    """
    b, c = baseline["init"]["rng"], current["init"]["rng"]
    _assert_same("rng.state_before_init", b["state_before_init"], c["state_before_init"],
                 "The global seeding mechanism itself changed (gloss/utils/seeding.py).")
    _assert_same("rng.state_after_init", b["state_after_init"], c["state_after_init"],
                 "Model construction consumed a different sequence of RNG draws.")
    _assert_same("rng.post_init_probe", b["post_init_probe"], c["post_init_probe"],
                 "The post-init RNG stream is offset — init draw order changed.")


def test_initial_weights_are_bitwise_identical(baseline, current):
    """Every seeded initial parameter, byte for byte. No tolerance, by design."""
    b, c = baseline["init"]["params"], current["init"]["params"]
    bad = [k for k in b if b[k]["hash"] != c.get(k, {}).get("hash")]
    assert not bad, _explain(
        f"{len(bad)}/{len(b)} initial parameters differ bit-for-bit from the baseline.\n"
        f"  first differing: {bad[:6]}\n"
        f"  e.g. {bad[0]}: baseline={b[bad[0]]['hash']} current={c.get(bad[0], {}).get('hash')}"
    )


# ------------------------------------------------------------------------------------------------
# Tier 3 — numerics
# ------------------------------------------------------------------------------------------------
def test_forward_is_bitwise_identical(baseline, current):
    """Exact logits and exact aux (router-orthogonality loss) on the frozen batch."""
    b, c = baseline["numerics"]["forward"], current["numerics"]["forward"]
    _assert_same("forward.logits", b["logits"], c["logits"], "The forward pass output changed.")
    _assert_same("forward.logits_exact", b["logits_exact"], c["logits_exact"],
                 "The forward pass output changed (exact float values).")
    _assert_same("forward.aux", b["aux"], c["aux"], "The router-orthogonality aux loss changed.")


def test_backward_is_bitwise_identical(baseline, current):
    """Exact loss and exact per-parameter gradients — catches drift in the backward path only."""
    b, c = baseline["numerics"], current["numerics"]
    _assert_same("backward_loss", b["backward_loss"], c["backward_loss"], "The training loss changed.")
    bad = [k for k in b["grads"] if b["grads"][k]["hash"] != c["grads"].get(k, {}).get("hash")]
    assert not bad, _explain(
        f"{len(bad)}/{len(b['grads'])} gradients differ bit-for-bit.\n  first differing: {bad[:6]}"
    )


def test_train_steps_are_bitwise_identical(baseline, current):
    """Three Adam steps: the exact loss sequence and the resulting weights.

    This is the cheapest honest stand-in for §6's *"a run reproduces the pre-change numbers"* — it
    exercises forward + backward + optimizer state, not just init.
    """
    b, c = baseline["numerics"], current["numerics"]
    _assert_same("train_losses", b["train_losses"], c["train_losses"],
                 "The optimization trajectory diverged from the baseline.")
    bad = [k for k in b["params_after_train"]
           if b["params_after_train"][k]["hash"] != c["params_after_train"].get(k, {}).get("hash")]
    assert not bad, _explain(
        f"{len(bad)}/{len(b['params_after_train'])} parameters differ after 3 training steps.\n"
        f"  first differing: {bad[:6]}"
    )


# ------------------------------------------------------------------------------------------------
# The guard's own precondition: the capture must be reproducible, or none of the above means anything.
# ------------------------------------------------------------------------------------------------
def test_capture_is_deterministic_within_a_process(current):
    """Two independent captures in one process agree exactly.

    If this fails, the fingerprint is noise and every other assertion in this file is worthless — so it
    is asserted rather than assumed. (The capture script is also run twice out-of-process during
    review; this covers the in-process half.)
    """
    again = compute_fingerprint()
    assert again == current, (
        "the parity fingerprint is NOT reproducible within a single process — the guard is invalid.\n"
        "Suspect: unseeded RNG in model construction, thread-count-dependent CPU reductions, or "
        "dict/set iteration order leaking into the fixture."
    )


def test_hermetic_no_relbench_or_qwen_needed():
    """The guard must run anywhere: synthetic tables, HashEncoder, CPU. No cache, no network, no GPU."""
    bundle, name_emb, cb = build_fixture()
    assert bundle.dataset_name == "parity-frozen"
    assert name_emb.shape[0] == len(bundle.node_types) * 2      # 2 feature columns per frozen table
    assert cb.num_seeds == 2 and not cb.is_padding.all()
    model, rb, ra, probe = build_model(bundle, name_emb)
    assert next(model.parameters()).device.type == "cpu"
    # and the three tiers are computable from it
    assert input_fingerprint(bundle, name_emb, cb)["fixture_version"]
    assert init_fingerprint(model, rb, ra, probe)["n_params"] > 0
    assert numerics_fingerprint(model, cb)["train_losses"]
