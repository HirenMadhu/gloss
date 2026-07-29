"""The **frozen** fixture + fingerprint logic behind the §6 bit-for-bit parity guard (changes.md §9.7).

Why this file exists at all, and why it does *not* import ``tests/conftest.py``
------------------------------------------------------------------------------
``changes.md §6`` demands: *"A ``arch: rt`` run reproduces the pre-change numbers bit-for-bit given the
same seed. This is the regression guard for the whole refactor."*  ``§9.7`` then flags the ordering
constraint that makes this module urgent: **P0.5** pins the pytorch-frame stype id space to a fixed
enum, which changes ``RelationalSignature.stype_emb.num_embeddings``, which changes the *number of RNG
draws* module construction consumes, which changes **every parameter initialised after it**. Once P0.5
lands, a "before" baseline can no longer be captured. So it is captured here, first.

A bit-for-bit guard must **own its input**. The shared fixtures in ``tests/conftest.py`` are a moving
target (they are being extended right now with extra child tables / FK roles for P0.1), and every such
edit would silently invalidate this baseline and then re-surface later as a phantom "parity regression"
that is really only fixture drift. Therefore the schema, the row values, the sampled minibatch and the
name table below are **declared inline and deliberately frozen**, and nothing here imports
``conftest``. Changing anything in this file is equivalent to invalidating the baseline, and must be
followed by a deliberate re-capture (``scripts/capture_parity_baseline.py``) in the same commit.

What is fingerprinted (three independent tiers, so failures are diagnosable)
---------------------------------------------------------------------------
1. ``init``    — batch-independent. Parameter creation order, per-parameter shape/dtype/byte-hash, and
                 the CPU RNG state *after* construction. **This is the tier P0.5 perturbs.**
2. ``input``   — the ``CellBatch`` the collate produces from the frozen graph. Isolates fixture/collate
                 drift from model drift, so tier 3 failures are never misread.
3. ``numerics``— an eval-mode forward (logits, aux), a backward (per-parameter grad hashes) and three
                 Adam steps (exact loss sequence + resulting parameter hashes) on the frozen batch.

Hermetic: synthetic tables only, ``HashEncoder`` name table (no schema cache, no Qwen, no download),
CPU, single-threaded. Routing arm is ``signature``; the ``dense``/``dense_wide`` arms are never run
(standing project rule).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

# ---------------------------------------------------------------------------------------------
# Frozen constants. Touching ANY of these invalidates the baseline — re-capture deliberately.
# ---------------------------------------------------------------------------------------------
FIXTURE_VERSION = "parity-fixture-v1"
SEED = 0
D_TEXT = 16                      # HashEncoder width for the frozen name table
SEQ_LEN = 16
MAX_FK = 2
SEED_TIME = 100.0

ENTITY = "customer"
NODE_TYPES = ["customer", "txn"]
EDGE_TYPES = [
    ("txn", "f2p_buyer", "customer"),
    ("customer", "rev_f2p_buyer", "txn"),
    ("txn", "f2p_seller", "customer"),
    ("customer", "rev_f2p_seller", "txn"),
]

# Frozen row values -> frozen pytorch-frame col_stats -> frozen value encoders.
CUSTOMER_ROWS = {"c_num": [1.0, 2.0], "c_cat": ["alpha", "beta"]}
TXN_ROWS = {"t_num": [0.5, 1.5, 2.5, 3.5], "t_cat": ["x", "y", "x", "y"]}
TXN_TIMES = [10.0, 20.0, 30.0, 40.0]
TARGETS = [1.0, 0.0]

# The model under fingerprint: current `arch: rt` MoRE, tiny (this runs on CPU inside pytest).
MODEL_CFG = dict(
    d_model=32,
    d_sig=16,
    n_blocks=2,
    n_heads=4,
    d_ff=64,
    enc_channels=32,
    out_dim=1,
    route_on="signature",
    num_experts=4,
    k=2,
    moe_placement="all",
)
N_TRAIN_STEPS = 3
LR = 1e-3

ARTIFACT_PATH = Path(__file__).resolve().parent / "parity_baseline.json"


# ---------------------------------------------------------------------------------------------
# Deterministic hashing helpers
# ---------------------------------------------------------------------------------------------
def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def tensor_hash(t: torch.Tensor) -> str:
    """Byte-exact hash of a tensor's contents (no tolerance — that is the point)."""
    t = t.detach().cpu().contiguous()
    return _h(t.numpy().tobytes())


def tensor_sig(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "hash": tensor_hash(t)}


def fhex(x) -> str:
    """Exact, human-diffable float (round-trips bit-for-bit)."""
    return float(x).hex()


# ---------------------------------------------------------------------------------------------
# The frozen synthetic bundle + minibatch (self-contained; no conftest, no relbench)
# ---------------------------------------------------------------------------------------------
def _materialize(rows: dict) -> tuple[object, dict]:
    """A frozen table -> (TensorFrame, col_stats) via pytorch-frame's own stats path."""
    import pandas as pd
    import torch_frame

    df = pd.DataFrame(rows)
    stype = {
        c: (torch_frame.numerical if pd.api.types.is_float_dtype(df[c]) else torch_frame.categorical)
        for c in df.columns
    }
    ds = torch_frame.data.Dataset(df, col_to_stype=stype)
    ds.materialize()
    return ds.tensor_frame, ds.col_stats


def frozen_bundle():
    """The frozen dual-FK ``GraphBundle``.

    The *data* (schema, row values, stats) is frozen here; the FK **role vocabulary** is delegated to
    ``graph._build_vocabs`` on purpose. The role vocabulary is a library concern that P0.1 is actively
    reshaping (column key -> ``(child, column, parent)`` triple), and a stale hand-rolled copy would
    silently desync from the row-graph builder that consumes it. Delegating is safe for a bit-for-bit
    guard because ``fk_role_id`` / ``metapath_id`` reach **no weight shape and no fingerprinted field**
    — nothing below hashes them, and §0's no-dataset-artifact rule forbids ``K`` from entering any
    parameter. If that ever stops being true, this call must be frozen too.
    """
    from torch_geometric.data import HeteroData

    from gloss.data.graph import GraphBundle, _build_vocabs

    cust_tf, cust_stats = _materialize(CUSTOMER_ROWS)
    txn_tf, txn_stats = _materialize(TXN_ROWS)

    data = HeteroData()
    data["customer"].tf = cust_tf
    data["txn"].tf = txn_tf

    node_type_id, fk_role_id, metapath_id = _build_vocabs(list(NODE_TYPES), list(EDGE_TYPES))

    bundle = GraphBundle(
        dataset_name="parity-frozen",
        data=data,
        col_stats_dict={"customer": cust_stats, "txn": txn_stats},
        node_types=list(NODE_TYPES),
        edge_types=list(EDGE_TYPES),
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )
    return bundle, cust_tf, txn_tf


def frozen_minibatch(cust_tf, txn_tf):
    """A 2-seed disjoint sampled ``HeteroData``: 1 untimed customer seed + 2 timed txns per seed."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["customer"].num_nodes = 2
    d["customer"].batch = torch.tensor([0, 1])
    d["customer"].seed_time = torch.tensor([SEED_TIME, SEED_TIME], dtype=torch.float64)
    d["customer"].n_id = torch.tensor([0, 1])
    d["customer"].y = torch.tensor(TARGETS, dtype=torch.float32)
    d["customer"].tf = cust_tf

    d["txn"].num_nodes = 4
    d["txn"].batch = torch.tensor([0, 0, 1, 1])
    d["txn"].time = torch.tensor(TXN_TIMES, dtype=torch.float64)
    d["txn"].n_id = torch.tensor([10, 11, 12, 13])
    d["txn"].tf = txn_tf

    # child (txn) -> parent (customer). seg 0: txns 0,1 -> cust 0 ; seg 1: txns 2,3 -> cust 1
    d["txn", "f2p_buyer", "customer"].edge_index = torch.tensor([[0, 2], [0, 1]])
    d["txn", "f2p_seller", "customer"].edge_index = torch.tensor([[1, 3], [0, 1]])
    d["customer", "rev_f2p_buyer", "txn"].edge_index = torch.tensor([[0, 1], [0, 2]])
    d["customer", "rev_f2p_seller", "txn"].edge_index = torch.tensor([[0, 1], [1, 3]])
    return d


def frozen_name_table(bundle) -> torch.Tensor:
    """Frozen ``[C, d_text]`` column-name table from ``HashEncoder`` (sha256-seeded, no model, no cache)."""
    from gloss.text.cache import HashEncoder
    from gloss.text.schema import build_column_name_embeddings

    return build_column_name_embeddings(bundle, HashEncoder(dim=D_TEXT))


def frozen_cell_batch(bundle, batch):
    from gloss.data.collate import to_cell_batch

    return to_cell_batch(batch, bundle, ENTITY, seq_len=SEQ_LEN, max_fk=MAX_FK)


CELL_BATCH_FIELDS = (
    "node_idxs", "col_idxs", "table_idxs", "is_padding", "is_seed_cell",
    "row_time", "is_timed", "n_id", "f2p_nbr_idxs", "seed_time", "target", "has_target",
)


# ---------------------------------------------------------------------------------------------
# The three fingerprint tiers
# ---------------------------------------------------------------------------------------------
def _prepare() -> None:
    """Pin everything that could make CPU float math or RNG non-reproducible."""
    from gloss.utils.seeding import seed_everything

    torch.set_num_threads(1)          # reduction order (and CPU politeness) must not vary
    seed_everything(SEED, deterministic_torch=True)


def build_fixture():
    """-> (bundle, name_emb, cell_batch). Deterministic; consumes no global RNG for the model."""
    bundle, cust_tf, txn_tf = frozen_bundle()
    batch = frozen_minibatch(cust_tf, txn_tf)
    name_emb = frozen_name_table(bundle)
    cb = frozen_cell_batch(bundle, batch)
    return bundle, name_emb, cb


def input_fingerprint(bundle, name_emb, cb) -> dict:
    """Tier 2 — the model's *inputs*. Drift here means the fixture or the collate moved, not the model."""
    from gloss.data.collate import column_vocab
    from gloss.text.schema import build_column_modality_ids, column_name_strings

    modality_id, n_stypes = build_column_modality_ids(bundle)
    vocab = column_vocab(bundle)
    return {
        "fixture_version": FIXTURE_VERSION,
        "column_vocab": {f"{nt}.{c}": i for (nt, c), i in sorted(vocab.items(), key=lambda kv: kv[1])},
        "column_name_texts": column_name_strings(bundle),
        "name_emb": tensor_sig(name_emb),
        "modality_id": tensor_sig(modality_id),
        "n_stypes": int(n_stypes),
        "cell_batch": {
            "num_seeds": int(cb.num_seeds),
            "seq_len": int(cb.seq_len),
            "max_fk": int(cb.max_fk),
            "n_real_cells": int((~cb.is_padding).sum()),
            "fields": {f: tensor_sig(getattr(cb, f)) for f in CELL_BATCH_FIELDS},
        },
    }


def build_model(bundle, name_emb):
    """Seed, construct, and report what the construction did to the RNG (tier 1's payload)."""
    from gloss.model.more import MoRE

    _prepare()
    rng_before = torch.random.get_rng_state()
    model = MoRE(bundle, name_emb, **MODEL_CFG)
    rng_after = torch.random.get_rng_state()
    # A probe drawn from the *post-init* stream: shifts if init consumed a different number of draws.
    probe = torch.randn(8)
    return model, rng_before, rng_after, probe


def init_fingerprint(model, rng_before, rng_after, probe) -> dict:
    """Tier 1 — batch-independent. **This is what P0.5 (pinning the stype enum) perturbs.**"""
    sd = model.state_dict()
    params = dict(model.named_parameters())
    stype_emb = getattr(getattr(model, "signature", None), "stype_emb", None)
    return {
        "config": dict(MODEL_CFG),
        "seed": SEED,
        "param_order": list(params.keys()),
        "state_dict_order": list(sd.keys()),
        "n_params": int(sum(p.numel() for p in params.values())),
        "params": {k: tensor_sig(v) for k, v in sd.items()},
        # The §9.7 tripwire, called out by name so a failure is self-explaining.
        "stype_emb_num_embeddings": None if stype_emb is None else int(stype_emb.num_embeddings),
        "rng": {
            "state_before_init": _h(rng_before.numpy().tobytes()),
            "state_after_init": _h(rng_after.numpy().tobytes()),
            "post_init_probe": [fhex(v) for v in probe.tolist()],
        },
    }


def numerics_fingerprint(model, cb) -> dict:
    """Tier 3 — forward logits, aux, gradients, and an exact 3-step Adam loss sequence."""
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        logits, aux = model(cb)
    fwd = {
        "logits": tensor_sig(logits),
        "logits_exact": [fhex(v) for v in logits.reshape(-1).tolist()],
        "aux": fhex(aux),
    }

    model.zero_grad(set_to_none=True)
    logits, aux = model(cb)
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), cb.target) + aux
    loss.backward()
    grads = {k: tensor_sig(p.grad) for k, p in model.named_parameters() if p.grad is not None}

    # Three optimizer steps: the closest cheap proxy for "a run reproduces bit-for-bit".
    model.zero_grad(set_to_none=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(N_TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        lg, ax = model(cb)
        l = F.binary_cross_entropy_with_logits(lg.squeeze(-1), cb.target) + ax
        l.backward()
        opt.step()
        losses.append(fhex(l.detach()))

    return {
        "forward": fwd,
        "backward_loss": fhex(loss.detach()),
        "grads": grads,
        "train_losses": losses,
        "params_after_train": {k: tensor_sig(v) for k, v in model.state_dict().items()},
    }


def compute_fingerprint() -> dict:
    """The whole fingerprint, in one deterministic pass."""
    _prepare()
    bundle, name_emb, cb = build_fixture()
    inp = input_fingerprint(bundle, name_emb, cb)
    model, rng_b, rng_a, probe = build_model(bundle, name_emb)
    init = init_fingerprint(model, rng_b, rng_a, probe)
    num = numerics_fingerprint(model, cb)
    return {"input": inp, "init": init, "numerics": num}


def env_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    import torch_frame

    return {
        "git_commit": commit,
        "seed": SEED,
        "fixture_version": FIXTURE_VERSION,
        "torch": torch.__version__,
        "torch_frame": torch_frame.__version__,
    }


def load_baseline(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else ARTIFACT_PATH
    with open(p) as f:
        return json.load(f)
