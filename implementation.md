# MoRE on RT — Implementation Plan

*Build guide for the design in `idea.md`. Targets the real [`snap-stanford/relational-transformer`](https://github.com/snap-stanford/relational-transformer) repo (ICLR 2026). All injection points reference actual classes in `rt/model.py`.*

---

## 0. The change in one paragraph

Replace RT's `FFN` (a SwiGLU MLP) inside each `RelationalBlock` with a **`MoEFFN`**: a pool of $M$ identical SwiGLU experts plus a top-$k$ router. The router reads a **per-cell relational signature** $z$ built once in the trunk from fields that **already exist in the batch** — `col_name_values` (frozen-LM schema embedding) and `sem_types` (modality) — plus an optional recency bin. Balance experts with a router-orthogonality loss (not a uniform aux loss), summed across blocks and added to RT's masked-cell loss. Everything else in RT is untouched.

---

## 1. Environment & data

Follow the RT README, then preprocess the **three smallest RelBench datasets** (by rows): `rel-f1` (74K, 9 tables), `rel-stack` (4.2M, 7 tables), `rel-trial` (5.4M, 15 tables, 140 cols).

```bash
git clone https://github.com/snap-stanford/relational-transformer
cd relational-transformer
pixi install
cd rustler && pixi run maturin develop --uv --release && cd ..

# download tasks/datasets
pixi run python scripts/download_relbench.py
mkdir -p ~/scratch && ln -s ~/.cache/relbench ~/scratch/relbench

# preprocess + embed the three smallest (rust sampler then text embeddings)
for db in rel-f1 rel-stack rel-trial; do
  (cd rustler && pixi run cargo run --release -- pre $db)
  pixi run python -m rt.embed $db
done
```

> Start everything on `rel-f1` — it is ~57× smaller than the next dataset, so the full smoke test + ablations run in minutes, not hours, and on a single GPU. Bring in `rel-stack`/`rel-trial` only for the transfer phase. (If you are on RelBench v2, `rel-arxiv` ≈ 222K papers is a viable lighter substitute for one of the larger two.)

**RT base config** (from the paper / `rt/main.py`): `num_blocks=12, d_model=256, num_heads=8, d_ff=1024, d_text=384`, bf16, `flex_attention`, `torch.compile`. Pretraining is masked-cell prediction; the held-out-database protocol and per-task checkpoints already exist in `scripts/`.

---

## 2. Where the MoE goes — `rt/model.py` anatomy

The relevant classes, as they exist today:

```python
class FFN(nn.Module):                  # <-- REPLACE this inside the block
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))   # SwiGLU

class RelationalBlock(nn.Module):
    def forward(self, x, block_masks):
        for l in ["col","feat","nbr","full"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])
        x = x + self.ffn(self.norms["ffn"](x))            # <-- MoE hooks here
        return x

class RelationalTransformer(nn.Module):
    # enc_dict["col_name"]: Linear(d_text, d_model)  -> the schema component s_i
    # batch["col_name_values"]: (B,S,d_text) frozen-LM column embedding
    # batch["sem_types"]: (B,S) ints in {0:number,1:text,2:datetime,3:boolean}  -> modality
    def forward(self, batch):
        ...
        x = x + self.norm_dict["col_name"](self.enc_dict["col_name"](batch["col_name_values"])) * (~is_padding)[...,None]
        ...                                               # type-value + mask embeddings
        for block in self.blocks:
            x = block(x, block_masks)                     # <-- thread z + collect aux here
        ...
        return loss_out, yhat_out
```

Three edits: (a) add a `RelationalSignature` module and a `MoEFFN`; (b) compute `z` once in `RelationalTransformer.forward` and pass it to each block; (c) thread the orthogonality loss back out and add it to `loss_out`.

---

## 3. Code

Drop these into `rt/model.py` (or a new `rt/moe.py` and import).

### 3.1 Relational signature (router input)

```python
class RelationalSignature(nn.Module):
    """Value-free per-cell routing signature z, computed ONCE in the trunk.
       z = RMSNorm( W_s * col_name_emb  +  stype_emb  +  recency_emb )."""
    def __init__(self, d_text, d_sig, n_stypes=4, use_recency=False, n_recency_bins=16):
        super().__init__()
        self.schema_proj = nn.Linear(d_text, d_sig, bias=False)   # s_i (schema/semantic)
        self.stype_emb   = nn.Embedding(n_stypes, d_sig)          # psi(sigma_i) (modality)
        self.use_recency = use_recency
        if use_recency:
            # bin 0 reserved for "no timestamp / unknown"
            self.recency_emb = nn.Embedding(n_recency_bins + 1, d_sig)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, col_name_values, sem_types, recency_bins=None):
        z = self.schema_proj(col_name_values) + self.stype_emb(sem_types)
        if self.use_recency and recency_bins is not None:
            z = z + self.recency_emb(recency_bins)
        return self.norm(z)                                       # (B, S, d_sig)
```

### 3.2 MoE FFN (drop-in for `FFN`)

```python
class MoEFFN(nn.Module):
    """Pool of SwiGLU experts; sparse top-k gate on the signature.
       Router sees route_feat (z by default); experts transform x.
       Dense expert combine = simple & correct (optimize with dispatch at scale)."""
    def __init__(self, d_model, d_ff, d_route, num_experts=8, k=2):
        super().__init__()
        self.num_experts, self.k = num_experts, k
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(num_experts)])
        self.router  = nn.Linear(d_route, num_experts, bias=False)

    def _gate(self, r):                                # r: (B,S,d_route)
        logits = self.router(r)                        # (B,S,E)
        topv, topi = logits.topk(self.k, dim=-1)
        g = torch.full_like(logits, float("-inf"))
        g.scatter_(-1, topi, topv)
        return F.softmax(g, dim=-1)                     # (B,S,E), zero off-support

    def forward(self, x, route_feat):
        g = self._gate(route_feat)
        y = x.new_zeros(x.shape)
        for e, expert in enumerate(self.experts):       # dense combine (MVP)
            y = y + g[..., e:e+1] * expert(x)
        return y, g                                     # return gates for diagnostics

    def ortho_loss(self):
        W  = F.normalize(self.router.weight, dim=-1)     # (E, d_route)
        G  = W @ W.t()                                   # (E,E)
        I  = torch.eye(self.num_experts, device=W.device)
        return ((G - I) ** 2).sum()
```

### 3.3 Block + trunk diffs

```python
class RelationalBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,
                 moe=False, d_route=None, num_experts=8, k=2, route_on="signature"):
        super().__init__()
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model)
                                    for l in ["feat","nbr","col","full","ffn"]})
        self.attns = nn.ModuleDict({l: MaskedAttention(d_model, num_heads)
                                    for l in ["feat","nbr","col","full"]})
        self.moe, self.route_on = moe, route_on
        self.ffn = (MoEFFN(d_model, d_ff, d_route, num_experts, k) if moe
                    else FFN(d_model, d_ff))

    def forward(self, x, block_masks, z=None):
        for l in ["col","feat","nbr","full"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])
        h = self.norms["ffn"](x)
        if self.moe:
            route_feat = z if self.route_on == "signature" else h   # see §5 for value/id
            y, _ = self.ffn(h, route_feat)
            return x + y, self.ffn.ortho_loss()
        return x + self.ffn(h), x.new_zeros(())
```

In `RelationalTransformer.__init__`, build the signature, pass MoE flags to blocks, store $\lambda$:

```python
self.signature = RelationalSignature(d_text, d_sig, use_recency=use_recency)
self.blocks = nn.ModuleList([
    RelationalBlock(d_model, num_heads, d_ff,
                    moe=True, d_route=d_sig, num_experts=num_experts, k=k,
                    route_on=route_on)
    for _ in range(num_blocks)
])
self.lambda_ortho = lambda_ortho
```

In `RelationalTransformer.forward`, compute `z` once (right after the token `x` is assembled) and thread it; accumulate the aux loss and fold it in just before `return`:

```python
z = self.signature(batch["col_name_values"], batch["sem_types"],
                   recency_bins=batch.get("recency_bins"))
aux = x.new_zeros(())
for block in self.blocks:
    x, ortho = block(x, block_masks, z=z)
    aux = aux + ortho
# ... existing norm_out + decode + masked loss producing loss_out ...
loss_out = loss_out + self.lambda_ortho * aux / len(self.blocks)
return loss_out, yhat_out
```

Finally, expose the new kwargs (`d_sig, num_experts, k, route_on, lambda_ortho, use_recency`) wherever `rt/main.py` constructs `RelationalTransformer`, and add them to the CLI/config so they can be swept.

---

## 4. Config / hyperparameters

| Knob | Start | Notes |
|---|---|---|
| `num_experts` $M$ | **4** (smoke) → 8 | $M \ll \#$columns (tens–140 here). Start 4 on `rel-f1` for speed/memory. |
| `k` (top-k) | **2** | $k{=}1$ is cheaper but flips experts at semantic boundaries; $k{=}2$ smooths. |
| `d_sig` | **128** | Router signature width. |
| `lambda_ortho` $\lambda$ | **0.5** | From HOPE; sweep $\{0.1, 0.5, 1.0\}$. |
| `route_on` | **"signature"** | Ablation switch — see §5. |
| `use_recency` | **False** → True | True needs the §10 data-pipeline addition. |
| MoE placement | **every block** → top-6 | If memory-bound, put MoE only in the upper half. |

**Parameter & FLOP accounting.** Replacing 1 FFN with $M$ multiplies FFN params ≈ $M\times$ (22M → ~90M at $M{=}8$), but only $k$ experts fire, so active FFN FLOPs are ≈ $k\times$ dense. Report **two controls**: (i) headline vs. **dense RT** at matched *active*-FLOPs; (ii) a **param-matched** dense RT with `d_ff` scaled by ~$k$, to show gains aren't just parameters.

---

## 5. Routing-signal ablation (the core scientific test)

This is the experiment that rules the RGCN null result in or out. Swap only the router's input:

| `route_on` | `route_feat` fed to `MoEFFN` | Extra ingredient | Transfers? |
|---|---|---|---|
| **signature** (ours) | `z` | — | ✅ |
| **hidden** | normed hidden `h` | — (still leak-free: RT context is causal) | ✅ (weaker) |
| **value** | value component only | trunk must expose the type-value sum as a tensor | ✅ |
| **identity** (baseline) | `id_emb(col_global_id)` | an `nn.Embedding` over a global column vocab + `col_global_id` in batch | ❌ by construction |
| **dense** (control) | n/a (`moe=False`) | — | n/a |

`signature` and `hidden` are one-line switches in the code above. `value` and `identity` each need the one listed ingredient. **The headline claim is: `signature` ≥ `value`/`dense` in-distribution, and `signature` ≫ `identity` on held-out schemas.** If `signature` ≈ `dense`, the method adds nothing — stop and reconsider (see `idea.md` §7).

---

## 6. Experiment plan

### Phase 1 — `rel-f1` only (smoke test + all core ablations)
Tiny and fast. Pretrain RT-MoE (masked-cell) on `rel-f1`, then evaluate on **all three `rel-f1` tasks**: `driver-dnf` (binary, AUROC), `driver-top3` (binary, AUROC), `driver-position` (regression, MAE). Run here:
1. **Routing-signal ablation** (§5) — signature / value / identity / dense.
2. **Balancing ablation** — ortho+tail vs. Switch-uniform aux vs. expert-choice.
3. **$M$ and $k$ sweeps** — {4, 8} × {1, 2}.
4. **Temporal axis** — `use_recency` off vs. on (after §10).

*Gate to Phase 2:* MoE (signature) matches or beats dense RT on ≥2 of 3 `rel-f1` tasks at matched active-FLOPs, **and** expert-usage is non-degenerate (entropy not collapsed, §8).

### Phase 2 — three smallest, leave-one-database-out transfer (the real story)
Pretrain on two of {`rel-f1`, `rel-stack`, `rel-trial`}, evaluate **zero-shot** on the held-out third (RT's protocol; optionally `contd-pretrain` on the held-out DB with the eval task held out). Three folds. Compare **signature vs. identity** routing — identity *cannot* transfer, so this should be a clean qualitative win and is the headline transfer result.

> Enumerate each dataset's tasks programmatically rather than hard-coding — RelBench task names beyond `rel-f1` are easy to get wrong. `rel-stack` and `rel-trial` each expose entity-classification, entity-regression, and recommendation tasks; pull them via the RelBench API (`get_task_names(db)`) / `rt/tasks.py`. Start with the entity (node-level) tasks; recommendation stresses the head differently and can come later.

---

## 7. Commands

The README's example scripts are hard-coded to `rel-amazon/user-churn`. Copy and edit them (or wire the new args through their config):

```bash
# Phase 1: pretrain RT-MoE on rel-f1 (single GPU is fine for rel-f1)
cp scripts/example_pretrain.py scripts/moe_pretrain.py
# edit moe_pretrain.py: dataset=rel-f1; pass moe=True, num_experts=4, k=2,
#   d_sig=128, route_on="signature", lambda_ortho=0.5
pixi run torchrun --standalone --nproc_per_node=1 scripts/moe_pretrain.py

# Phase 1: evaluate on each rel-f1 task (reuse RT's finetune/eval path)
cp scripts/example_finetune.py scripts/moe_finetune.py
# edit: task in {driver-dnf, driver-top3, driver-position}; load the pretrained ckpt
pixi run torchrun --standalone --nproc_per_node=1 scripts/moe_finetune.py

# Phase 2: leave-one-DB-out (held-out = rel-trial shown; rotate over all three)
# pretrain on rel-f1 + rel-stack, zero-shot eval on rel-trial tasks
```

Set up logging first: `pixi run wandb login` (or `wandb disabled`).

---

## 8. Metrics & diagnostics

**Task metrics** come free from RT's existing eval (AUROC for binary, MAE for regression). Add three MoE-specific diagnostics:

**(a) Expert-usage vs. relation frequency.** Collect gates `g` over a validation pass; per-expert usage $f_e = \text{mean}_{\text{tokens}} \mathbb{1}[g_{\cdot,e} > 0]$. Compute entropy $H(f)$ and compare $f$ against the empirical column/relation-frequency distribution. **Claim: usage should track the long tail, not flatten to uniform.**

```python
@torch.no_grad()
def expert_usage(model, loader, n_experts, device):
    import torch
    tot = torch.zeros(n_experts, device=device)
    for batch in loader:
        _, _ = model(batch_to(batch, device))            # gates captured via a hook
        # register a forward hook on each MoEFFN to accumulate (g>0).float().sum((0,1))
    f = tot / tot.sum()
    H = -(f.clamp_min(1e-9) * f.clamp_min(1e-9).log()).sum()
    return f.cpu(), H.item()
```

**(b) Routing-invariance / leakage test (signature routing).** With `route_on="signature"`, a cell's gate is a pure function of its own $(c, \sigma, \Delta)$. Assert it is identical across two different sampled contexts for the same target row:

```python
def test_routing_invariance(model, row, ctx_a, ctx_b, tol=1e-6):
    g_a = gate_for_target(model, row, ctx_a)   # gate vector at the target cell
    g_b = gate_for_target(model, row, ctx_b)   # different BFS sample of neighbors
    assert (g_a - g_b).abs().max() < tol, "router depends on context -> leakage risk"
```

This is a genuine selling point: it is a *unit-testable* guarantee that routing cannot leak future/neighbor information. (Under `route_on="hidden"` the test won't hold exactly — routing then depends on neighbors — but it is still leak-free because RT's neighbors are causal.)

**(c) Specialization probe (the HER evidence, transplanted).** Cluster columns by their argmax expert. **Signature routing should group semantically-similar columns *across tables*** (e.g. all monetary-numeric columns together); identity routing should partition by table. This is the qualitative figure for the type-overfitting claim.

---

## 9. Milestones / go-no-go

1. **Wiring** — RT-MoE trains on `rel-f1`, loss decreases, ortho loss is finite and non-zero. *(sanity)*
2. **In-distribution parity** — signature-MoE ≥ dense RT on ≥2/3 `rel-f1` tasks at matched active-FLOPs. *(go/no-go for the whole idea — see `idea.md` §7)*
3. **Non-degenerate routing** — usage entropy not collapsed; specialization probe shows cross-table semantic clusters.
4. **Transfer** — signature ≫ identity on leave-one-DB-out (Phase 2). *(the headline result)*
5. **Temporal lift** — recency axis helps horizon-sensitive tasks (after §10).

If (2) fails, the most likely cause is the RGCN null result; pivot to the `value`-augmented signature or to attention-mask MoA routing before abandoning.

---

## 10. Caveats & known gaps

- **Recency needs one data-pipeline addition.** $\Delta_i = T_{\text{seed}} - \tau(\text{row}_i)$ is a per-context quantity the model's `forward` doesn't currently receive. The rust sampler (`rustler`) already knows the seed time and each row's timestamp (it uses them to exclude future rows), so emit a per-cell `recency` and bin it, or compute it in `rt/data.py` from the timestamps already in hand; pass it as `batch["recency_bins"]` (bin 0 = no timestamp). **Until wired, run with `use_recency=False`** — schema + modality routing works out of the box and covers Phase 1's core ablations.
- **Dense expert combine is notional sparsity.** The MVP computes every expert on every token (correct, simple). It does not realize the FLOP saving; for `rel-f1` this is irrelevant. At `rel-stack`/`rel-trial` scale, switch to a masked/gathered dispatch or grouped matmul (and only then is the "conditional compute" property real).
- **Text cells aren't a masked-loss target in RT** (`"masking text not supported"`), so don't expect the MoE to specialize text *prediction*; it still routes and transforms text tokens for downstream tasks.
- **`col_name_values` vs. "column of table".** The reference code routes on the column-name embedding; table identity enters separately (via the column-attention mask). If you want the paper's "of table" semantics in the signature, concatenate a frozen table-name embedding into $s_i$ the same way — but the minimal faithful choice is to route on `col_name_values` as-is.
- **Task-name accuracy.** Only `rel-f1`'s three tasks are hard-coded here with confidence; enumerate `rel-stack`/`rel-trial` tasks via the API.
- **Preprint maturity.** HER (2511.07603) was withdrawn pending revision and HOPE is a recent preprint; the *mechanisms* we borrow (type-collapse avoidance, router orthogonality) are sound, but cite them as working preprints, not settled results.