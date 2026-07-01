# MoRE Ablation Suite v2

*Supersedes `moe_ablation_suite.md`. Aligned to the actual code (`rt_substrate.py`, `moe.py`). Adds the router-input axis ("route on the hidden state or not") as a first-class study, and keeps the four architecture additions (shared expert, cosine router, Top-P, HMoE). Runs on **rel-f1** and **rel-trial**. Includes explicit good/bad predictions.*

---

## 0. What this tests, and how the two datasets are used

Two independent things are under test:

1. **Router input** — what the gate reads. Your code exposes this as `route_on`. We add one mode (`hybrid`) so we can answer your question: *does adding the evolving hidden state to the router help, given it costs pre-computability?*
2. **Architecture additions** — shared expert (**S**), cosine router (**C**), Top-P (**P**), hierarchical two-level gating (**H**), layered on the winning router.

The two datasets play two roles:

- **Within-dataset (complexity contrast).** rel-f1 (9 tables, ~16 numeric columns, few relations) is a **negative control**: little to specialize on, so a correct MoE gives small gains. rel-trial (15 tables, 140 columns) is the **positive case**. A variant that helps trial but not f1 is behaving as predicted; one that *hurts* f1 is showing overhead/overfitting.
- **Cross-dataset (transfer).** rel-f1 and rel-trial share ~no column names, so "pretrain on one, zero-shot the other" is a clean transfer test. This is the only setting that separates **semantic vs identity routing** — identity routing is *undefined* on an unseen schema (its per-column id table has no entries), while signature routing is defined by construction. Absolute zero-shot numbers will be modest (single-DB pretraining, vs RT's six), but the **router ranking** is the signal.

---

## 1. Router-input axis (the focus)

Mapped to `route_on`. `hybrid` is new; everything else exists in your `RTSubstrate` / `RelationalBlock._route_feat`.

| `route_on` | router reads | `d_route` | pre-computable? | question it answers |
|---|---|---|---|---|
| `dense` | — (no router) | — | — | anchor: does any MoE help? |
| `dense_wide` | — (FFN × k) | — | — | param-matched anchor |
| `signature` | `z` | `d_sig` | **yes** | MoRE default (static, transferable) |
| `hybrid` **(new)** | `[z ; h]` | `d_sig + d_model` | no | **does adding the hidden state help?** |
| `hidden` | `h` | `d_model` | no | pure content routing (what does the explicit signature add over routing on content?) |
| `value` | `value_feat` | `d_model` | yes | control: semantics vs raw value |
| `identity` | `id_emb` | `d_sig` | yes | control: semantics vs identity (cannot transfer) |

Three questions fall out cleanly:
- **Does routing help at all, and track complexity?** `dense`/`dense_wide` vs `signature`.
- **Is the semantic signal doing the work?** `signature` vs `value` vs `identity` (in-distribution *and* transfer; the transfer gap is the real test).
- **Should the router also see the evolving state?** `signature` vs `hybrid` vs `hidden`. `hybrid` keeps the schema/temporal signal and adds layer-adaptivity (the ViMoE finding: later layers are more semantic); `hidden` drops the explicit signal entirely.

---

## 2. The four architecture additions

Base notation: `h = norms["ffn"](x)` is the block hidden state the experts transform; `route_feat` is the router input from §1; `E_j` is a `SwiGLU` expert.

### S — shared + routed experts
$$y = E_{\text{shared}}(h) + \sum_{j \in \mathcal{S}} G_{j}\, E_j(h)$$
Always-on expert absorbs the computation common to all columns; routed experts capture only the residual. **This is the addition most likely to help everywhere** — it defuses the RGCN null result (routed experts must beat zero, not re-derive the whole function) and stabilizes training. Cost: +1 always-active expert per MoE block.

### C — cosine / normalized router
Learnable keys $e_j \in \mathbb{R}^{d_{\text{route}}}$, temperature $\tau$:
$$s_{ij} = \frac{\text{route\_feat}_i^\top e_j}{\lVert \text{route\_feat}_i\rVert\,\lVert e_j\rVert}, \qquad G_i = \mathrm{softmax}(\mathrm{TopK}(s_{i\cdot}/\tau, k)).$$
Reported to beat linear routers for cross-domain generalization (your transfer setting). Unifies with balancing: `ortho_loss` now acts on the **keys** instead of `router.weight`.

### P — Top-P (adaptive expert count)
Softmax over all experts, then keep the smallest set reaching cumulative mass $P$; renormalize. Compute scales with column complexity (generic column → 1 expert; rich column → several). Efficiency lever; report **mean active experts** $\bar k$ per dataset.

### H — hierarchical two-level gating
$$G^{(1)} = \mathrm{softmax}(W_1\,\text{route\_feat})\ \text{over } \Gamma \text{ groups}, \quad G^{(2)}_g = \mathrm{softmax}(W_{2,g}\,\text{route\_feat})\ \text{within group},$$
$$y = \sum_g G^{(1)}_g \sum_{j\in\text{group}_g} G^{(2)}_{g,j} E_j(h).$$
Default level 1 = **learned coarse groups** from `route_feat` (no extra plumbing). Tier-3 variant: level 1 = **modality**, by threading `cb.sem_types` through `forward` as an extra routing tensor. **Highest-variance addition**; needs a per-level balance term and is the one most likely to hurt on rel-f1.

---

## 3. The suite

Naming: router arm from §1 (default `signature`), plus addition tags S/C/P/H. `M0 = base MoEFFN` (top-k, no additions), `M=4`, `k=2`, `moe_placement="all"` unless swept.

### Tier 0 — Anchors (both datasets, within-dataset)
`dense`, `dense_wide`.

### Tier 1a — Router study, within-dataset (both datasets)
Base `M0` under each router: `signature`, `hybrid`, `hidden`, `value`, `identity`. → pick winner **R\***. Tests "does semantics hurt in-distribution" (should not) and the **hidden-or-not** bump.

### Tier 1b — Router study, transfer (train A → zero-shot B, both directions)
Same five routers. This is where `signature`/`hybrid` should separate from `value`/`identity`. `identity` ≈ trivial (undefined on unseen schema); `hidden` defined but no explicit schema signal.

### Tier 2 — Additions ladder on R\* (both datasets, within-dataset)
Forward selection + backward leave-one-out, so each addition is bracketed from both sides:

| ID | Config | Direction |
|---|---|---|
| R\* | base | forward start |
| R\*+S | +shared | forward |
| R\*+S+C | +cosine | forward |
| R\*+S+C+P | +top-p | forward |
| **Full** = R\*+S+C+P+H | +hmoe | forward end |
| Full−S / Full−C / Full−P / Full−H | leave-one-out | backward |

If `signature` and `hybrid` finish within one seed-std in Tier 1a, run Tier 2 on **both** (the additions may interact differently with static vs dynamic routing); otherwise run on R\* only.

### Tier 3 — Internal sweeps (survivors only, rel-trial)
Cosine $\tau \in \{0.1, 0.3, 1.0\}$; Top-P $P \in \{0.5, 0.7, 0.9\}$; HMoE grouping {learned, modality} and $\Gamma \in \{4, 8\}$; shared additive vs blended, 1 vs 2 shared; $M \in \{4, 8\}$, $k \in \{1,2\}$.

### Tier 4 — Efficiency + diagnostics (every run)
Active params, active FLOPs, $\bar k$ (Top-P), routing-time fraction with/without precompute (`signature` only), wall-clock. Diagnostics: routed-expert usage entropy vs relation-frequency; HMoE level-1 group occupancy; leakage-invariance test (`signature`: a cell's gate identical across two neighbor samples — will *not* hold for `hybrid`/`hidden`, by design).

---

## 4. Predictions — good vs bad

### Router study (Tier 1)
| Router | rel-f1 (within) | rel-trial (within) | transfer (A→B) | note |
|---|---|---|---|---|
| `signature` | ≈ dense | > dense | **above trivial** | the default; transferable |
| `hybrid` | ≈ / slightly > signature | **> signature (larger)** | ≈ signature (uncertain) | layer-adaptive; **but loses pre-compute** |
| `hidden` | ≈ signature | ≈ signature | < signature | no explicit schema signal |
| `value` | ≈ | ≈ | < signature | value ≠ semantics |
| `identity` | ≈ / slightly > (memorizes) | ≈ / slightly > | **≈ trivial (undefined)** | the clean transfer failure |

**Read:** in-distribution, the routers will look similar (identity can even edge ahead by memorizing). The story is the **transfer** column: `signature`/`hybrid` stay above trivial, `identity` collapses. On the **hidden-or-not** question, expect `hybrid` to give a *small* within-dataset bump that is **larger on rel-trial** (more layers/columns to exploit adaptivity); keep it only if that bump is real and it does not hurt transfer, since it costs the precompute win.

### Additions (Tier 2)
| Addition | rel-f1 | rel-trial | verdict |
|---|---|---|---|
| **S** shared | ≈ / slightly > | **largest single win** | robust; promote to default |
| **C** cosine | ≈ | > (heterogeneous cols) | conditional (complex schema / transfer) |
| **P** top-p | ≈ quality, **↓ compute** | ≈/> quality, adaptive | efficiency play |
| **H** hmoe | **likely < (overkill)** | > iff grouping healthy, else unstable | complexity-gated, highest variance |
| **Full** | risky (H may drag) | best iff H behaves | only wins if every part earns its place |

**Summary:** robust winner = **S**. Conditional winners (trial only) = **C**, **H**. Efficiency = **P**. Most likely to disappoint, especially on f1 = **H**. If **S** fails to help even on rel-trial, worry about the core idea, not the add-ons.

---

## 5. Protocol

Tasks — rel-f1: `driver-dnf` (AUROC), `driver-top3` (AUROC), `driver-position` (MAE). rel-trial: `study-outcome` (AUROC), `study-adverse` (MAE), `site-success` (MAE) — confirm the entity-task list via the RelBench API. 3 seeds, mean ± std (variance matters for H and P). Headline vs `dense` at matched active-FLOPs; also report `dense_wide` (param-matched). Same pretrain+finetune recipe across arms.

Decision rules: **keep** an addition iff it helps rel-trial in *both* forward and backward tests and does not hurt rel-f1 beyond one std; **complexity-gate** (enable only for complex schemas) iff it helps trial but hurts/neutral on f1; **drop** iff it fails on trial or raises variance without a quality gain; for efficiency items accept ≤ ~0.5 std quality cost for a material compute drop.

---

## 6. Code deltas (against your `moe.py` / `rt_substrate.py`)

### 6.1 Router input — add `hybrid`, keep `d_route` explicit
```python
# rt_substrate.py  RTSubstrate.__init__  — replace the d_route line:
if route_on in ("signature", "identity"):
    d_route = d_sig
elif route_on == "hybrid":
    d_route = d_sig + d_model          # [z ; h]; not pre-computable (accepted)
else:                                   # hidden, value
    d_route = d_model
is_moe_arm = route_on not in ("dense", "dense_wide")

# rt_substrate.py  RelationalBlock._route_feat  — add one entry:
def _route_feat(self, h, z, value_feat, id_emb):
    return {
        "signature": z,
        "hidden": h,
        "value": value_feat,
        "identity": id_emb,
        "hybrid": torch.cat([z, h], dim=-1),
    }[self.route_on]
```
`z` is `[B,S,d_sig]`, `h` is `[B,S,d_model]`; concat is `[B,S,d_sig+d_model]`, matching `d_route`. No other changes — `forward` already threads `z`/`value_feat`/`id_emb`, and `hybrid` needs both `z` (arg) and `h` (local), both in scope.

### 6.2 Additions — extend `MoEFFN` (flags keep arms comparable)
```python
# moe.py  MoEFFN.__init__  — add kwargs:
def __init__(self, d_model, d_ff, d_route, *, num_experts=4, k=2,
             use_shared=False, cosine=False, tau=0.3, top_p=None):
    ...
    self.shared = SwiGLU(d_model, d_ff) if use_shared else None
    self.cosine, self.tau, self.top_p = cosine, tau, top_p
    if cosine:
        self.keys = nn.Parameter(torch.randn(num_experts, d_route))
    else:
        self.router = nn.Linear(d_route, num_experts, bias=False)

# gates(): cosine logits + top-k OR top-p selection
def gates(self, route_feat):
    if self.cosine:
        rc = F.normalize(route_feat, dim=-1); kc = F.normalize(self.keys, dim=-1)
        logits = (rc @ kc.t()) / self.tau
    else:
        logits = self.router(route_feat)
    if self.top_p is None:                                   # top-k (existing behavior)
        topv, topi = logits.topk(self.k, dim=-1)
        masked = torch.full_like(logits, float("-inf")); masked.scatter_(-1, topi, topv)
        return F.softmax(masked, dim=-1)
    probs = F.softmax(logits, dim=-1)                         # top-p
    sp, idx = probs.sort(dim=-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < self.top_p                 # smallest set reaching P
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, idx, keep)
    g = probs.masked_fill(~mask, 0.0)
    return g / g.sum(-1, keepdim=True).clamp_min(1e-9)

# forward(): add shared branch
def forward(self, x, route_feat):
    g = self.gates(route_feat)
    y = torch.zeros_like(x)
    for e, expert in enumerate(self.experts):
        y = y + g[..., e:e+1] * expert(x)
    if self.shared is not None:
        y = y + self.shared(x)
    return y, g

# ortho_loss(): decorrelate keys when cosine, else router.weight
def ortho_loss(self):
    W = F.normalize(self.keys if self.cosine else self.router.weight, dim=-1)
    gram = W @ W.t()
    return ((gram - torch.eye(self.num_experts, device=W.device, dtype=W.dtype)) ** 2).sum()
```
Thread `use_shared`/`cosine`/`tau`/`top_p` from `RTSubstrate.__init__` → `RelationalBlock` → `MoEFFN` (same path as `num_experts`/`k`).

### 6.3 HMoE — separate class (keeps `MoEFFN` flat)
```python
# moe.py  new class; level-1 learned-coarse gate + per-group level-2 (dense combine MVP)
class HMoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, d_route, *, n_groups=4, experts_per_group=2, k2=1):
        super().__init__()
        self.n_groups, self.k2 = n_groups, k2
        self.g1 = nn.Linear(d_route, n_groups, bias=False)
        self.g2 = nn.ModuleList(nn.Linear(d_route, experts_per_group, bias=False)
                                for _ in range(n_groups))
        self.experts = nn.ModuleList(
            nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(experts_per_group))
            for _ in range(n_groups))
    def forward(self, x, route_feat):
        p1 = F.softmax(self.g1(route_feat), dim=-1)           # [...,G] soft over groups
        y = torch.zeros_like(x)
        for gi in range(self.n_groups):
            l2 = self.g2[gi](route_feat)
            tv, ti = l2.topk(self.k2, dim=-1)
            m = torch.full_like(l2, float("-inf")); m.scatter_(-1, ti, tv)
            p2 = F.softmax(m, dim=-1)                          # within-group top-k2
            grp = torch.zeros_like(x)
            for e, expert in enumerate(self.experts[gi]):
                grp = grp + p2[..., e:e+1] * expert(x)
            y = y + p1[..., gi:gi+1] * grp
        return y, (p1,)
    def ortho_loss(self):                                     # decorrelate level-1 gate rows
        W = F.normalize(self.g1.weight, dim=-1)
        return ((W @ W.t() - torch.eye(self.n_groups, device=W.device, dtype=W.dtype)) ** 2).sum()
```
Modality-grounded level 1 (Tier 3): pass `cb.sem_types` (or a modality embedding) as an extra routing tensor through `RTSubstrate.forward` → block → `HMoEFFN`, and replace `self.g1(route_feat)` with a gate on the modality signal. Add a level-1 balance term (entropy of mean `p1`) if groups collapse. Keep dense combine for rel-f1; switch to top-1-group dispatch for rel-trial scale.

---

## 7. Run order
1. Tier 0 + Tier 1a on **rel-f1** (fast; first read on routers).
2. Tier 0 + Tier 1a on **rel-trial**; pick R\*.
3. Tier 1b **transfer** (both directions) — the core semantic-vs-identity result.
4. Tier 2 additions ladder on R\* (both datasets).
5. Tier 3 sweeps on rel-trial for survivors.
6. Tier 4 efficiency folded into every run.

Meeting headline: Tier 1a routers side by side (does semantics hurt in-distribution + the hidden-or-not bump), Tier 1b transfer (identity collapses, signature holds), and the Tier 2 forward ladder (shared as the robust win; cosine/HMoE complexity-gated across the f1↔trial contrast).