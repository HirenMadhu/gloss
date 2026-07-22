# The single-table model (SetJoin) — current architecture reference

Self-contained description of the model as it stands after the S10–S12 gates (2026-07). Everything
here matches the code on the `setjoin` branch: [gloss/setjoin/model.py](gloss/setjoin/model.py),
[gloss/model/moe.py](gloss/model/moe.py), [gloss/model/column_encoder.py](gloss/model/column_encoder.py),
[gloss/setjoin/collate.py](gloss/setjoin/collate.py), [gloss/setjoin/recency.py](gloss/setjoin/recency.py).

**One sentence:** per prediction seed, flatten the many-to-one closure of the seed row into **one wide
row** of cell tokens, collect all 1-hop child rows into **one table-tagged union set**, encode the wide
row with a CLS transformer and the set with a seed-conditioned set transformer — and make **every FFN in
both encoders a signature-routed Mixture-of-Experts** (MoRE's `MoEFFN`, reused verbatim): the router
reads a *value-free* relational signature, the experts transform the evolving hidden state.
*"Route on semantics, transform the content."*

**Naming:** this is the **single-table** model (SetJoin). The **multi-table** model (MoRE) is the
cell-token relational transformer with the same MoE FFN inside RT's relational blocks; the single-table
model reuses its data pipeline, cell encoder, text stack, and `MoEFFN` without modifying any of it.

---

## 1. Inputs: what one seed's batch contains

From the same leakage-safe temporal neighbor sampler as the multi-table model (disjoint per-seed
sampling, fanout 64, no row with `row_time > seed_time`), the collate builds a `JoinBatch` with two
structures per seed:

**The wide seed row** — `W = 128` **cell** tokens `[B, W]`: the seed row's own cells plus the
many-to-one closure (parents and grandparents, join depth 2) flattened in. Every cell carries its
global column id, node-type id, **join-path id** (which m2o walk produced it; path 0 = the seed's own
row), row time, and timed flag. A path whose walk dies (NULL FK, sampler miss, temporal exclusion)
emits exactly **one marker slot** (`wide_missing`) so the model can distinguish "this FK is absent"
from "columns were truncated". Seed-own cells are emitted first so they survive the cap.

**The union set** — `N = 128` **row** elements `[B, N]`: all 1-hop child rows across all child
relations, merged into one set, **most-recent-first**, truncated at `N`. Each element carries its
child-table id, the `fk_role_id` of the child→seed relation, row time, and hop (1 for direct children;
hop 2 exists in the code but is gated off — see §9). Two FKs from one table into the seed's table get
distinct `fk_role_id`s.

Plus per child relation the **post-sampler, pre-truncation child counts** `[B, R]` (so truncation
never hides multiplicity from the model), and per seed the seed time, target, and labelled mask.

### Recency bins (used everywhere below)

Fixed, context-independent log-spaced buckets of the age
$\Delta = t_{\text{seed}} - t_{\text{row}} \ge 0$ (one bin per order of magnitude, edges
$10^0 \dots 10^{18}$ in the native time unit; **bin 0 reserved for untimed/pad**):

$$\mathrm{rec}(u) = \begin{cases} 0 & \text{untimed} \\ 1 + \#\{i : 10^i \le \Delta\} & \text{timed} \end{cases}$$

A row's bin depends only on its own timestamp and the seed time — never on the batch or the sampled
neighborhood. This is what makes recency a legal *routing* feature (§5).

---

## 2. Cell encoding (RT's additive cell token, unchanged)

Each sampled node type is encoded **once** and shared between the wide grid and the set elements
(`CellEncoder.encode_type`). Per cell (row $u$, column $c$, value $v_{u,c}$):

$$x_{u,c} \;=\; \underbrace{W_v\,\mathrm{Enc}_{\text{stype}}(v_{u,c})}_{\text{value component}} \;+\; \underbrace{W_{\text{name}}\, n_c}_{\text{schema component}}$$

- $\mathrm{Enc}_{\text{stype}}$ — pytorch-frame's per-modality encoders (categorical embedding, linear
  numerical, multicategorical, timestamp, text-embedding), output width `enc_channels = min(d_model, 256)`.
- $n_c \in \mathbb{R}^{d_\text{text}}$ — the **frozen** column-name embedding
  (`Qwen/Qwen3-Embedding-4B`, $d_\text{text} \approx 2560$), precomputed offline and cached; gathered by
  global column id. No LM forward passes in training.
- $W_v: \mathbb{R}^{\text{enc}} \to \mathbb{R}^{d}$, $W_{\text{name}}: \mathbb{R}^{d_\text{text}} \to \mathbb{R}^{d}$, $d = d_{\text{model}} = 256$.

---

## 3. The wide-row encoder → `seed_repr`

**Token assembly.** Each wide slot's cell token gets additive structural tags (all learned embeddings
into $\mathbb{R}^d$):

$$h_w \;=\; x_{u,c} \;+\; P(\text{path}_w) \;+\; R(\mathrm{rec}_w)$$

A missing-parent **marker** slot has no cell, so instead:

$$h_w \;=\; m \;+\; T(\text{missing table}_w) \;+\; P(\text{path}_w) \;+\; R(\mathrm{rec}_w)$$

with $m$ a single learned missing-embedding. Pad slots are zeroed and key-masked.

**Encoder.** A learned CLS token is prepended (never key-masked, so an all-pad row still reads out
finite), then `n_wide_layers = 2` pre-norm **MoE layers**. Each layer, on states
$X \in \mathbb{R}^{B \times (W{+}1) \times d}$ with pad mask:

$$\begin{aligned} X &\leftarrow X + \mathrm{Drop}\big(\mathrm{MHA}(\mathrm{LN}(X)) \big) \\ X &\leftarrow X + \mathrm{Drop}\big(\mathrm{MoEFFN}(\mathrm{LN}(X);\, Z)\big) \end{aligned}$$

where MHA is standard multi-head self-attention over the wide slots ($H = 4$ heads, head dim $d/H$):

$$\mathrm{head}_i = \mathrm{softmax}\!\Big(\tfrac{Q_i K_i^\top}{\sqrt{d/H}} + M_{\text{pad}}\Big) V_i, \qquad Q_i = X W_i^Q,\; K_i = X W_i^K,\; V_i = X W_i^V$$

($M_{\text{pad}}$ = $-\infty$ on pad keys). $Z$ is the per-token **routing signature** (§5) — the MoE
router reads $Z$, the experts transform the hidden state. Readout:

$$\text{seed\_repr} = X^{(L)}_{\,:,\,0} \in \mathbb{R}^{B \times d} \quad (\text{the final CLS state})$$

---

## 4. The union-set encoder → `context`

**Row pooling.** Each child row (and each of its flattened parents) is compressed from its $C$ cells to
one vector by a single shared gated attention pool (`RowPool`; the frozen name tokens inside the cells
already disambiguate columns and tables, so one pool serves every node type):

$$\alpha_c = \mathrm{softmax}_c\big(w_2^\top \tanh(W_1 x_{u,c})\big), \qquad r_u = \sum_c \alpha_c \, x_{u,c}$$

**Element assembly (additive row scatter).** Element $n$'s content is the child row itself (at path
`FK_NONE`) **plus** each of the child's own flattened parents (at that FK's role id), summed into one
slot, then tagged:

$$E_n \;=\; \Big(\sum_{(u,\,p)\,\in\, n} r_u + F(p)\Big) \;+\; F(\text{fk\_role}_n) \;+\; T(\text{table}_n) \;+\; R(\mathrm{rec}_n) \;+\; H(\text{hop}_n)$$

followed by LayerNorm and masking. $F, T, R, H$ are learned embeddings (FK role, table type, recency
bin, hop). The **table tag is load-bearing**: `fk_role_id` is keyed by canonical FK column name and can
collide across tables.

**Set self-attention.** A learned **null element** is concatenated and never masked (attention always
has ≥ 1 key, so empty sets are well-defined and trainable), then `n_set_layers = 2` MoE layers with
exactly the same layer equation as §3 — self-attention over the $N{+}1$ elements, MoE FFN routed on the
element signature (§5). **No positional encoding on the set path** — permutation invariance of the
pooled context is a tested contract.

**Seed-conditioned PMA readout.** `n_pma = 4` learned queries, each shifted by a projection of the
seed representation, cross-attend into the contextualized set $X$:

$$q_i = p_i + W_q\,\text{seed\_repr}, \qquad \mathrm{ctx}_i = \mathrm{MHA}\big(q_i,\, X,\, X\big), \qquad \text{context} = W_o\,[\mathrm{ctx}_1; \dots; \mathrm{ctx}_4]$$

(cross-attention: queries from the PMA seeds, keys/values from the set; same masked softmax-attention
form as §3). So *what to extract from the children depends on who the seed is*.

---

## 5. The routing signatures (value-free by construction)

The router never sees cell **values**, hidden states, or any neighborhood/global statistic — only a
pure function of a token's own `(schema position, modality, recency, structure)`. Both signatures live
in $\mathbb{R}^{d_{\text{sig}}}$, $d_{\text{sig}} = 64$, and are computed once per forward (shared
across layers).

**Wide tokens are cells** → they route on the true MoRE cell signature, plus the join path:

$$z^{\text{wide}}_w = \mathrm{RMSNorm}\big( W_s\, n_c \;+\; \psi(\text{modality}_c) \;+\; \varphi(\mathrm{rec}_w) \;+\; \pi(\text{path}_w) \big)$$

($W_s$ projects the same frozen name embedding as §2; $\psi$ = pytorch-frame stype embedding;
$\varphi$ = recency-bin embedding; $\pi$ = join-path embedding). Marker/pad slots keep only the
path + recency terms (there is no column), and a learned CLS signature is prepended to match the CLS
token.

**Set elements are rows** → they route on the row-level analog:

$$z^{\text{elem}}_n = \mathrm{RMSNorm}\big( T'(\text{table}_n) \;+\; F'(\text{fk\_role}_n) \;+\; \varphi'(\mathrm{rec}_n) \;+\; H'(\text{hop}_n) \big)$$

with a learned null-element signature prepended to match the null token.

Changing any cell *value* changes neither signature; a seed's signatures are identical across two
different neighbor samples (both are unit-tested invariants).

---

## 6. The shared MoE FFN (`MoEFFN` — the one mechanism, reused verbatim from the multi-table model)

Every FFN in both encoders (2 wide layers + 2 set layers = 4 MoE FFNs in the standing config) is the
same module: a pool of $M$ identical SwiGLU experts plus a sparse top-$k$ router. **The crucial
asymmetry:** the router reads the value-free signature $z$; the experts transform the evolving hidden
state $h$.

**Expert** (SwiGLU, $d_{\text{ff}} = 1024$):

$$E_j(h) = W_2^{(j)}\big( \mathrm{SiLU}(W_1^{(j)} h) \odot W_3^{(j)} h \big)$$

**Router** (linear, no bias; $M = 4$ experts, $k = 2$):

$$\ell = W_g\, z \in \mathbb{R}^{M}, \qquad g_j = \begin{cases} \dfrac{e^{\ell_j}}{\sum_{j' \in \mathrm{top}k(\ell)} e^{\ell_{j'}}} & j \in \mathrm{top}k(\ell) \\ 0 & \text{otherwise} \end{cases}$$

(softmax over the top-$k$ support only; gates sum to 1). **Output:**

$$y = \sum_{j \in \mathrm{top}k} g_j \, E_j(h)$$

The router input width is fixed by the arm ($d_{\text{sig}}$ for `signature`, $d_{\text{model}}$ for
`hidden`), so the router dimension never silently changes across ablation arms. Combine is **dense**
(all $M$ experts run on every token; the gate zeros the non-selected ones) — the simple correct MVP;
true sparse dispatch is deferred, so no active-FLOP-parity claims.

**Balance = router orthogonality, not load balancing.** Per MoE layer, with row-normalized router
directions $\hat W_g$ ($\hat w_j = w_j / \lVert w_j \rVert$):

$$\mathcal{L}_{\text{ortho}} = \big\lVert \hat W_g \hat W_g^\top - I \big\rVert_F^2$$

summed over all MoE layers and returned as `aux`. This pushes the experts' *routing directions* apart
without forcing uniform usage — expert load is free to follow the long tail of relation/column
frequencies (a uniform load-balancing loss would fight exactly the structure we want the router to
learn).

**Routing arms** (`route_on`): `signature` (the method — router reads $z$) | `hidden` (router reads the
pre-FFN LayerNormed hidden state — the "does value-free routing matter?" control) | `dense` (plain GELU
transformer layers, `aux = 0` — the no-MoE baseline gate).

Optional additions living in the class but **off in the standing config** (the S/C/P/H ablation was
negative): shared always-on expert, cosine router, Top-P adaptive expert count, hierarchical two-level
gate (`HMoEFFN`).

---

## 7. Head and objective

$$\text{cnt} = W_{\text{cnt}}\, \log(1 + \text{child\_counts}), \qquad \text{logits} = \mathrm{MLP}\big( \mathrm{LN}([\,\text{seed\_repr}\,;\,\text{context}\,;\,\text{cnt}\,]) \big)$$

(MLP: $3d \to d$, GELU, dropout, $d \to$ `out_dim`; `out_dim = 1` for binary and regression). The raw
per-relation child counts enter here — post-sampler, pre-truncation — so multiplicity survives the set
cap. Loss:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{ortho}} \sum_{\text{MoE layers}} \mathcal{L}_{\text{ortho}}, \qquad \lambda_{\text{ortho}} = 0.5$$

$\mathcal{L}_{\text{task}}$ = BCE-with-logits over labelled seeds (binary) or masked MSE on z-scored
targets (regression; standardized with TRAIN stats, de-standardized for metrics). Regression is
reported as **NMAE = MAE / train-std**; classification as AUROC. TEST split, 3 seeds.

---

## 8. Standing configuration (adopted at the S10 gate, confirmed by S11/S12)

`d256 2+2 ff1024, signature e4 k2` — **16.9 M params**, the ≤30M mean-rank winner of the 18-config
backbone sweep (also the unrestricted winner).

| Group | Setting | Value |
|---|---|---|
| Model | `d_model` / `n_heads` | 256 / 4 |
| | `n_wide_layers` / `n_set_layers` | 2 / 2 |
| | `d_ff` (per expert) | 1024 |
| | `enc_channels` | 256 (= min(d_model, 256)) |
| | `n_pma` / dropout | 4 / 0.1 |
| MoE | `route_on` | `signature` |
| | `num_experts` / `k` / `d_sig` | 4 / 2 / 64 |
| | `lambda_ortho` | 0.5 |
| | shared expert / cosine / Top-P / HMoE | all off |
| Data | fanout / `wide_len` / `set_size` | 64 / 128 / 128 |
| | m2o join depth (wide) | 2 (parents + grandparents) |
| | hop-2 set elements (`fanout2`) | **off** (S11 rejected) |
| | per-relation recency cap | **off** (S12 rejected) |
| Train | batch / lr / weight decay / epochs | 128 / 3e-4 / 0.01 / 30 |
| | text encoder | Qwen3-Embedding-4B, frozen, cached |

---

## 9. Components in the code but not in the standing model

- **Axial cell encoder** (`n_axial_layers > 0`): per-table row-level + column-level cell attention
  grids before pooling — RT's same-row/same-column patterns at cell granularity. Default 0.
- **Hop-2 union elements** (`fanout2 > 0`): grandchild/sibling/co-child rows as extra set elements.
  **Rejected at the S11 gate** — hurt 3 of 4 measured tasks; the 2-hop-signal hypothesis is dead on
  this substrate.
- **Per-relation recency cap**: keep the $N$ most recent per (hop, table, FK-role) group before the
  merged sort. **Rejected at S12** — the plain merged recency sort already keeps the golden rows.
- **`set_size = 256`**: not adopted globally (dilutes driver-top3) but a useful per-task knob — flips
  study-outcome above RT and helps driver-dnf / user-attendance.
- **Slot-attention readout** (`readout ∈ {pma, measure, slot}`, default `pma`): see §11. `pma` is the
  standing model; `measure`/`slot` are an S13 axis under evaluation (`results/setjoin_readout/`).

## 10. Where it stands (headline numbers)

Per-task sweep bests (3 seeds, TEST): **beats RT-from-scratch on 6/9** leaderboard entity tasks and the
multi-table (MoRE) grid best on 3/9; `rel-event/user-ignore` 90.5 AUROC is the best result of any model
in this repo. The single-table model also decays flatter over the prediction horizon and is ~2.6×
faster at inference than the multi-table model at deployed configs. Winner's-curse caveat: per-task
bests are best-of-18 configs.

## 11. The slot-attention readout axis (S13; `pma` is standing) — `SlotReadout` in `model.py`

The §4 PMA readout pools child rows with a **softmax convex combination** (weights sum to 1), so it is
**invariant to the number of children** — it structurally cannot represent COUNT/SUM/AVG-over-a-relation,
the dominant relational signal (the head's raw `w_cnt` log-counts half-admit this). The `readout` axis
(parallel to `route_on`) replaces it with a **differentiable GROUP BY**, keeping the count instead of
normalizing it away. It sits **after** the set encoder, touches no routing (the MoE router stays
value-free), and emits the same `context [B, d]` the head expects — so `readout=pma` is the unchanged,
bit-for-bit-reproducible default.

**Measure pool** (the one change — do *not* divide the aggregation): on assignment `attn [B,K,N]` and
`V = W_v(H)`, $m = \sum_N \text{attn}$ (soft COUNT), $S = \text{attn}\,V$ (SUM),
$\mu = S/(m{+}\epsilon)$ (MEAN recovered), $\text{update} = W_o[\,\mu\,;\,\log(1{+}m)\,;\,S/\log(1{+}N)\,]$
(count/sum log-compressed vs cardinality; channels configurable). Empty groups → true-zero $m,S,\mu$.

- **`slot`** — one **schema-seeded** slot per `(table, fk_role)` relation ($K=\lvert\text{fk\_relations}\rvert$,
  from `paths.slot_relations`; collate tags each element with `set_group_idx`). Slot init
  $= W_{\text{slot}}[\,T(\text{table})\,;\,F(\text{fk\_role})\,] + p_{\text{slot}} + W_{\text{seed}}(\text{seed\_repr})$.
  Assignment logits $= W_q(\text{slot})\,W_k(H)^\top/\sqrt d + \gamma\cdot\text{keymatch}$; `slot_mode`:
  **hard** ($\text{attn}=\text{keymatch}$, deterministic per-relation pooling), **soft** (softmax **over
  the $K$ slots**, learned $\gamma$ key-bias), **iterative** (GRU refinement, `slot_iters` steps).
  One-shot arms use the residual `slot_vectors = slot + update` (the pooled measure reaches the head
  while the query carries schema identity + seed — a uniform seed term otherwise cancels inside the
  over-slot softmax). `context = W_ctx(slot_vectors.reshape(B, K·d))`; `slot_vectors [B,K,d]` is kept
  accessible for a future TabQA / relational-tokenizer fold.
- **`measure`** — `n_pma` learned **sigmoid-gated** queries (not softmax over queries — that keeps it
  count-aware), same measure channels, no grouping. Isolates *counting* from *grouping*.

Config: `readout, slot_mode, slot_group_key, slot_iters, slot_gamma_init, slot_gamma_learnable,
slot_seed_cond, readout_channels`. Train-time diagnostics (`readout.last_diag`): learned $\gamma$,
per-element assignment entropy, per-slot utilization (dead/dominant-slot detection).
