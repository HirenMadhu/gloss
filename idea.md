# Mixture of Relational Experts (MoRE) on the Relational Transformer

*Design rationale. Pair with `implementation.md` for the build.*

---

## 1. Thesis

We add a Mixture-of-Experts layer to the Relational Transformer (RT, [arXiv:2510.06377](https://arxiv.org/abs/2510.06377)) whose **router conditions only on a cell's relational metadata — its schema embedding, its modality, and its causal recency — while the experts transform the cell's evolving content.** This single asymmetry ("route on semantics, transform the content") gives us soft, data-dependent parameter sharing across relations that is *temporally leak-free by construction* and *transfers to unseen schemas for free* — neither of which the existing message-passing graph-MoE methods (GMoE, HER, HOPE) can do, because none of them sit on a substrate like RT.

The contribution is **conjunctive and specific**, not "first MoE for relational data." The individual ingredients (top-$k$ gating, semantic routing, orthogonality-regularized balancing) exist in prior art. What does not exist is their composition on a **temporal, multi-modal, schema-agnostic relational transformer**, with a routing signal that is provably causal w.r.t. the RelBench seed time.

---

## 2. The one observation it's built on

RT's input token for a cell $(v, c, t)$ = (value, column, table) is an **additive** decomposition (this is literally how `rt/model.py` builds it):

$$x_i \;=\; \underbrace{W_d\, r_i}_{\text{value component }v_i} \;+\; \underbrace{W_{\text{col}}\, E_{\text{LM}}(c_i)}_{\text{schema component }s_i}$$

- $r_i$ is the datatype-specific normalized value encoding (numeric/boolean: $(v-\mu_c)/\sigma_c$; datetime: globally normalized; text: a frozen text-encoder embedding).
- $E_{\text{LM}}(c_i)$ is a **frozen language-model embedding of the column name** ("price of product"), 384-dim (MiniLMv2). In the reference code this is `batch["col_name_values"]`, projected by `enc_dict["col_name"]`.

So inside every RT token there is already a vector $s_i$ that says *what kind of cell this is*, living in a semantic space where "price of product" and "amount of transaction" land near each other **across databases**. RT also already carries a per-cell modality tag, `batch["sem_types"]` $\in \{\text{number}, \text{text}, \text{datetime}, \text{boolean}\}$.

The method is just: **build the router's input from $s_i$ and the modality (and recency), never from the value and never from raw table/column identity.** That small choice dissolves three hard problems at once (§5).

---

## 3. The method

**Relational signature (per cell, value-free):**

$$z_i \;=\; \mathrm{RMSNorm}\Big(\, W_s\, E_{\text{LM}}(c_i) \;+\; \psi(\sigma_i) \;+\; \phi(\Delta_i) \,\Big)$$

- $W_s E_{\text{LM}}(c_i)$ — projection of RT's frozen-LM column embedding ("what relation/column is this", semantic & schema-transferable).
- $\psi(\sigma_i)$ — learned embedding of the modality $\sigma_i$ (the multi-modal axis).
- $\phi(\Delta_i)$ — recency encoding, $\Delta_i = T_{\text{seed}} - \tau(\text{row}_i) \ge 0$ (the temporal axis). *Optional in v1 — needs one data-pipeline addition; see `implementation.md` §10.*

**MoE layer (replaces RT's SwiGLU FFN inside each block).** A shared pool of $M$ FFN experts, each identical in form to RT's FFN, with a sparse top-$k$ gate driven by the signature:

$$G_i \;=\; \mathrm{softmax}\big(\mathrm{TopK}(W_g\, z_i,\; k)\big), \quad k \in \{1,2\}, \qquad y_i \;=\; \sum_{j \in \mathrm{TopK}} G_{i,j}\, E_j\big(x_i^{(\ell)}\big)$$

The crucial asymmetry: **the router sees only $z_i$ (metadata); the experts $E_j$ transform the full evolving hidden state $x_i^{(\ell)}$.** Because $z_i$ is static per cell, it is computed once in the trunk and reused at every layer — routing is cheap, and the causality argument (§5) is trivial.

**Balancing — tail-tolerant, not uniform.** Drop the Switch/GShard auxiliary loss (its minimizer is *uniform* expert usage, which actively fights the long-tailed relation frequencies of real databases). Instead let usage follow the tail and prevent expert collapse with an orthogonality regularizer on the router's expert directions (the HOPE mechanism, transplanted to the router matrix):

$$\mathcal{L} \;=\; \mathcal{L}_{\text{task}} \;+\; \lambda \,\big\lVert \hat{W}_g^{\top}\hat{W}_g - I \big\rVert_F^2, \qquad \hat{W}_g = \text{column-normalized } W_g$$

Expert-Choice routing is the drop-in alternative when a hard compute cap is required (balance-by-construction, no aux loss).

**That is the entire method:** *a mixture of FFN experts routed on each cell's (schema, modality, recency) signature, with orthogonality-regularized tail-following balance.*

---

## 4. Why this is the soft-basis story done right

This is the input-conditioned generalization of RGCN's basis decomposition $W_r = \sum_b a_{rb} V_b$ — but the interpolation is governed by **semantic similarity** instead of relation identity. Two limits make it precise:

- **Hard top-1, one expert per column, routing on identity** $\Rightarrow$ recovers per-relation weight matrices (the heterogeneous-GNN limit).
- **Soft routing with $M \ll \#\text{columns}$ on the semantic embedding** $\Rightarrow$ soft sharing where semantically-similar columns share experts: exactly $a_{rb} \to G_b(z_i)$, except the sharing structure is grounded in the frozen-LM space and therefore **transfers across schemas**.

That last clause is the thing RGCN's identity-indexed coefficients can never give you, and it is what makes the foundation-model claim coherent.

---

## 5. The properties, and exactly how we differ from prior art

**Heterogeneity without type-overfitting — solved without masking.** The known failure (HER): if the router sees type/identity, it learns a trivial type→expert partition and the MoE just re-derives a heterogeneous model. HER fixes this by *stochastically masking* the type embedding. We fix it differently and deterministically: the router **never sees identity** — only the semantic embedding — and the **expert bottleneck** $M \ll C$ makes type-partitioning impossible, forcing the router to merge columns. Routing on $s_i$ ensures the merges are semantically sensible (a monetary-numeric cluster, an id-categorical cluster, a free-text cluster). *Different mechanism from every heterograph-MoE paper.*

**Temporal correctness — by construction.** Every component of $z_i$ is measurable from cells with $\tau \le T_{\text{seed}}$: $s_i$ and $\psi$ are static, $\Delta_i$ uses only the cell's own past timestamp, and **no global structural statistic (degree, neighborhood density) ever enters the router.** This is the piece the entire graph-MoE literature ignores: GMoE/HER/HOPE-style structural router features are computed on the full graph and *leak future edges* in a seed-time setting. Note part of this safety is *inherited* — RT's context is already a causal BFS — but our explicit recency axis converts mere correctness into temporal *specialization* (recent-activity vs long-range-history experts). A concrete, checkable consequence: with signature routing, a cell's gate is a pure function of its own $(c, \sigma, \Delta)$ and is **identical regardless of which neighbors got sampled** — a unit-testable invariance (see `implementation.md` §8).

**Schema transfer — free.** $z_i$ references no dataset-specific IDs, so $G(z_i)$ is defined on any unseen schema; an expert tuned to a semantic region fires on a new database's columns that embed there, no retraining. This matches RT's leave-one-database-out zero-shot protocol. Among surveyed methods only MoEMeta and HOPE gesture at schema-invariant routing, and neither uses a frozen-LM schema space.

**Multimodality** is the $\psi(\sigma_i)$ term; **sparsity / conditional compute** is the top-$k$; **soft relational sharing** is §4.

**Deliberately deferred: structural / receptive-field routing.** The honest structural features (degree, neighborhood density) are exactly the ones that leak, and RT already does multi-hop via stacked neighbor-mask attention — so the basic method does not need to own receptive fields. The clean extension is MoA-style routing over RT's four attention masks (column / feature / neighbor / full), or routing on the hidden state (safe here because RT's context is causal). Both stay out of v1.

**Compact contrast.** vs **Switch/GShard** — relational signature instead of a learned-from-scratch gate, tail-following instead of uniform balance. vs **GMoE** — FFN-on-cell-tokens routed on semantics, not per-node hop-GNN experts. vs **HER** — bottleneck + semantic routing instead of type masking, on RT tokens not HGT states, plus temporal and transfer. vs **HOPE** — borrows its orthogonalization but routes on transferable frozen-LM semantics (not learned intra-dataset prototypes) and is a full FFN-MoE, not a prediction head. vs **RGCN** — input-conditioned and schema-transferable, not fixed identity-indexed.

---

## 6. Novelty positioning (for the meeting, one sentence)

> *We route a shared pool of relational experts on each cell's frozen-LM schema embedding, modality, and causal recency — giving soft, semantically-grounded parameter sharing across relations that is temporally leak-free and transfers to unseen schemas by construction, none of which the existing message-passing graph-MoE methods can do.*

The defensible differentiators, in decreasing order: (1) temporal-causal routing (absent from all prior graph-MoE, forced by RelBench); (2) the RT transformer substrate (MoE at the FFN / over relational attention masks — a structurally new injection surface vs. all message-passing graph-MoE); (3) column-multimodality routing; (4) schema-invariant zero-shot transfer.

---

## 7. What would falsify it (honest risks)

- **The RGCN null result.** RT's frozen-LM token may already let a *single* shared FFN absorb all columns, in which case a small semantically-routed expert pool just recovers dense RT. **This is the primary thing the first experiment must rule in or out** (the routing-signal ablation: signature vs. value vs. identity vs. dense).
- **Value-free routing too coarse.** If the *value* genuinely should change processing, the signature is blind to it. Mitigation: append a learned low-rank slice of $v_i$ to $z_i$.
- **Boundary flips.** Top-$k$ on a smooth semantic space can flip experts at cluster boundaries; $k{=}2$ smooths it, Soft-MoE-style convex routing is the stable-but-costlier fallback.
- **Sparsity is notional under sampled contexts.** Per-token routing scatters experts across a minibatch, so the FLOP win is real only at scale or with capacity-based grouping; the MVP uses dense expert combination for correctness.