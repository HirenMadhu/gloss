"""Shared two-level phase presets (changes.md §5), imported by both array runners.

Factored out of `run_ablation.py` so `run_gridsearch.py` can use the identical switch sets — if these
drifted between the two runners, a grid-search "winner" would not be the same architecture as the
headline run, which is the kind of mismatch that silently invalidates a comparison.

Each preset flips ONE group of switches relative to the previous, so a delta stays attributable:
  phase0a — row tokens ONLY; the cell level behaves exactly like RT (the isolation gate)
  phase0b — + collapse the four masked cell attentions into one
  full    — the proposed design: cell RoPE, row biases, MoE at BOTH levels (row experts shared+routed)
"""
TWO_LEVEL_PHASES = {
    "phase0a": dict(cell_attention="four_mask", cell_rope_time=False, time_mode="buckets",
                    pool_query="mean", role_bias="none", time_bias="none", row_ffn="dense",
                    broadcast="additive", head_mode="row_token"),
    "phase0b": dict(cell_attention="full", cell_rope_time=False, time_mode="buckets",
                    pool_query="mean", role_bias="none", time_bias="none", row_ffn="dense",
                    broadcast="additive", head_mode="row_token"),
    "full":    dict(cell_attention="full", cell_rope_time=True, time_mode="rope",
                    pool_query="hybrid", role_bias="name_derived", time_bias="rope", row_ffn="moe",
                    row_use_shared=True, broadcast="additive", head_mode="row_token"),
}
