# Ticket: F-as-data attention backend → reopen verify graphs

Status: PHASE A DONE 2026-09-24 (GPU-gated, committed). Phases B/C still parked.

## Why this ticket exists

The ver/M=4 frozen exec was closed as invalid (row 7): MTP verify passes
`venv={1,F+4}` and F bakes into the full-attn layers three ways — route
(Prompt/SmallT/ChunkedSmallT flips with `max_visible_keys`), grid (small_t
splits/`target_ctas`/`page_limit` derive from it), window
(`logical_capacity`/`implementation_window` + CTA window cull attend wrong KV
= wrong tokens). Field sweep confirmed the invariant: everybody graphs fixed
shapes with F as device data (seq lens / positions / page tables); nobody's
attn backend makes host-side launch decisions off F. Our envelope does. So the
rework below is a *backend project, not engine.cpp* — only after it lands does
reopening verify graphs make sense.

Standing product (unchanged by this ticket): spec-off body graph ~14 ms/tok
(default-off), MTP eager on purpose, blanket disable with notice.

## Phase A — attention rework: F stops entering launch params

Goal: identical numerics, F-invariant launch decisions.

1. **Route by (width, batch) only.** `causal_attention_resolve_route`
   (`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`):
   remove the `envelope.max_visible_keys` terms (Prompt `prompt_limit`
   comparison, `prompt_visible_keys` comparison). Route becomes a pure function
   of (q_heads, width, batch, storage).
2. **Grid by capacity.** `causal_attention_split_capacity` /
   `causal_attention_chunk_tokens` (same file + `launch.h`) and the small_t
   launch path (`small_t.cu`: `causal_small_t_launch_capacity`,
   `target_ctas`, `page_limit`): size by static capacity / geometry, never by
   the F-derived window. The `include(min)..include(max)` interval-maximum
   policy must go; replace with the capacity bound.
3. **Mask by position.** Kernels keep correctness from positional data
   (`position_batch`, `valid_columns`, device block tables) —
   `logical_capacity` / `implementation_window` stop meaning "F+4". The CTA
   window cull (`small_t.cu` window check) must cull on positional bounds, not
   the baked envelope.
4. Same treatment for the early-F Prompt path (`prompt.cu`) if it reads the
   envelope for sizing.

Bar A (all eager, no graphs yet):
- DONE 2026-09-24, GPU-gated. Solo Paris/Rome/FOX coherent + deterministic
  (rerun-stable); spec-off eager 28.2/27.8 ms/tok vs 30.1/27.6 baseline (flat);
  spec-on warmup passes, canonical Paris chain 6511/314/9338/369 accept-3 +
  accept-1 rewind lines present; splits=85 constant over 3706 SmallT calls
  (widths 1+4, F 19..57) — route/grid F-invariant at runtime.
- Accepted deviation (filed, pre-existing): conc-vs-solo trailing-token
  divergence reproduces on the PRE-Phase-A binary too (old: P16/R16/R24 False;
  new: R only) — batch-composition reduction order, not a Phase-A regression.
  Solo old-vs-new differs only at near-tie trailing tokens (early-EOS vs
  '\n\nuser' template bleed; new outputs cleaner, FOX-64 fully coherent).
- One self-inflicted bug found+fixed in-ticket: the first static-max edit left
  the `window in [min,max]` guard in `causal_small_t_launch_capacity::include`,
  zeroing splits for tight verify envelopes ({1,F+4} rejects every static end
  128..262144) → splits=0 → Tensor-ctor FATAL. Fix: guard removed, capacity is
  the static max over all ends. Lesson: when de-Fing a function, grep its
  closures for envelope reads too.
- No eager regression: 28.2/27.8 vs 30.1/27.6 ms/tok (flat; bigger grids at
  small F are the known cost — reported, not hidden). BF16 (the gated model)
  never reads `implementation_window`; INT8 partials now always take the
  big-window variant config (same partial math, occupancy-only difference —
  ungated, no INT8 KV model on hand, correctness from positional bounds).

## Phase B — borrow ORT snapshot/restore (do it here, not later)

Portable idea from the ONNX Runtime GenAI MTP doc, worth stealing as part of
this ticket: snapshot/restore of the GDN recurrent state that **preserves
buffer addresses** (in-place copy-back, never realloc), because CUDA-graph
replay requires stable addresses.

- Applies to the spare/shadow GDN slot machinery on the MTP path
  (`pool_->copy_slot`, shadow slots in `step_mtp_decode`).
- Rule: restore writes back into the same device buffers the graph captured;
  any path that reallocates or rebinds state addresses is a graph-break and
  must fail closed (park the exec dead, fall back eager — same pattern as the
  serve seam's `graph_mark_dead`).
- Bar B: reject/rewind telescope identical with and without the new
  snapshot path (accept-3 frozen chain 6511/314/9338/369 + accept-2 rewind);
  address-stability check in verbose mode (pointer audit like `[gptr]`).

## Phase C — reopen verify graphs (only after A + B are green)

- Frozen execs keyed by shape, never `update()` across keys: `dec/M=1`
  (existing serve seam) + new `ver/M=4` wrapping `target_verify_batch`
  in `step_mtp_decode`. Uploads / syncs / D2H stay outside capture.
- WSL rule still applies: capture records but never executes eagerly here —
  replay immediately after instantiate for that step's outputs.
- Keep the program-level MTP blanket (`program_impl.cpp`) until THIS phase's
  bar is green; lifting it early re-FATALs startup via `prepare_graphs`.
- Bar C (the original ticket bar): spec-on starts without `--no-cuda-graph`
  and no FATAL; Paris drafts 6511/314/9338 accept 3 + rewind; spec-off
  GRAPH=1 still ~14 ms/tok; spec-on beats eager ~45 ms/tok or MTP stays
  graph-off per bar-4; 8/8 spec-off matrix.
- No default flip in the same commit as any of this.

## Must not (whole ticket)

- No fusion kernels. No KVarN / DFlash2 / MTP-10. No fused-attn work.
- No per-F exec cache (rejected: an exec per token of context is a leak).
- No pinning the envelope to fake F-invariance (load-bearing for correctness:
  tight envelopes keep drafts off unwritten columns).
- No `cudaGraphExecUpdate` across shapes — instantiate per key or skip replay.
- No mixed-prefill graphing. No accept/rewind logic changes.

## Work order

A → B → C, staged commits, one GPU question at a time. Stop lines: if Bar A
fails numerics, stop (no B/C); if Bar B fails rewind equality, stop; if Bar C
fails bar-4, MTP stays graph-off and this ticket closes with A+B kept.
