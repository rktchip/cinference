# Task 12 — Chunked-prefill interleave (design)

Status: DESIGN ONLY 2026-09-28. No GPU, no servers, no timing in this lane.
Parent runs the value legs (all TBD numbers below). Flag-gated, default off.

## 1. Problem (measured inputs, parent-owned)

- S3 split at conc-8: prefill dominates — ~70% of wall spec-off, ~90% MTP-bypass.
  Absolute prefill share: **~7.5 s spec-off / ~20 s bypass** (conc-8 window).
- Decode batching is healthy: short-prompt conc-8 runs B=8 steps (~33 ms)
  vs ~13 ms single — decode is NOT the target.
- S3 stall probe (task-12): admitting a p8k prompt stretches a running p2k
  stream **5.1 s → 8.3 s (+63%)**. No prefill interleave exists today: while a
  new request's prefill advances, running decode streams stall.

## 2. Where prefill currently blocks (serial path)

1. **Single staged-prefill lane (legacy/CLI path).**
   `src/runtime/engine/engine_core.h` — `run_prefill_step()` (~line 1401):
   one request owns staged prefill via `scheduler_.prefill_lane()`; any other
   request's decode waits. `set_legacy_single_lane()` / `legacy_single_lane_`
   (~line 282) gates who may claim it; the batch path must never claim it
   (B2 lane gate, ~line 1563-1573).
2. **Serve pump serializes steps under one lock; prefill-only steps decode
   nothing.** `src/serve/generation_service.cpp` (~lines 580-696): inbox
   drain → `schedule_step` → `dispatch_step` → `run_batch_step` →
   `on_step_done` is one `pump_mutex_` critical section, and "a prefill-only
   step decodes nothing while the prompt advances" (~line 691). Whenever the
   plan carries prefill without decode rows, every running stream idles.
3. **Scheduler chunk budget exists but is FCFS + strict FIFO.**
   `src/batch/scheduler.cc` — `schedule_step()` (~line 170) mixes one
   chunked-prefill slice (capped at `chunk_tokens_`) with one token per
   decode row, but the budget is first-come-first-served in `running_` order
   and `admit_waiting()` (~line 164) breaks on a blocked head. `admit_one()`
   (~line 119) reserves the FULL prompt blocks + one decode slot up front, so
   a p8k admit lands as one large prefill claimant that contends the budget
   for many consecutive steps instead of yielding fairly to running decode.
4. **One TextContext forward per step carries whatever the plan holds.**
   `src/runtime/engine/engine.cpp` — `Engine::run_batch_step` (~lines
   880-1010): prefill rows and decode rows share a single forward
   (`step_decode_layers`), so the fix belongs in plan construction
   (scheduler), not execution — execution already handles mixed rows.

## 3. Proposed chunk scheduler

- **Chunk size: start at 1024, shared with rung-2.**
  1024 is the existing default in three places (`src/serve/hook_loop.h:98`,
  `src/serve/hook_loop.cpp:48`, `src/serve/serve_options.h:36`
  `prefill_chunk = 1024`) and the width rung-2's batched MTP mirror
  (`TextContext::mtp_prefill_chunk`, `text.h:281-284`, `text.cpp:465`) is
  built against (M=N pass is API-legal at `T in [1, prefill_chunk]`).
  Same width ⇒ the S7 per-slice GDN snapshot (§5) stays 1:1 with slices and
  no width-dependent code is added. Drop to 512 only if the mixed-step gate
  (§6, gate 1) fails — smaller chunks bound mixed-step latency but double
  the MTP-fill seedings, so 1024-first is the cheap experiment order.
- **Budget rule (decode preservation):** per mixed step, cap
  `prefill_tokens` so decode rows are never starved:
  (a) decode rows always ride (unchanged — every runnable decode seq is
  planned every step); (b) at most ONE prefill slice per step beyond the
  running streams' own slices, i.e. a newly admitted long prompt yields to
  in-flight prefill slices round-robin instead of holding the budget for
  `ceil(8192/1024) = 8` straight steps; (c) admit of a prompt longer than
  `2 × chunk` while any stream is decoding is deferred slice-wise, never
  row-wise — the request is admitted (KV reserved) but its slices interleave
  one-per-step with running decode.
- **Reuse, not new machinery:** `StepPlan`/`PrefillSlice`
  (`src/batch/scheduler.h:26-53`), `AssembleBatch`
  (`src/batch/batch.cu:8+`), `EngineHooks::dispatch_step`
  (`src/batch/cinference_hooks.cc:31`) already carry mixed plans end to end.
  The change is confined to `RequestScheduler::schedule_step` slice
  selection + an interleave counter per seq. `IsGraphSafe`
  (`scheduler.h:58`) already routes mixed steps eager — no graph interaction.

## 4. KV / catch-up interplay (merged resume code)

Resume (row-23 task 8, merged default-off; `engine.cpp:230-250`,
`NINFER_MTP_TOGGLE_AFTER`/`NINFER_MTP_TOGGLE_OFF`, `catchup_store_`,
`ServeSlot::{stage_start, staged_n, staged_ids/pos, staged_overflow,
mtp_valid_pos, anchor_valid}` ~lines 730-738, `mtp_resume_catchup()` ~line
838):

- **Known conflict:** on any mixed step, decode rows that ride along have
  their staging invalidated — `engine.cpp:976-987`: prefill lanes get
  `staged_n = 0` + `stage_start = mtp_valid_pos`, decode rows get
  `staged_overflow = true` (resume fails closed to the ordinary path).
  Interleave makes EVERY step mixed for the duration of a long admit, so a
  resume window overlapping a p8k admit can never complete staging.
- **Rule:** while interleave is active (any admitted-but-incomplete prefill
  longer than one chunk), suspend resume staging (`mtp_stage_decode_hidden`
  skipped; lanes hold `mtp_valid_pos`/anchor) instead of recording
  unrecoverable coverage. Resume then fails closed for exactly the admit
  window and re-arms after — no silent ordinary-path fallback mid-window,
  no `staged_overflow` poisoning from steps the toggle did not cause.
- Prefill lanes keep the existing per-slice reset (976-983); chunked
  prefill already advances `computed_len` per slice
  (`scheduler.cc:195-216`), so each chunk boundary is the natural
  re-arm frontier.

## 5. GDN recurrent-state handling for partial prefixes

- Engine owns one GDN slot per serve lane (`engine.cpp:187-192`); prefill
  advances lanes in place, destroying pre-slice states — which is why S7
  snapshots each prefill lane pre-forward via `pool_->copy_slot`
  (`engine.cpp:951-954`).
- Chunk k+1 must source GDN from chunk k's published destination slot
  (`TextContext::set_linear_state_slots`, `text.cpp:286`;
  `GdnStateAction::UpdateInPlace` is width-1 outside the batched verify,
  `text.cpp:1677`). Interleave does not change this chain — slices of one
  seq still execute in order — but the chain now spans steps shared with
  foreign decode rows, so the per-chunk contract must be: **KV blocks +
  GDN state published atomically per chunk** (existing
  `advance_computed` + `pool_.publish_row` in `on_step_done`,
  `scheduler.cc:195-216` already does this; keep it invariant, never batch
  publishes across chunks).
- MTP-fill seeding (`mtp_prefill_fill`, `engine.cpp:1651`) stays per-slice
  from the S7 snapshots; at 1024-wide chunks the snapshot count per p8k
  admit is 8 either way — serial or interleaved — so fill cost is unchanged,
  only its steps are spread across mixed forwards.

## 6. Gates (values first; parent measures)

| # | Gate (value first) | Method |
|---|---|---|
| 1 | **Running p2k streams keep ≥80% of undisturbed tok/s during a p8k admit** (proposal; kill below). Today: +63% wall stretch (5.1→8.3 s). | conc-8 soak: 7× p2k decoders steady-state, admit 1× p8k, per-stream tok/s vs no-admit baseline. |
| 2 | **New p8k request TTFT ≤ 1.2× today's serial-admit TTFT** (proposal). Interleave must not punish the newcomer beyond a bounded delay. | Solo p8k TTFT vs admitted-into-busy p8k TTFT. |
| 3 | **Mixed-step forward ≤ 1.5× pure-decode step at same decode membership** (proposal). Bounds the per-step tax decode pays per chunk. | `NINFER_SERVE_STEP_TRACE` step log or nsys; compare M=(1024+8) vs M=8. |
| 4 | **Byte-identical outputs** on frozen prompts (Paris/Rome-style prefix check) with flag on vs off. | Existing verify-replay / prefix-exact harness. |

TBD by parent value legs: undisturbed conc-8 tok/s baseline, serial p8k TTFT
baseline, mixed-vs-decode step ratio at 1024. If measured values miss the
proposals by >2×, rescale the proposal — do not silently pass.

## 7. Predicted effect (from S3 splits)

Conc-8 prefill share **~7.5 s spec-off / ~20 s bypass** is today paid twice:
once in the newcomer's TTFT, once as stalled decode in running streams
(+63% probe). Interleave does not shorten total prefill work — 8k tokens
still cost 8k tokens of forward — but converts it from head-of-line stall
into shared steps: running streams keep emitting every step (gate 1), and
the newcomer's TTFT stretches only by the decode-preservation cap (gate 2)
instead of jumping the queue. Expected shape post-fix: p8k-admit running
p2k wall **8.3 s → ≤6 s** (≤20% over undisturbed 5.1 s); conc-8 prefill
share stays ~constant in absolute seconds but overlaps decode instead of
blocking it. Bypass (~20 s share) gains proportionally more, since its
prefill fraction is larger.

## 8. Kill threshold

**Kill if gate 1 fails after the 1024→512 chunk step-down:** running-stream
tok/s during a p8k admit still <80% of undisturbed baseline ⇒ revert the
flag to default-off and close task-12 as serialize-and-admit (document the
measured floor). Single criterion, no retuning loop. Any output divergence
(gate 4) kills immediately at any step.

## 9. Build order (flag-gated, default off)

1. **Step 1 — planner only (`NINFER_PREFILL_INTERLEAVE=1`, default off):**
   round-robin slice selection + one-slice-per-step cap for newly admitted
   long prompts in `RequestScheduler::schedule_step`; resume staging
   suspended while interleave active (§4). Gates 3+4.
2. **Step 2 — soak + thresholds:** parent value legs fill §6 TBDs; confirm
   gates 1+2 at 1024; step down to 512 only on gate-1 failure.
3. **Step 3 — default decision:** pass all four gates ⇒ propose default-on
   (separate sign-off, not this ticket); any kill trigger ⇒ revert to
   default-off, keep code behind flag, close with measured floor.
