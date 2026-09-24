# C++ optimization + bug audit (read-only, Windows tree @ 66ae8c7 + docs)

Method: read-only inspection of the serve hot path
(`text.cpp` forward/sample → `engine.cpp` step → `generation_service.cpp`
pump → `hook_loop`/`scheduler`). No builds, no edits, no GPU use.
Line numbers are the Windows tree at commit `66ae8c7` plus uncommitted
README docs. Nothing below was executed — every item needs its listed test.

## Optimizations (highest impact first)

### O1. Batch the per-row sample loop — text.cpp:803-810
STATUS 2026-09-24: IMPLEMENTED uncommitted (binary f874864c, build-clean,
no GPU yet) — audit premise corrected: no new kernel needed, the batched
entry already exists (decode.cpp:55). sample_decode_rows now makes ONE
ops::sample call over all decode rows; B>1 broadcasts the shared serve
config via N tiny D2D copies into a work_-arena array (no new device code;
new kernel rejected — .cpp files carry no __global__ in this tree). RNG
keys are row-index-independent per the op contract; serve greedy has zero
per-row side effects (penalties 0, counts null) → bit-identical by
construction. Owed on GPU: T=0 ids solo+conc + spec-on frozen before landed.
PROVEN 2026-09-24 (binary f874864c): 8/8 conc==solo + spec-on frozen
(Paris F=19 [6511 314 9338] accept 3, Italy [6511 314 14898] accept 3,
EOS-through-commits). Bit-identical as constructed — O1 may land.
What: the stochastic branch of `sample_decode_rows` loops over decode rows
on the host, building 3 slice views and launching `ops::sample` per row.
At cap 8 that is 8 launches + 24 view constructions per step, serialized.
Change: one multi-row sample entry (rows × domain in, rows out) against the
same device config; keep per-row logical positions so shared-seed draws stay
independent.
Risk: sampling semantics — a batched kernel must reproduce per-row streams
exactly, or temperature≠0 output drifts.
Test: T=0 id equality solo+concurrent, then fixed-seed temperature run:
identical token ids before/after.
Impact: 8/10 — removes the only per-row CPU loop on the decode step.

### O2. Kill the per-step host positions upload — text.cpp:797-802
What: `host_positions` vector (n_dec) is filled on the host and H2D-copied
every step, but its content is always the iota `first_col + i`.
Change: device-side iota/fill into the preallocated positions tensor, or a
persistent buffer updated only when `first_col` changes.
Risk: near zero (pure function of two ints); the copy call itself is the
only thing that can break.
Test: T=0 determinism + temp≠0 fixed-seed equality.
Impact: 3/10 — small bytes, but it is a host round-trip on every step.

### O3. Overlap the token D2H sync — text.cpp:814-817
What: `cudaStreamSynchronize` blocks the pump every step until decode tokens
land on the host.
Change: double-buffered pinned token staging — sync buffer N-1 while step N
launches (prefill work does not need decode tokens).
Risk: pipeline hazards — a stale buffer read corrupts a whole step silently.
Test: T=0 id equality across solo/concurrent + latency delta (this must show
a win or be reverted).
Impact: 4/10 if it wins, 0/10 otherwise — measure, do not assume.

### O4. Cache env-gate lookups — ladder_trace.h:28, text.cpp:822
What: `ladder_armed()` (`getenv`) runs per `ladder_dump` call (~130/step
when off); `NINFER_LOGITS_DUMP` + `NINFER_SERVE_STEP_TRACE` re-query per step.
Change: resolve once into function-static / atomic flags at first use.
Risk: env changed mid-run stops being honored (we only set these at boot).
Test: boot with each flag on and off; confirm traces appear/disappear.
Impact: 2/10 — ~130 linear environ scans per step eliminated.

### O5. Index req/state lookups — scheduler.cc:38-56, hook_loop.cpp:187-202
What: `find_request*`/`find_state_by_seq` are linear scans; 
`erase_states_for_reqs` is O(finished × states) full-map scans per step.
Change: `unordered_map<req_id,…>` / `unordered_map<seq_id,…>` alongside the
vectors (single-seq requests keep the vectors as iteration order).
Risk: index invalidation on every insert/erase path — a stale index returns
the wrong request, which is worse than slow.
Test: concurrent admit/finish churn test + two-curl coherence.
Impact: 2/10 at cap 8; grows if the cap is ever raised.

### O6. Reuse publish scratch — generation_service.cpp:556-599
What: `decoded`, `finished`, `publish` vectors are constructed per pump
iteration; per token there is a fresh `std::vector<TokenId> one{token}` plus
two `std::string`s (`piece`, `deposit`).
Change: hoist vectors out of the loop with `clear()`; thread-local static
scratch for the single-token detok input.
Risk: detok statefulness across reuse (see B4); leftover capacity is harmless.
Test: streamed-text equality + long-run stability.
Impact: 2/10 — allocator traffic off the per-token path.

### O7. Gate per-step validation for release — text.cpp:740-761, 951-962
What: plan/batch cross-checks and 8× `require_tensor_shape` run on every
step in all builds.
Change: keep unconditionally in debug, compile-gate the O(seqs) loops behind
`NDEBUG` for release (keep the null/empty checks, they are free).
Risk: losing a live invariant trip that once caught a real defect (these
checks earned their keep during the soup era).
Test: two-curl coherence + a fault-injection run (corrupt offsets in a test)
with gates on and off.
Impact: 1/10 — small, and the safety they buy is real. Lowest priority.

### O8. (Pointer, not a new finding) 64-layer × N-row launch fan-out
Already tracked in README Known-skips §6 (fused single-M attention). The
per-row slices inside the ragged attention entry are the biggest structural
cost on the step; O1–O6 are change next to it. Do not start outside that ticket.
NIGHT VERDICT 2026-09-23: ticket ran (2b fused pure-decode, NINFER_FUSED_ATTN=1
default-off, binary 9cd510b5): quality held 8/8 both flags, speed UNMOVED
(B=2 lost, B=4 +6% noise — decode is EXL3-math-bound). Parked. O1–O6 remain
change next to the ~370 EXL3 launches, not the 16–64 attn.

## Suspected bugs (filed, NOT fixed)

### B1. Serve silently serves greedy for temperature≠0 — engine.cpp:340-352,562
`serve_sampling_` is hardcoded (`temperature = 0.0F`, `top_k = 20`) at
construct and reasserted on every step (`card_->set_sampling(serve_sampling_)`
at :562). The per-request `resolved_sampling` in `generation_service.cpp`
never reaches the card on the serve path. If the HTTP layer accepts
`temperature ≠ 0`, the response is greedy anyway with no error or warning.
Either plumb per-request sampling to the card or reject nonzero temperatures
at validation. Verify with one temperature=0.7 request vs greedy.

### B2. Unbounded scheduler waiting list — scheduler.cc:18-28
`submit()` always enqueues; nothing caps `waiting_`. Admission to running is
gated by pages (`throw_if_pages_exhausted` is wired at
generation_service.cpp:418), but a flood of arrivals grows host memory
without bound. Cap pending or shed with 429 at submit.

### B3. State-leak path in seq_states_ — hook_loop.cpp:192-202
States are erased only for req ids present in the `finished` list, via a full
map scan. Any path that drops a request without listing it there (abort,
expire, exception between `on_step_done` and erase) leaks its state
(and its text buffer) forever. Audit every req exit path for erase coverage;
consider RAII/erase-on-last-reference.
NIGHT VERDICT 2026-09-23: CONFIRMED REAL — aborted reqs (pool-exhaustion;
evict_done drops them from scheduler vectors) never reach `finished`, and
states register at drain-before-admit so even aborted waiters leak. FIXED
uncommitted (scheduler.cc on_step_done now lists every done req in running_
+ waiting_; erase is idempotent; own pump breaks on aborted own req). Needs
the churn test on GPU before it counts as landed.
GPU PROOF 2026-09-24 (binary e1f602d0 = eafcd04): 8/8 green + SSE clean +
spec-on frozen (F=19 accept 3, EOS-through-commits, accept-2 rewind) — B3
code inert on all gate paths as predicted. Churn attempted: 10×ctx2k conc
(all served) + oversize (clean http400) + 8×6.8k-ctx (4 served, 4 clean
http503 at admission) — server healthy, post-churn 8/8 green, ZERO
abort/exhaust/evict lines. The abort branch never fired: the scheduler sheds
at admission before mid-step grow can fail. Fix stands on path review +
regression-green; live-fire of the branch itself needs fault injection
(a later ticket, not a gate).

### B4. Unbounded per-request text growth — generation_service.cpp:~598
`target->text += piece` accumulates the full completion in memory with no cap.
Fine at 24 tokens; at 262k contexts this is a large realloc-churned string
per request. Cap or stream-and-drop when the client only needs deltas.

### B5. Hardcoded top_k=20 inside the "explicit argmax" — engine.cpp:342
Same construct as B1: if temperature 0 resolves to a top-k-truncated argmax
rather than a full-vocab argmax, T=0 output equals top-1 only when the top-1
token survives truncation — true in practice, but the truncation is load-
bearing and undocumented. Assert or comment the assumption.
NIGHT VERDICT 2026-09-23: DOCUMENTED (engine.cpp comment: truncation is
load-bearing, frozen coherence is the evidence, top_k=0 is the remedy).
B1 itself re-confirmed OPEN (temperature forced 0 at engine.cpp:464-476,
per-request sampling at generation_service.cpp:428 never reaches the card);
fix needs GPU (behavior change).

## Self-score

- Deepest find: B1 (8/10) — silent behavior violation, verifiable in one request.
- Best perf bet: O1 (8/10) — the only per-row host loop on the step.
- Honesty note: O4/O6/O7 are 1–2/10 microhygiene; listed so they are not
  rediscovered, not because they move the ~30 ms/token number. The step is
  dominated by 64 layers + lm_head (see README); nothing here claims otherwise.
- Overall audit value: 7/10 — one likely-real bug (B1), one solid optimization
  (O1), the rest are small truths. Nothing was executed; treat every line
  above as a hypothesis with a named test, not a conclusion.
