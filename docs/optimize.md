# C++ audit pass 2: speed + consolidation (read-only, Windows tree @ 66ae8c7)

Scope: `src/` hot path only (forward -> step -> pump). Read-only: nothing
built, nothing edited, nothing measured. Line numbers are the Windows tree at
`66ae8c7` plus uncommitted docs. This pass EXTENDS
`docs/cpp-optimization-and-bug-audit.md` (O1-O8 speed, B1-B5 bugs) — items
already filed there are referenced, not repeated. Every item below needs its
listed test before anyone touches code.

## Consolidation targets (same behavior, fewer lines, one owner)

### C1. Two copies of the q/gate per-head deinterleave scatter (MERGE FIRST)
Where: `src/models/qwen3_5/execution/text.cpp:~1230-1260` (attn_mix) and
`src/ops/wrapper/attn_input_proj.cpp:~155-200` (EXL3 branch). Same comment
("NOT [q; gate] halves ... matching oracle exllamav3 deinterleave_qg"), same
geometry, two separate 2D-copy lambdas (`cudaMemcpy2DAsync` at text.cpp:1248
and attn_input_proj.cpp:192).
STATUS 2026-09-24: IMPLEMENTED uncommitted (binary ca8a26ed, build-clean,
no GPU yet) — new `src/ops/wrapper/qg_scatter.h::scatter_qg_heads`, both
sites call it, all guards kept, CUDA_CHECK everywhere (fail-fast wins over
the tagged throw). Owed on GPU: T=0 Paris/Rome ids + oracle layer-0/1
exactness before it counts as landed.
PROVEN 2026-09-24 (binary ca8a26ed): 8/8 conc==solo + SSE clean + spec-on
frozen (F=19 accept 3, EOS-through-commits, accept-2 rewind) — ids identical
with the merge. Oracle layer exactness still open (needs the exllamav3
oracle harness, not just serve ids).
Rewrite: one helper, e.g.
`detail::scatter_qg_heads(dst_q, dst_g, src, head_dim, n_q_heads, stream)`,
called from both sites. Deletes ~60-80 lines and — more important — kills the
divergence class that caused the interleave bug in the first place: two copies
of tricky geometry WILL drift.
Risk: the two call shapes may differ subtly (flats vs projection inputs);
diff both bodies line-by-line before merging. Medium risk, high value.
Test: T=0 Paris/Rome id equality + oracle layer-0/1 exactness (the same bar
that proved the original fix).
Saves: ~60-80 lines. Impact: 8/10 (correctness insurance, not speed).

### C2. Split text.cpp (1892 lines) by concern
Where: `src/models/qwen3_5/execution/text.cpp` — forward, sampling
(`sample_decode_rows` ~:790-830), ladder/debug dumps, MTP slices in one TU.
Rewrite: three TUs — forward, sample, debug/dump — keeping anonymous-namespace
helpers with their users. Zero behavior change; the win is incremental-build
time (editing sampling no longer recompiles the whole forward) and reviewability.
Risk: ODR/internal-linkage mistakes during the split (two helpers with the
same name in different TUs). Low risk if split is purely mechanical.
Test: full rebuild + two-curl coherence + think-split ctest.
Saves: 0 lines (moves them), but each file becomes reviewable. Impact: 5/10.

### C3. prefill.cpp (1248) vs decode.cpp (854): shared card prologue
Where: `src/models/qwen3_5/program/prefill.cpp` and `decode.cpp` — both set
sampling, state slots, and execution envelopes (e.g. prefill.cpp:47-49
`card.set_sampling` + `set_linear_state_slots`; decode path repeats the shape).
NIGHT VERDICT 2026-09-23: KILLED — no duplication exists. `configure_text_card`
(prefill.cpp:45) already owns the prologue and decode.cpp has ZERO
set_sampling/set_linear_state_slots calls (it routes through
ordinary_batch_body). Nothing to merge.
Rewrite: one `setup_card(card, sampling, src_slot, dst_slot, ...)` inline
helper in a shared header. Candidate only — the two bodies were not diffed in
this pass; confirm overlap first.
Risk: prefill and decode envelopes differ on purpose (sliding window, MTP
extent); a merged helper must take those as parameters, not assume them.
Test: two-curl coherence + MTP spec-off bars.
Saves: ~40-60 lines if the overlap is real. Impact: 4/10.

### C4. One per-step scratch struct (folds O2/O3/O6 together)
Where: `text.cpp:797-802` (host_positions), `:814-817` (host_tokens),
`generation_service.cpp:556-599` (decoded/finished/publish/one{token}).
Rewrite: a reused `StepScratch` (positions, tokens, publish vectors) owned by
the engine/pump, `clear()`ed per step instead of constructed. Single owner,
no per-step allocation, no scattered vectors to audit.
Risk: stale-content bugs if a clear() is missed — a whole step silently reads
last step's data. Mitigate with clear-at-top + debug-mode poison fill.
Test: streamed-text equality + long-run stability + ASAN/debug run.
Saves: ~30 lines of declarations. Impact: 5/10 (allocator traffic off step).

### C5. One sampling-config resolver (folds B1/B5 together)
Where: `engine.cpp:340-352,562` (hardcoded greedy) vs
`generation_service.cpp` `resolved_sampling` (per-request, never reaches card).
STATUS 2026-09-24: PHASE 1 IMPLEMENTED uncommitted (binary ca8a26ed,
build-clean, no GPU yet) — temperature≠0 now 400s at validation on BOTH
paths (openai_chat_request.cpp parse_sampling + anthropic_messages_request
range block, code temperature_not_supported); absent/null/0 still greedy.
Owed on GPU: greedy T=0 ids byte-identical + temp=0.7 returns 400.
PROVEN 2026-09-24 (binary ca8a26ed): absent/0/0.0 → greedy Paris identical;
0.7 + 1.0 → HTTP 400 temperature_not_supported. (Anthropic path code-read
identical; live 400 check owed on that path.)
Rewrite: single `resolve_sampling(request)` function; card reads from it every
step. This is filed as bug B1 — listed here because the fix IS a
consolidation: two sources of truth become one.
Risk: changes temperature!=0 behavior from (silently greedy) to (actually
temperature) — outputs WILL differ from today. That is the fix, but it must be
announced, not snuck in. Test: greedy T=0 ids unchanged; temp=0.7 differs.
Saves: ~15 lines. Impact: 7/10 (bug fix wearing a consolidation coat).

### C6. Indexed req/state lookup (folds O5)
Where: `scheduler.cc:38-56`, `hook_loop.cpp:187-202`. Linear scans +
O(finished x states) erase become `unordered_map` + single erase each. The
erase loops collapse from nested scans to direct lookups — shorter AND faster.
Risk: index invalidation on insert/erase (see O5). Test: admit/finish churn +
two-curl. Saves: ~25 lines. Impact: 4/10 at cap 8.

### C7. Gate ladder/debug traces once per step, not ~130 times
Where: `text.cpp` ladder_dump call sites (guarded inside by `ladder_armed()`,
`ladder_trace.h:28` getenv per call) + `NINFER_LOGITS_DUMP` + step trace.
NIGHT VERDICT 2026-09-23: DONE (ladder_trace.h: `ladder_armed()` now resolves
once into a function-static; the ~18 dump sites keep their guard as backstop).
The two text.cpp LOGITS_DUMP getenvs were already step-level — left alone.
Needs the flags-on/off boot check on GPU to close fully.
Rewrite: resolve all three flags once at step top into bools; pass down or
early-out. ~130 getenv-guarded calls become one gate.
Risk: near zero. Test: flags on/off produce traces or silence.
Saves: ~10 lines of repeated guard noise. Impact: 3/10.

### C8. WATCH ITEM: draft vs decode forward duplication (do NOT merge yet)
Where: `src/models/qwen3_5/program/speculative/mtp.cpp` (237 lines) +
engine MTP seam vs `decode.cpp`. If the MTP lane grows a second forward loop
next to the decode loop, that is the "second dispatch" the project rules
forbid — flag it at landing review. No action until the accept bar is met.

## New speed notes (not in pass 1)

### S9. Fused-QKV scatter: 64 layers x N slices of 2D D2D per step
Where: text.cpp:201-204 + :505-509 + :536-537 + the C1 scatters, times 64
layers. Each is correct and each is a separate launch. The structural fix is
the Known-skips fused single-M ticket — explicitly out of scope for this pass.
Noted so nobody "optimizes" one slice in isolation and claims a win.
NIGHT VERDICT 2026-09-23: that ticket ran (2b, NINFER_FUSED_ATTN=1 default-off)
and parked on no-movement; the corrected profile says the mlp (61% of step
weight traffic), not attention slices, is the lever. Impact if the ticket lands: 9/10. Impact of touching one slice: ~0/10.
Impact if the ticket lands: 9/10. Impact of touching one slice: ~0/10.

### S10. Block-tables upload every step -> device mirror + delta update
Where: engine.cpp per-step upload path (~:495-542). Tables change only on
admit/finish/evict, yet the full table uploads each step.
NIGHT VERDICT 2026-09-23: KILLED — superseded, no per-step full upload exists.
Serve uses on-device gather (`GatherBlockTablesKernel`, batch.h: "no host-side
O(nums×max_blocks) gather per step") and pool mutations publish rows directly
(`publish_row` after every grow). A dirty-bit mirror would duplicate machinery
that already landed. Do not revive.
Rewrite: persistent device mirror, re-upload only rows whose lease changed
(admit/finish set a dirty bit). Mostly a copy-size win at cap 8, grows with
context length (8k today, 262k ambition).
Risk: stale-table corruption is silent and total — needs a debug-mode
full-compare + the first-row-nonzero assert kept unconditionally.
Test: two-curl + long session with churn + concurrent admit/finish.
Impact: 4/10 today, 7/10 at 262k contexts.

### S11. lm_head micro-launch cap m=8 (already policy, note the ceiling)
Where: serve path lm_head launches. The cap serializes the single largest
matmul per step. Raising it is a VRAM-atomicity question (split-k atomicAdd
nondeterminism, see SPLIT_TARGET pin), not a code edit — do not touch without
the determinism harness from the freeze gate. No code change proposed.

## Suggested merge order (lowest risk first)

C7 -> C4 -> C6 -> C1 -> C5 -> C3 -> C2, with S10 after C1, and C8/S9 parked
on their existing tickets. C1 before C5 because geometry has one owner before
sampling changes behavior. Nothing merges without its listed test; C1/C5/C3
additionally require the frozen Paris/Rome re-run.
NIGHT UPDATE 2026-09-23: C7 done (pending flags boot check); C3 KILLED (no
duplication); S10 KILLED (superseded by device gather). Live order is now
C4 -> C6 -> C1 -> C5 -> C2. C5/B1 fix still needs its GPU behavior test;
C1 needs ids + oracle exactness; everything else on the list needs a GPU run.

## Self-score

- Best consolidation: C1 (8/10) — deletes the exact duplication class that
caused our most expensive bug; the line savings are secondary.
- Best new speed bet: S10 (4/10 now, 7/10 at scale) — the only per-step copy
whose size grows with the project's stated context goal.
- Honesty note: C2/C3/C7 are hygiene (3-5/10); total mechanical savings
across C1-C7 are ~200 lines, roughly 3% of the hot path. The file is already
dense — there is no 30% shrink hiding here, and anyone claiming one is
selling a rewrite. The real size win is PREVENTING the second forward loop
(C8): a duplicated decode path would add ~800 lines overnight.
- Overall: 7/10 — one merge that matters (C1), one fix disguised as a merge
(C5), one scaling win (S10), no fantasies. Nothing executed; every line above
is a hypothesis with a named test.

## Proposed rewrites (NOT live code — review-only sketches)

Status: none of the below is applied anywhere. Each sketch is written against
the exact lines cited so a future lane can lift it verbatim, then run the
listed test. Where the two call sites disagree today, the sketch picks one
behavior and names the difference.

### R1. C1 merged q/gate scatter helper (new header, both sites call it)

New file sketch `src/ops/wrapper/qg_scatter.h`:

```cpp
#pragma once
// Single owner for the checkpoint q_proj per-head interleave geometry:
// [q-head h; gate-head h] per 2*head_dim group (cf. oracle exllamav3
// deinterleave_qg), NOT [q; gate] halves. Scatters into contiguous q/gate
// flats; k/v halves follow unchanged. All pointers are BF16 device memory.
namespace ninfer::ops::detail {

inline void scatter_qg_heads(void* q, void* gate, void* k, void* v,
                             const void* full, std::size_t full_rows,
                             std::int32_t q_rows, std::int32_t kv_rows,
                             std::int32_t n_heads, std::int32_t head_dim,
                             std::size_t cols_T, cudaStream_t stream) {
    if (q_rows != n_heads * head_dim) {
        throw std::logic_error("scatter_qg_heads: q/g interleave geometry mismatch");
    }
    constexpr std::size_t kElem = sizeof(std::uint16_t);
    const char* src = static_cast<const char*>(full);
    const std::size_t src_pitch = full_rows * kElem;
    auto block = [&](void* d, std::size_t d_pitch, std::size_t d_row,
                     std::size_t s_row, std::size_t rows) {
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(d) + d_row * kElem, d_pitch,
                                     src + s_row * kElem, src_pitch,
                                     rows * kElem, cols_T,
                                     cudaMemcpyDeviceToDevice, stream));
    };
    const std::size_t q_pitch = static_cast<std::size_t>(q_rows) * kElem;
    const std::size_t kv_pitch = static_cast<std::size_t>(kv_rows) * kElem;
    for (std::int32_t h = 0; h < n_heads; ++h) {
        const std::size_t grp = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
        block(q, q_pitch, grp, grp * 2U, static_cast<std::size_t>(head_dim));
        block(gate, q_pitch, grp, grp * 2U + static_cast<std::size_t>(head_dim),
              static_cast<std::size_t>(head_dim));
    }
    block(k, kv_pitch, 0, static_cast<std::size_t>(2 * q_rows),
          static_cast<std::size_t>(kv_rows));
    block(v, kv_pitch, 0, static_cast<std::size_t>(2 * q_rows + kv_rows),
          static_cast<std::size_t>(kv_rows));
}

}  // namespace ninfer::ops::detail
```

Call-site rewrites (both keep their existing guards, which stay put):

text.cpp:1239-1265 becomes:
```cpp
            ops::detail::scatter_qg_heads(q_flat.data, gate_flat.data,
                                          k_flat.data, v_flat.data, full.data,
                                          static_cast<std::size_t>(nr), qr, kr,
                                          nh, hd, static_cast<std::size_t>(T), s);
```
(the `hd <= 0 || nh <= 0 || qr != hd * nh` throw is subsumed by the helper's
geometry check; the fused-shape guard at :1221-1225 stays.)

attn_input_proj.cpp:183-211 becomes:
```cpp
        ops::detail::scatter_qg_heads(q.data, gate.data, k.data, v.data,
                                      full.data, static_cast<std::size_t>(kRows),
                                      kQRows, kKvRows, kHeads, kHeadDim,
                                      static_cast<std::size_t>(cols), stream);
```
(its `static_assert(kQRows == kHeads * kHeadDim)` and require_matrix guards
stay; note its hardcoded k/v offsets 12288/13312 equal exactly
2*q_rows and 2*q_rows+kv_rows, so the helper reproduces them.)

Behavior deltas being settled, not hidden: (a) text.cpp wrote
`dst_row * 2U` where the helper writes `dst_row * sizeof(uint16_t)` —
identical today, the helper names the assumption; (b) attn_input_proj threw
`runtime_error` with a tag on copy failure, the helper uses CUDA_CHECK like
text.cpp — pick CUDA_CHECK everywhere (fail-fast beats tagged-late here).
Test: T=0 Paris/Rome ids + oracle layer-0/1 exactness.

### R2. C4 pump scratch (generation_service.cpp:556-587)

Before: `plan/decoded/finished/publish` constructed per pump iteration;
per token a fresh `vector<TokenId> one{token}` + two strings.

After sketch — a reused member (lives on the service, cleared per iteration):
```cpp
struct PumpScratch {
    batch::StepPlan plan;
    std::vector<std::pair<std::uint64_t, TokenId>> decoded;
    std::vector<std::uint64_t> finished;
    std::vector<std::pair<std::shared_ptr<ServeRequestState>, TokenId>> publish;
    void clear() { plan = batch::StepPlan{}; decoded.clear(); finished.clear(); publish.clear(); }
};
// per pump iteration: scratch_.clear(); ... use scratch_.decoded etc. ...
// publish.reserve(decoded.size()) stays.
```
Per-token detok input (pump runs on the httplib ThreadPool, so thread-local
is the correct lifetime, NOT a shared member):
```cpp
thread_local std::vector<TokenId> tl_one(1);
thread_local std::string tl_piece, tl_deposit;
// per token: tl_one[0] = token; tl_piece.clear(); tl_piece = engine_->decode_tokens(tl_one); ...
```
Risk note (kept, not solved): a missed clear() serves last step's data —
keep clear-at-top plus debug-mode poison fill. Test: streamed-text equality +
long-run stability + debug run.

Sample-side scratch (text.cpp:797-802) folds into the device-iota from O2:
```cpp
// replaces host_positions vector + host fill loop + copy_i32:
Tensor positions = work_.alloc(DType::I32, {ndec_cols});
fill_iota_i32(positions, first_col, n_dec, stream);  // new tiny kernel
```
(new kernel, ~10 lines in small_t.cu next to existing fills; the per-row
sample loop itself stays until O1 lands.)

### R3. C5 single sampling resolver (engine.cpp:338-353 + admit path)

Phase 1 sketch — one resolver, validation at admit, card reads it:
```cpp
// single construction site for serve sampling (replaces the hardcoded
// explicit_argmax block at engine.cpp:338-353):
static ops::SamplingConfig make_serve_sampling(float temperature, std::int32_t top_k,
                                               float top_p, float min_p, std::uint64_t seed) {
    ops::SamplingConfig c{};
    c.temperature = temperature; c.top_k = top_k; c.top_p = top_p; c.min_p = min_p;
    c.presence_penalty = 0.0F; c.frequency_penalty = 0.0F;
    c.seed = seed; c.token_counts = nullptr;
    return c;
}
```
plus admit-time rule: while the card holds ONE config per step (card-level
`set_sampling`), requests with temperature != 0 are rejected at validation
(HTTP 400, "temperature sampling not served yet") instead of silently served
greedy. Phase 2 (per-row configs through `ops::sample`, which already takes
per-row positions) is a separate ticket. The 400 is the honest behavior
change; greedy-by-default T=0 ids must be byte-identical before/after.

### R4. C6 indexed lookup (scheduler.cc:38-56, hook_loop.cpp:187-202)

Sketch — index alongside the vectors (vectors keep iteration order):
```cpp
// scheduler: std::unordered_map<std::uint64_t, std::size_t> wait_index_;
// rebuilt on every insert/erase path (submit, admit, erase). find becomes:
Request* RequestScheduler::find_request_mut(std::uint64_t req_id) noexcept {
    auto it = wait_index_.find(req_id);
    if (it == wait_index_.end()) { return nullptr; }
    return &waiting_[it->second];
}
// hook_loop states: std::unordered_map<std::uint64_t /*seq_id*/,
//                  std::shared_ptr<ServeRequestState>> state_by_seq_;
// erase_states_for_reqs collapses from O(finished x states) scan to:
for (std::uint64_t req : finished) { /* erase that req's seq entries by key */ }
```
(seq->req reverse map or store req_id on the state; either is ~5 lines.)
Test: admit/finish churn + two-curl. Invalidation audit required on every
path that mutates the vectors — grep `waiting_.` / `running_.` / `erase` and
touch each one.

### R5. C7 single step-top trace gate (text.cpp:791,822 + layer loop)

Sketch — resolve once, pass down:
```cpp
// top of the decode/forward step:
const bool trace_step = std::getenv("NINFER_SERVE_STEP_TRACE") != nullptr;
const bool dump_logits = std::getenv("NINFER_LOGITS_DUMP") != nullptr;
const bool ladder_on = ladder::ladder_armed();  // single getenv, was ~130
// layer loop: if (ladder_on) { ladder::ladder_dump(...); }  (ladder_dump keeps
// its own guard as backstop; the per-call getenv then never fires on the hot path)
```
The `NINFER_LOGITS_DUMP` block at :822-... is already single-gated; leave it.
Test: each flag on/off.

### R6. S10 dirty-bit table mirror (engine.cpp ~:495-542 region)

Sketch — mirror + per-row dirty bits, set on the only mutation paths:
```cpp
// admit/finish/evict paths set tables_dirty_[row] = true.
// step upload becomes:
for (each row r with tables_dirty_[r]) {
    upload row r device table from host mirror (2D D2D, one row);
    tables_dirty_[r] = false;
}
// debug builds: full-compare mirror vs device every N steps; the
// first-row-nonzero assert stays unconditional (it caught real corruption).
```
Mutation-path audit first: every writer of the host table must set its bit —
miss one and the device serves a stale table silently. That audit is the
ticket, not the loop above.

### What was deliberately NOT rewritten here

C2 (file split) and C3 (prefill/decode prologue) are mechanical moves whose
sketch would be the code itself — no value in duplicating 100+ lines into a
doc. S9/S11 are parked on existing tickets. C8 is a rule, not code.
