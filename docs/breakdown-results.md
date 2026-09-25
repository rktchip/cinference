# NEXT-2 breakdown — interim results (log arm, no nsys yet)

Date: 2026-09-25. Binary: graph-build (commit 2296247 + 4484b59 docs-only).
Env: CUDA_EXL3_AUTOTUNE=0, CUDA_EXL3_SPLIT_TARGET=0, MTP --draft-tokens 3.
Server :8901, NINFER_MTP_DEBUG=1, STEP_TRACE=1, GRAPH=verbose.

## Walls (serial, greedy, thinking off)

- FOX-64: 2.25s / ~55 words
- PARIS-150: 4.78s / ~127 words
- CODELOW-96: 3.69s / ~33 words (low-accept prompt works — entropic output)

## Accept histogram (83 decode steps, all 3 prompts)

- accepted=0: 17, =1: 19, =2: 9, =3: 38
- New tokens per step = accepted + bonus(1): 1×17, 2×19, 3×9, 4×38
  = 234 tokens / 83 steps. E = 2.82 (95% CI ≈ ±0.26, single-prompt caveat).
  (Earlier 3.82 wrongly counted commit rows incl anchor replay as new tokens.)
- Partials are the majority: 45/83 (54%) accept-0..2; full accept 38/83 =
  46% ± ~11pts. Data case against replay-on-partial, for slots (same cost
  every step).
- k-sweep priced from this histogram: E = 1.80 at k=1, 2.36 at k=2,
  2.82 at k=3. Third draft pays iff one draft step + widening verify
  M=3→M=4 costs < ~20% of the k=2 step. nsys gives both numbers.
- Ceiling: beat 14.3 ms/tok → draft+verify+overhead < ~40ms/step
  (14.3×2.82); 15% margin → < ~34ms, ~31ms at low CI end.

## Graph health

- vgraph: 1 capture, rest replay. cgraph: 1 capture, rest replay.
- capture-fail: 0. eager-disabled: 4 (first-step seen rule, expected).
- Verdict: both seams healthy on replay. No recapture pathology.

## Conc-2 finding (corrected)

- conc-2 FOX on the spec-on server: makespan 3.64s, per-req 3.64/3.64
  (together, not serial — my earlier "fully serial" misread).
- serve-step lines: 58× `seqs=2 m=2` + 1× `seqs=2 m=64` (joint prefill).
  The scheduler DOES batch. serve-mtp count unmoved (83→83) because the
  MTP gate (n_dec==1) bypasses step_mtp_decode for m=2 — conc-2 runs the
  ordinary target-only path. Contradiction resolved: batched but never
  spec'd — concurrency penalty without batching benefit on the MTP server.
- Second problem (committee): MTP-server ordinary path ~58% slower than
  spec-off contract (3.64s vs row-17 2.3s for 2×FOX — confirm same
  workload). Likely cause: graphs default spec-off only
  (graph_eligible !mtp-gated), so the MTP server's off-branch runs eager.
  That fails the concurrency contract the moment the MTP server is default.
  Cheap check first: [graph] eager-shape lines on m=2 steps; then nsys
  SM-busy B=1-replay vs B=2-eager on both servers.

## Ceiling (committee metric — superseded by nsys section below, kept for trail)

- First version wrongly used E=3.82 (double-counted anchor replay).
  Correct E=2.82, budgets <40ms / <34ms per step. See nsys section.

- (T_draft + T_ver)/E[tok/step] needs nsys means — NOT yet measured.
- E = 3.82 sets the denominator: every 1ms cut from draft+verify = ~0.26
  ms/tok off the ceiling. Commit rows (317-83 = 234 extra forwards over
  83 steps) are the wall to kill — 20b slots math holds: kill commits and
  the ceiling is draft+verify over 3.82.

## nsys kernel means (same binary, FOX-64)

Spec-on MtP3 capture (/root/mtp-nsys, 18.9MB) vs spec-off capture
(/root/specon-nsys2, 9.0MB). Decode medians (prefill-free) agree
across arms: gemv K4 ~20-29µs, K3 ~35-51µs, gemm m16 ~42-83µs.
Per m=1 body row ≈ 4×K4 + 1×K3 ≈ 116µs × 64 layers ≈ 7.4ms EXL3.

Wall-derived spec-on step: 22.6ms/tok × E2.82 ≈ 64ms/step vs budget
<40ms (beat 14.3) / <34ms (15% margin). Component split (stated
assumptions: 7.4ms/row, E[a]=1.82, refills inside verify bucket):
drafts 3 rows ≈ 22ms + commits (a+2) ≈ 28ms + verify M=4 + MTP
refills ≈ 5-10ms + gaps = ~64ms. Commits ≈ half the step.
Slots ceiling ≈ (22+5..10)/2.82 ≈ 9.6-11ms/tok — under 14.3 WITH
margin, if drafts hold. Drafts alone (22/2.82 ≈ 7.8ms/tok) already
beat spec-off per-token: the path is real, commits are the wall.
Exact T_ver needs NVTX-per-phase or refill-aware split (m16 mixes
verify + per-row MTP refills) — flagged, not blocking 20b.

Spec-off FOX wall 1.00s vs spec-on 2.22-2.30s same prompt (2.2×).
T_1 consistent with 14.3 contract.

## Next

1. Wider histograms: CODELOW + one more prompt, several hundred steps,
   each with spec-off baseline (CODELOW has no off-number yet — 3.69s
   is not a veto until it does).
2. Conc-2 SM-busy both servers (batch size + graph-vs-eager HAVE from
   logs; only SM-busy needs nsys).
3. Then 20b slots (NEXT-3). Native exe (NEXT-1) re-gate anytime.
