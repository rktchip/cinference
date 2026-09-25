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

## Reconciliation (committee corrections folded, off-GPU)

- NVTX: ABSENT in /root/mtp-nsys (captured -t cuda only) —
  nvtx_gpu_proj_sum SKIPPED, no NVTX data. GPU-projected buckets need a
  re-capture with -t cuda,nvtx. Until then: my draft/verify split is
  CPU-range arithmetic on async launches and the verify half is
  MISATTRIBUTED (a sync inside the "draft" range bills queued verify work
  as draft). Accepted.
- Verify 5–10ms is impossible: full target pass floor ~8.4ms (15GB÷1.79),
  realistically ≥ T_1 ~14ms. True step recomposition (pooled E=2.82):
  drafts ~22 + verify ~14-16 + commits ~28 + gaps ≈ 64-66ms ✓ matches
  wall-derived 64ms. The 5–10 bucket was sync-billed draft time.
- Makespans: the "32ms/tok" scare is wall÷tokens including ~1.0-1.2s TTFT.
  nsys FOX run: 14 steps, wall 2.30 = TTFT ~1.1 + decode ~1.2s;
  55 toks/14 steps E=3.93 → 22.6×3.93 ≈ 88ms/step ✓ row-18 22.6 decode
  rate CONFIRMED on the capture binary. No re-capture needed for the rate;
  the ~0.6-1.0s "outside decode" is TTFT/prefill + eager anchor replay.
  Per-regime step budgets (FOX E=3.9): beat 14.3 → <53ms; margin → <45ms.
  Pooled (E=2.82): <40/<34. PARIS (E=2.12): <30/<26.
- Slots ceiling with gaps KEPT: (64-28)/2.82 = 12.8ms/tok — beats 14.3
  ~10%, MISSES 12.2 margin. 9.6–11 needs gaps gone = on-device accept
  and/or cheaper drafts. Filed as the bar, not the hope.
- Drafts are the next wall: 7.8ms/tok = 55% of spec-off's budget; ~7.3ms
  per draft step (half a target pass for a 1-layer head = eager launches
  on WDDM and/or full-vocab head and/or the misattribution above). k-rule
  from this histogram: k=3 → 36ms/12.8; k=2 → ~28.7/~12.2;
  k=1 → ~21.3/~11.9 (gaps kept). Decision AFTER draft graphing +
  head-truncate check, not before.
- What survives untouched: commit bucket (~2 M=1 rows/step × 14-15ms ≈
  28ms). Slots go-case holds wherever draft/verify lands — it depends
  only on what slots remove.

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

## E recompute (committee item 1 — closed, with a reversal)

Empirical ground truth per request (log F-advance + last commit = gen):
gen = (F_last − F_0) + commit_last, verified EXACT on 6 clean requests
(sumC == output column: PARIS 153, CODELOW 98×4).

- Finding: each step adds commit-len NEW tokens (bonus + accepted +
  extrapolated next-bonus). E = mean(accepted) + 2, NOT +1. Max/step = 5.
- The pooled 83-step 3.82 was CORRECT. My "correction" to 2.82 was the
  error (it dropped the extrapolated bonus, which the output column proves
  is emitted). Committee's double-count charge REFUTED by exact matches.
- Regime E (clean): PARIS 153/49 = 3.12 (not 2.12); CODELOW 98/20 = 4.90
  (not 3.90 — near the 5.0 max, deterministic digits draft perfectly);
  PRIME 70/15 ≈ 4.67 but stop-token-truncated (reason=stop), use with care.
- FOX req1 (F_0=53 ≠ prompt 32, adv=34) is CONTAMINATED (likely merged
  warmup/double-run) — discarded, re-run if FOX-E is ever load-bearing.
- Ship math IMPROVES: pooled step 64ms − 28 commits = 36ms → 36/3.82 =
  9.4ms/tok (ships with margin). PARIS-regime: 36/3.12 = 11.5 < 12.2 ✓.
  CODELOW-regime: 36/4.9 = 7.3 ✓✓. ALL regimes ship on gaps-kept math.
- Break-even E at 36ms step vs off 14.3: 2.52 ✓ (committee's number
  stands). Adaptive gate threshold ~2.5 rolling E — every regime measured
  clears it (lowest clean: PARIS 3.12).

Per-prompt E, SUPERSEDED by the recompute above (these used the +1 model —
kept for trail): FOX 3.71 (n=14, contaminated — discard), PARIS 2.12 → 3.12,
CODELOW 3.90 → 4.90, PRIME 3.87 → ~4.67 (stop-truncated).
CODELOW is NOT low-accept at greedy (deterministic digits draft perfectly);
PARIS prose is the most partial-heavy regime at T=0 (17 accept-0 + 16
accept-1 of 49). Combined true E over 225 clean-ish steps ≈ 4.48
(mean accepted 2.48 + 2; FOX-req excluded as contaminated).
Off walls: FOX 1.03, PARIS 2.09, CODELOW 1.40, PRIME 1.02. On loses every
regime 2.2–2.7× (FOX 2.2-2.3, PARIS 4.78, CODELOW ~3.75, PRIME ~2.77).
CODELOW veto stands only with its off-number: 3.75 vs 1.40. Filed.

## Re-bucketed step (GPU-projected NVTX, /root/mtp-nvtx, 77 MTP steps)

Range semantics (read in source): MtpProposal = draft head+argmax only;
MtpForward = mtp_forward_core = draft-module forward AND per-row MTP
refills (both call it); target verify has NO top-level range (only
per-layer verify.layer.*); ordinary rows share the layer ranges.

- Drafts: proposal 225 × 0.61ms = 0.14s; forward-core share ≈ 3/8.5 of
  657 × 0.90ms = 0.59s → ~2.7ms → drafts ≈ 0.6×3 + 2.7 ≈ 4.5-5ms/step.
  DRAFTS ARE CHEAP. The 22ms draft bucket was misattributed verify.
  Committee's verify-wall hypothesis CONFIRMED.
- Verify call: per-layer full 737µs × 16 + GDN 286µs × 48 ≈ 25ms GPU per
  target-verify call. T_ver ≈ 25-30ms (committee's ~30 vindicated).
  (Width-blended avg — exact M=4-only split needs per-instance query.)
- Refills: ~5 core calls × 0.9 ≈ 4-5ms/step (new explicit bucket).
- Commits: ~28ms/step wall (stands).
- Step recomposition: 5 (drafts) + 25-30 (verify) + 28 (commits) +
  4-5 (refills) + gaps ≈ 64-70ms ✓ wall-derived 64.
- Consequences: k-table flips — drafts cheap + m16 flat across M ⇒ k=3-4
  likely wins (committee's call). Small-m kernel (m=2-8, dequant-once,
  gemv-like) becomes TOP lever: pulls verify 25→~10 AND fixes conc-2..8
  in the same move (the original "lifts both" ask). Row 16 covered m=1
  ONLY — "kernels done forever" does NOT cover m=2-8. Filed.
- Slots ceiling restated: (5+25+5+gaps)/3.82. Gaps-kept ≈ 12.8 (ships ~
  10% over 14.3, needs on-device accept for 12.2). Small-m on top:
  ≈ (5+10+5)/3.82 ≈ 5ms/tok. Order: slots (removes 28) → small-m
  (removes ~15 + conc). T_ver picked small-m over draft-graphing.
- TTFT (warm, steady-state): on 1.8-2.0s (CODELOW p66) / 1.3-1.5s (PRIME
  p49) vs off warm ~0.1s. First-req capture ≈ 0.2s of it. ~1.5s steady
  overhead (MTP prefill + eager anchor + first-step capture) is a SHIP
  BLOCKER independent of 20b (20b doesn't touch TTFT): at 1.5ms/tok saved
  it takes ~1000 tokens to earn back. Filed with numbers.

## Conc SM-busy (both servers, nsys) + amended bars/order

Spec-off conc-2 (/root/conc2-off, makespan 2.19): ZERO gemv — all m16
GEMMs (b4 med 42.7µs, b3 med 54µs). EXL3 ≈ 1.52s / 2.19s wall = ~70%
GPU-busy. Math-bound, no WDDM punishment. m=2 ≈ 2× m=1 work with no
overlap benefit at B=2 (GEMM-bound); B=4/8 amortize via bigger Ms.
MTP-server conc-2 (/root/conc2-mtp, makespan 3.73): 26k gemv m=1
instances (608ms) ON TOP of m16 (1.66s) = 2.9s/3.73 = 78% busy. The
penalty is m=1 eager rows (no graphs on the MTP server) where spec-off
runs m=2 efficient GEMMs or B=1 replay — plus scheduler races between
batched-m=2 (zero MTP) and serial-B=1 (MTP-eligible) across runs.
Verdict: committee flag resolved as graphs+math, not driver. Multi-seq
graphs (± conc-aware MTP gating = fix-5) are the aggregate lever after
all — row 17 "polish" verdict REVERSED, pending 20b + easy-fix order.

## Next (in order — committee amended)

Ship bar (prediction ≠ bar): decode ≤12.2 on high-accept regimes; TTFT at
parity with off (BLOCKER until fixed); PARIS no worse than off; conc guard
unchanged. Adaptive gate (~2.5 rolling E → off path) only safe once the
MTP server's off-branch is graphed (eager today = the 58%).

1. BUILD 20b, k-parametric slots + graph the MTP server's off-branch.
   GO signed by Chip — three corrections folded, no geometry change.
2. Small-m kernel ticket (m=2-8 dequant-once gemv-like): top lever per
   T_ver verdict — pulls verify AND conc in one move. Write it next.
3. Draft graphing only if small-m stalls; k decision after (k=3-4 lead).
4. Native exe DEFERRED (reviewer overruled: graphs remove the launch
   overhead; no mid-algorithm re-baseline). Revisit after 20b ships.
