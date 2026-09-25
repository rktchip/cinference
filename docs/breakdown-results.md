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
- commit lens (accepted+2): 2×17, 3×19, 4×9, 5×38 = 317 tokens / 83 steps
- E[accepted] = 151/83 = 1.82; E[tokens/step] = 317/83 = 3.82
- FOX-like repetitive prompts accept-3 heavy; CODELOW drags the mean.
  Full accept is the plurality (38/83 = 46%) but partials are routine —
  rewind path must exist (answers Chip's question with data).

## Graph health

- vgraph: 1 capture, rest replay. cgraph: 1 capture, rest replay.
- capture-fail: 0. eager-disabled: 4 (first-step seen rule, expected).
- Verdict: both seams healthy on replay. No recapture pathology.

## Conc-2 finding (decisive)

- conc-2 FOX on the SAME spec-on server: makespan 3.64s, per-req 3.64/3.64
  (fully serial — zero overlap), and serve-mtp count did NOT move (83
  before → 83 after): conc-2 steps emitted ZERO MTP lines.
- Conclusion: multi-row steps bypass step_mtp_decode entirely (ordinary
  target-only path, engine.cpp:634 gate). Conc-2 never specs — it runs
  spec-off eager per stream, serialized. This confirms fix-5 scoping and
  halves the conc-2 suspicion: the doubling is scheduler serialization +
  m=2 math, not a graph-shape miss on the MTP path. The remaining half
  (spec-off B>=2 eager vs B=1 replay) still needs the nsys SM-busy check.

## Ceiling (committee metric, log-derived)

- (T_draft + T_ver)/E[tok/step] needs nsys means — NOT yet measured.
- E = 3.82 sets the denominator: every 1ms cut from draft+verify = ~0.26
  ms/tok off the ceiling. Commit rows (317-83 = 234 extra forwards over
  83 steps) are the wall to kill — 20b slots math holds: kill commits and
  the ceiling is draft+verify over 3.82.

## Next

1. nsys cuda-only decode window: T_draft / T_ver@M=4 / T_commit means +
   SM-busy B=1-replay vs B=2-eager (decides the rest of the conc question).
2. Then 20b slots (NEXT-3). Then native exe (NEXT-1) re-gate.
GPU yielded after run. Server killed, 5090 idle.
