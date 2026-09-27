# AUTONOMOUS PLAN — 20 steps (frozen 2026-09-26)

Picks up: arms A, B, C1 sealed; capture next; code lane holds split,
verify-replay branch, row-23. Committee deltas applied: small-m gated on
bench (step 3 -> 13); FP8 KV isolated numerics step (19); ship bar relative
to same-session spec-off.

RULES
- Identity = exe sha256. Stamps start+end, shared <=512 MiB, mem co-tenant
  check, LOADCLK. Thinking-off hashed prompts; FOX = lowaccept.
- Every step: paper prediction (ms + confidence) BEFORE code. Then gate,
  kill, measure.
- Mirrored old/new exe interleave (A-B-B-A), one session. MTP modes on
  ms/step; MTP vs spec-off on ms/token.
- Ship bar (relative): MTP >=15% faster than same-session spec-off on regime
  average; lowaccept no worse; TTFT parity.
- Diagnostics time-box: 2 sessions, then park. No new oracle features. No
  sweeps without a paper prediction.
- Escalate to Chip ONLY: correctness gate fails at non-tie; result misses
  prediction by >2x either way; step changes model files or prod defaults;
  or everything blocked.

GPU LANE (now)
1. Traced spec-off capture (fresh nsys, --cuda-graph-trace=node). Buckets vs
   floors (linears 6.0, head 0.31, GDN 0.17ms); rank measured-minus-floor.
   Size fusion: removable x avg small-kernel cost.
2. Finish ABCCBA (C2, B2, A2). Seal S1-pre: ms/step per mode, ms/token.
3. Post-sequence GEMV bench: per-shape GB/s M=1 AND v3/gemm M=4 vs floor.

SPEC-OFF HUNT (helps every mode)
4. Fix bucket #1 (method per trigger). Kill: <1/3 predicted delta -> revert.
5. Fix bucket #2, same procedure.
6. Fix bucket #3 only if >=0.5ms over floor.

MTP PATH
7. Build split (8b6ff35) + re-verify: NULL-OK, caps 0/3, interdiff == 0;
   spare-snapshot drop rides here.
8. 8/8 + FOX 500 drift on split; divergences only at ties (2-ULP).
9. Verify replay (verify-replay-probe) on split. Gates: bit-exact, hists
   match. Predict ~10-15 ms/step (provisional).
10. Confirm trailing-row replay in nsys; if eager, fix capture. Gate:
    bit-exact else diff <= floor.
11. S1-post: new slots vs spec-off, A-B-B-A, regimes + thinking-on
    code-prompt row. Read vs relative bar.
12. Delete legacy rows; free shadow; oracle slots only when oracle on.
13. Small-m ONLY if step 3 shows v3 M=4 >1.3x floor. Route verify M=4 AND
    m=2..8. Predict verify ~25-30 -> ~10-12ms (medium).
14. DEMOTED 2026-09-27 (FOX wins by 22.5%; lowaccept premise gone). Safety net
    only for pathological prompts: adaptive gate (rolling E<2.5 -> off) +
    k resweep (2/3/4). Lowaccept >= off.
15. On-device accept ONLY if capture shows sync gaps >=1ms/step.

LONG CONTEXT / CONCURRENCY
16. Row-23 rung-1+2. Gates: TTFT parity p2k+p32k; hists unchanged.
17. Memory diet: shared column pool <=2 lanes; engine sizes pool+KV, refuses
    instead of spilling. Report free GiB + token budget.
18. S3 conc 1/2/4/8 at p2k, after step 13.
19. FP8 KV numerics step: re-verify (8/8 + drift), then S1/S2 at 32k.

MODEL-SIDE
20. MTP layer 8-bit: histogram A/B (values only). Adopt if lowaccept mean
    accept rises >=0.1.

PARKED WITH TRIGGERS
- Split-seq decode attention: 32k capture shows attention >=20% of step.
- Recipe re-quant lower bpw: quality run (qbench KL) after step 19.
- Background, code lane: SSE disconnect fix.

REPORT: per landed step, max 10 lines: S-table / measured vs predicted / next.
