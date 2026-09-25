# Row 22 — 262k multi-stream long-context design (multi-model)

Status: PROPOSED 2026-09-25. NEXT-4 in execution order (after NEXT-1 exe + NEXT-2 breakdown + NEXT-3 20b slots; design/paper may start anytime under hold). Needs Chip sign-off. No GPU spent. 5090 HELD.
Importance: HIGHEST (per Chip). Qwen3.8-27B-EXL3 is the measuring stick,
not the product. Engine must serve multiple models at 262k, multi-stream.

## Goal

Design (then spike) the serving path for 8×262k multi-stream without
significant per-token slowdown vs short-context: paged KV, chunked prefill,
prefix cache, quantized KV. Multi-model bar from day one.

## Scope (design first, no GPU)

- Audit today: page/slot scheme, SmallT/Chunked/Prompt paths, GDN slot
  pool, lane/conc scheduler. Name exactly what OOMs or goes quadratic
  at 262k (expect: contiguous KV, dense attention on long prefix,
  unchunked prefill blocking decodes, full-precision KV per token).
- Borrow list (steal, don't invent): paged blocks + device-side tables,
  chunked prefill interleaved with decode, prefix-cache shared prompts,
  fp8 KV / KVarN-style compression, GDN-linear O(1) step exploited.
- Deliverable: design doc + file-level change list + eval plan. No code
  until sign-off part 2.

## Spike bars (part 2, needs GPU release, serial)

- Models: this Qwen hybrid + one pure transformer + one long-ctx eval
  (needle/summarization). No single-model overfit.
- 262k single-stream: correct output, per-token wall within +25% of 2k
  baseline on same binary (excludes prefill; prefill chunked + timed sep).
- 8×long multi-stream: no OOM, aggregate scales, no quality regression
  on short prompts (8/8 stays green).
- Prefill never blocks decode beyond budget: p99 inter-token gap bounded,
  stated in numbers.

## Kill / pivot criteria

- If audit shows dense attention dominates at 262k with no FA/DFlash
  path on this toolchain → pivot to FA-integration ticket, stop here.
- If KV-per-token math says 8×262k > VRAM even with fp8 → scope to
  4×262k or prefix-sharing requirement, say so explicitly.
- No spec-on coupling: row 20 proceeds independently. No native-exe
  coupling: row 21 proceeds independently.

## Merge read (can any of 20/21/22 merge?)

- 20 + 21: NO merge. Different layers (engine MTP path vs CMake/MSVC
  toolchain). Either order. Both re-gate the same 14/23 + 8/8 numbers.
- 20 + 22: NO code merge. 20 touches commit/verify topology; 22 touches
  KV paging + scheduler + attention scaling. Merging conflates a wrong
  token (spec) with an OOM (context). Design of 22 may START now (no GPU:
  audit + doc), code waits until 20 lands or dies.
- 21 + 22: NO merge. 21 is parity of today's binary; 22 changes what the
  binary does. Gate 21 first so 22 has a known-good native baseline.
- Recommended order: 21 parity → 20 salvage → 22 spike. 22 design (paper
  only) runs anytime, including under the GPU hold.

## Cost

Design: zero GPU. Spike: multi-session GPU, serial, after 20/21. No
default flip. No single-model claims.
