# Row 20 — Lane→lane speculative advance + ORT snapshot/restore test

Status: SUPERSEDED 2026-09-25 by row-20b (slots). Do not execute; retained as design record. No GPU spent. 5090 HELD.

## Why this exists

Today spec-on does ~7 target passes for 5 tokens at accept-3
(anchor + verify + 5 commit rows, lane untouched via spare→shadow).
That is 0.7 tok/pass — slower than spec-off 1-for-1 by construction.
Normal MTP math needs the common case near 4 tokens per 1 target pass.

## Why rewind exists at all

Rewind is not a feature. It is the price of speculation:

- Drafts guess K tokens. Verify checks them. Partial accept is the
  normal outcome (accept-1 and accept-2 happen constantly in the logs).
- Transformer rewind = decrement a length counter. KV rows for rejected
  tokens are simply ignored. Free.
- GDN rewind = restore overwritten recurrent state. There is no length
  counter to decrement. Without a restore path, a partial accept leaves
  a corrupt lane and every later token is wrong.
- Spec-off needs no rewind because it never speculates. Spec-on without
  rewind is just wrong tokens. So the choice is never "rewind or not" —
  it is "pay restore on partial accept, or pay replay on every accept."
  Today we pay replay always. This ticket prices the other side.

If the spec was wrong means: if drafts were always fully accepted,
rewind would never fire. They aren't (logs show accept-1 rewinds
regularly), so the rewind path must exist and be bit-exact.

## What changes

- Verify runs lane→lane (advance in place, assume full accept).
- Spare becomes the pre-step snapshot (taken once per step, before verify).
- Accept-3 full: done, lane already correct, zero commit rows.
- Partial accept: restore lane from spare (ORT pattern: same addresses,
  D2D copy, no realloc), replay only the diverged suffix lane-into-lane.
- Bonus row follows the same rule as the accepted prefix.

## ORT pattern test (explicit)

- ONNX Runtime GenAI note: snapshot/restore must preserve buffer
  addresses "which CUDA-graph replay requires."
- Test here: GDN slot bases + mbase scratch + reset arena must be
  process-stable across capture and replay (Phase B audit said yes —
  re-prove under lane→lane writes, not just spare writes).
- Prove: capture commit-correction exec once, replay across F with only
  contents patched (tokens / pos / slots), addresses identical.
- Fail-closed: any address motion → ticket stops, spare→shadow stays.

## Bars (all must pass, same binary)

- Paris frozen 6511/314/9338/369 accept-3 + accept-1/accept-2 rewinds,
  byte-identical vs today's spare→shadow outputs.
- Fuzz accept patterns (forced accept-0/1/2/3) — no divergence vs eager.
- Spec-off untouched: 14.3 + true 8/8 on the same binary.
- Spec-on FOX-64 streaming: must beat 14 to ship. Lands 17-20 → close
  as priced, keep 23. Lands >23 → revert.

## Kill criteria

- GDN addresses move across steps → kill, keep spare→shadow.
- Any rewind mismatch vs eager → kill, keep spare→shadow.
- Wall misses <14 → keep as experiment, default stays spec-off 14.

## Cost

One GPU session + fuzz contingency, serial. No default flip in this ticket.
