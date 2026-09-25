# Row 23 — Batch the MTP prefill mirror (TTFT fix)

Status: PROPOSED 2026-09-25. Interleaves with 20b (disjoint code:
mtp_prefill_fill vs step_mtp_decode). Needs Chip sign-off. No GPU spent.
Importance: SHIP BLOCKER (TTFT parity is in the amended bar).

## Diagnosis (measured, /root/mtp-ttft.log)

TTFT linear at ~27ms/prompt-token (p66 2.2s → p444 11.8s, intercept
~0.4s) vs spec-off warm ~3ms/tok. 9×. Strictly linear ⇒ token-by-token.
Source: mtp_prefill_fill (engine.cpp:1213+) runs ordinary_decode_batch
(FULL target row ≈ 14ms) + mtp_forward_batch per prompt token.

## Fix (two rungs)

Rung 1 (stepping stone, this ticket): reuse prefill hiddens. The ordinary
prefill JUST computed every row's target hidden batched — slice per-row
hidden from prefill output instead of re-running ordinary_decode_batch
per row. Per-row cost drops to MTP-forward-only (~1-2ms). TTFT ≈ 0.3 +
N×0.002 (p66 → ~0.45s). MTP KV still warms sequentially (GDN trajectory
exact). NOTE: rung-1 still scales linearly (~5.3ms/ptok sequential MTP
warming): p2k would add ~10.6s, p8k ~42s. It canNOT satisfy the bar alone.
Rung 2 (REQUIRED to ship): one batched M=N pass through the MTP layer.
All inputs (target hiddens + next-token embeddings) are known once target
prefill finishes; skip lm_head/logits during warming (nothing reads them).
A few ms total, prompt-length-independent.
Bar (amended): TTFT parity at REALISTIC length (p2k or longer) with
explicit tolerance (state it: e.g. within 2× off warm at p2k), not p66.
Bar: TTFT within 2× of spec-off warm on p66/p128; Paris/FOX outputs
byte-identical vs today; 8/8 untouched.

Rung 2 (only if rung 1 misses): width-N GDN columns so the mirror batches
to M=N. Needs backend support (gdn snapshot trips past 1 column today) —
file as backend ticket, do NOT expand this one.

## Kill criteria

- Any output divergence vs today on frozen prompts → revert.
- Rung 1 lands < 2× off → close, ship it. Else escalate to rung 2 ticket.

## Cost

Half GPU session, serial. No default flip. No 20b dependency (either order,
different function).
