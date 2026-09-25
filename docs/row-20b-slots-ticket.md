# Row 20b — Slots-first speculative state commit (replaces Row 20 lane-restore)

Moves S1 (decode ms/tok conc-1): predicted step 36ms -> (T_ver + 1xT_1 +
refills) ~44-49ms -> /3.82 ~11.5-12.8 high-accept. Kill: S1 slots >= S1
legacy (22.6) same binary -> revert (flag-off). S2/S3/S4 untouched (row-23
owns S2; conc owns S3).

Status: PROPOSED 2026-09-25. NEXT-3 in execution order (after NEXT-1 exe + NEXT-2 breakdown). Needs Chip sign-off. No GPU spent. 5090 HELD.
Supersedes: Row 20 lane-restore — DEAD, do not revive. No commit rows, no state copies, no replay, no restore path.
Default flip: none. Spec stays opt-in regardless of outcome.

## 1. Goal

Replace Row 20 with slots-first commit: keep k+1 per-position GDN state
slots produced during verify, select committed state with a device-side
slot-index write instead of copying or replaying state.

## 2. Why Row 20 died (committee consensus)

Lane-restore required commit rows / bulk copies / replay on accept — pure
overhead on the fast path, scales wrong with depth. Consensus: delete it,
commit by index instead. "Replay diverged suffix" also read backwards:
after restore-to-pre-step you replay the accepted prefix, not the rejected
suffix. 20b fixes the wording by removing the path entirely.

## 3. Design: slots-first + layout (named explicitly)

Slot count: verify has k+1 columns (bonus + k drafts); every accept
outcome a ∈ 0..k needs the state after column a = k+1 states. Layout B
(t[0]=lane in place for col 0, t[1..k]=column slots; static per lane,
init-filled once). Off-by-one named: table length == width (k+1), NOT k
(an off-by-one here breaks exactly one accept value — forced accept-0..3
fuzz comes first).
- Recurrent: chained per column (col c reads t[c-1], writes t[c]).
- Conv: NO gather — one width-1 snapshot per column with chained slots,
  using the real kernel (t[c] holds the exact post-column-c window).
- Commit: a==0 is a no-op (both states already in place); a>=1 copies
  t[a] -> lane via pool copy_slot (conv+recurrent together).
- Trailing row: t[a] covers through F+a, but out_a (@F+1+a) was only
  argmaxed, never executed — one M=1 row executes it (was commit_len
  rows). E (tokens/step = a+2) is UNCHANGED: decoded list + F-advance
  identical; only compute rows die. The "+1 disappears" correction is
  declined with this derivation (filed 2026-09-25).

Slot mapping: COPY-BACK first cut. After accept a: copy slot[a] (or lane
itself under layout B for a=0) back into the fixed lane address. One
state read + one write per step (~0.2ms at ~150MB state, ~1% of a 21ms
step). No graph invalidation (spec-off graph, off-branch graph, commit
exec all keep capturing against the fixed lane address). Rotating slots
with an on-device indirection table (Phase A "F as data" rule) is the
later optimization, NOT this ticket.
Conv state: if gdn_input_proj_conv_snapshot can't write per-column
destinations, DON'T block — rebuild the window after accept with a tiny
gather: window after column a = last (kernel−1) pre-step inputs followed
by x_0..x_a, all resident from verify. No per-column conv output needed.

## 3b. Split build (committee order — bisect safety) + transition hazard

Off-branch graphing LANDED (e40aec1, no-regress). FORCE_OFF switch LANDED
(bf16f41, fallback decode 13.3 ✓, TTFT 1.3s tax ⇒ gate unsafe pre-row-23).

HAZARD (committee): off→on resume degrades drafts silently. MTP cache
misses off-stretch positions (conc 2→1, gate fallback→resume); verify
masks it (correct, lower accept — no gate catches it) or a token-by-token
re-warm stalls. Test (after slots + row-23): mid-request off-N-steps→on,
accept histogram + step times post-resume vs uninterrupted. Fix: rung-2
batched warming as resume catch-up (save target hiddens ~10KB/tok during
off steps, one M=N pass on resume). Rung-2 required three ways: TTFT,
conc transitions, adaptive gate. Rung-1 + rung-2 land TOGETHER (rung-1
alone: conc-8 × p2k ⇒ last TTFT ~85s — ships nothing).
Accounting nit: req3 TTFT 2.8 = req2's serial mirror (~1.8) + own (~1.0);
mirror prefills serialize across concurrent admissions. Closes.

Off-branch graphing lands + gates FIRST (independent): MTP-server conc-2
must match spec-off (≈2.2s band), 58% penalty gone. THEN slots +
commit-by-index against the known-good off path. Fuzz failure then
bisects one change, not two.

### 3.1 Verify is causal, hidden/logits already correct

Draft + target verify run causally over the window. Hidden states and logits
from verify are already the correct autoregressive values for every position.
No linears/head recompute on accept — only GDN recurrent state commits.

### 3.2 Per-position GDN state slots from verify

During verify, materialize one GDN recurrent state slot per window position:
slots 0..k (slot i = state after accepting i tokens). Written once, in place,
as a byproduct of the verify pass that already runs.

### 3.3 Device-side slot-index write

On accept of n tokens, commit = single device-side index write selecting
slot n as live lane state (pointer/index only). No commit rows. No memcpy.
No replay. On-device, no host round-trip on the accept path.

## 4. VRAM cost

(k+1)x lane recurrent state. Accepted at conc-1/2 (only configs targeted).
State vectors only — not KV, not activations, not a second model.

## 5. Fallback: low-VRAM variant (only if slots exceed budget)

- Full accept (n == k): lane/shadow index swap. No copies, no fixup.
- Partial accept (n < k): recurrence-only fixup from position n forward,
  using cached q/k/v/g/beta + conv slice resident from verify. No linears,
  no head recompute. Spare/shadow retained ONLY for the swap (one buffer,
  alloc-once at init). Primary path does not need it.

## 6. No address-stability probe needed

Buffers (lane slots + one spare/shadow) allocated once at init, fixed
addresses. Slot selection by stable index, never realloc/pointer chase.
Assert alloc-once in code and move on.

## 7. Correctness bar (8/8 required) + first-gate checklist (classic misses)

Greedy token sequences spec-ON must be identical to spec-OFF on: Paris,
FOX, Rome, plus low-accept prompt (code/numbers, hostile to drafter).
Near-tie divergences allowed only where two logits are within float tie
tolerance AND verified near-tie on re-decode (log position, logits, delta).
Anything else = fail, stop, no perf bands. (Byte-identical vs today is
explicitly NOT required: today = m==1 gemv_plain family, verify = M=4
family, bits differ on first full accept.)

First gate (before ANY perf numbers):
- Slot the CONV state, not just recurrent. Each slot needs its own conv
  window. Wrong here doesn't crash — slow quality drift hundreds of toks
  in, past the 8/8 matrix. Run a 500+ tok continuation and diff vs
  spec-off, don't trust 8/8 alone.
- Rewind the MTP layer's own cache on partial accepts. Draft positions
  write into it — length must reset to accepted position, like target
  attention layers.
- Force accept-0 through accept-3 in fuzz FIRST. Slot-selected state vs
  today's commit-row state within tolerance (not byte-identical); greedy
  output token-identical to spec-off on Paris.
- Rejected positions never readable: target attention length == accepted
  position after every step. Stale draft KV beyond is harmless ONLY if
  nothing reads past that length (assert it, don't assume it).

Suspect predictions across accept values (oracle sweep reads directly;
conv-first: the recurrent diff is predicted downstream of stale conv
through the trailing row — if rec drops to the cap-0 floor after the conv
fix, the recurrent chain is cleared with no separate investigation):
- Token offset dropped (every column publishes to one slot): clean at
  full accept (last writer wins = post-col-k), error growing as accept
  falls.
- Window shifted by one (slot c holds window ending c-1): wrong at EVERY
  accept incl full accept; error shaped like a one-position shift.
- Wrong slot/batch base (gdn_conv.cuh publish address): sentinel survives
  in the intended slot (tNaN > 0) and some other slot gets overwritten.
- Missing writeback entirely: lane conv == pre-step window
  (convBPre == 0) at every accept incl cap-0.
- Garbage reads: huge max with small mean (vs systematic = large mean).
- Partial write (extent/stride bug, some heads/channels written, others
  not): tNaN strictly between 0 and 1. Its own signature — neither a
  missing write (tNaN == 1) nor a content bug (tNaN == 0, tVsA > 0).
Two suspects share "clean at full accept" — the sentinel tells them
apart (static: col-1 conv source IS t[0]/lane, text.cpp:1590ff, and the
width-1 chain is mathematically exact, gdn_projected_conv.cu:43ff, so the
kernel is cleared and only bindings remain suspect):
- Offset dropped (every column publishes to one slot): last writer wins
  = post-col-k, clean at full accept, error growing as accept falls;
  NEVER-written slots keep NaN (tNaN high, concentrated in t[1..2]).
- Chained stale source (col-0 conv never landed in lane, col 1..3 chain
  on the stale window): every slot gets written (tNaN == 0) with values
  missing x_0; with a 3-entry window x_0 ages out by column 3, so full
  accept is clean and a=1,2 are wrong. convBPre != 0 distinguishes it
  from a missing writeback.
RESOLVED (no sweep needed): the chained-stale suspect is dead. It rested
on a slice-A fact — "lane conv pristine" — that was true only under
slice-A bindings (c71695c: conv-input dump bound, table NOT passed, so
verify never touched the lane). Under slice B (067bd42: table bound,
t[0]=lane) col 0's conv publishes to the lane by code (dst_c = coltab
slice 0; EXL3 live kernel = SnapshotHistoryPublish). Table-bound in the
VTARG runs is proven, not assumed: NINFER_SLOTS=1 was set, and a table/
width mismatch throws (no silent fallback) — no FATAL, so the coltab
branch ran. Col 1 reads genuine post-col-0 output; VTARG-SAME stands as
evidence. Remaining: publish-side offset/base/extent — the sentinel
(tNaN fraction) picks among the three.
Terminology: VTARG-SAME cleared verify's LOGITS only, not its snapshot
writes — the suspect line (gdn_conv.cuh publish address) is inside
verify's kernel. "Verify exonerated" means logits only.

## 8. Perf bands (filled)

Metronome, conc-1, x3 per prompt, spec-on vs spec-off same prompts.
- 14–17: SHIP zone (opt-in only, no default flip) ONLY if spec-on beats
  spec-off by >=10–15% at conc-1 across x3 AND not worse on low-accept.
  Both required.
- 17–20: REVERT. No ship, close 20b.
- 20–23: REVERT. Parity-or-worse = revert, not tune-and-retry.
- >23: REVERT + defect note with metronome logs.
Report median of 3 + range every cell. Low-accept veto blocks ship.

## 9. Kill criteria

- Correctness < 8/8 (excl logged near-tie allowances).
- (k+1) slots + spare exceed VRAM at conc-1 → try fallback once; if it
  needs linears/head, kill.
- Fixup needs beyond cached q/k/v/g/beta + conv slice → kill.
- Any host round-trip / copy / replay on accept path in profile → kill.
- Revert bands → revert, no second session.

## 10. Cost

One GPU session, serial. Correctness first (8/8), then metronome bands.
No parallel sweeps, no tuning loop. Needs 5090 release.
