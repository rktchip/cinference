# Harness notes (companion to s_gate_preflight.sh)

Durable rig-debug facts. These apply to every future gate, not one ticket.

## Crash triage

- Exit 139 (SIGSEGV) / 134 (SIGABRT) is a HOST-side fault. Look at host
  code on the request path first: rig host buffers, per-request
  bookkeeping, snapshot/restore indexing. A device-side fault normally
  surfaces as a CUDA error (e.g. illegal address), not a killed process.
- Buffered logs do NOT flush on a segfault. The last line logged is not
  necessarily where it died. Trust a backtrace, not the log tail.
- `ninfer-serve` installs a crash handler (SIGSEGV/SIGABRT): it flushes
  stderr, prints a `backtrace_symbols_fd` trace, and re-raises. A crash
  outside gdb — including production — still names its frame.
- Known serve bug (2026-09-25, out of slots scope): a client disconnect
  mid-stream (e.g. after a 503 queue-timeout) segfaults later in the SSE
  write path (`handle_chat_completions` streaming lambda via
  `write_response_core`). Two crashes, both after a 503-cancelled first
  request; five clean requests with no cancel. Oracle gates work around
  it with `--pending-timeout-ms 600000` (no 503 => no disconnect).
  Default queue timeout is too short for oracle-slowed steps (~1s+/step
  with 3 verifies + host snapshots).

## Liveness probe (loud-or-nothing)

- A check that declares a server dead must fail LOUDLY when it cannot
  tell. 2026-09-25: a bare `pgrep -f` failed silently (tool stderr
  hidden) and a live server was declared dead, spawning a false crash
  item. Never trust pgrep alone.
- Loud probe: `ps aux | grep "[n]infer-serve"` (shows the line or
  nothing, visibly) AND `/proc/<pid>/exe` readlink AND a socket check
  (`curl /health`). If any leg is ambiguous, the verdict is UNKNOWN —
  never "dead". A "dead" verdict requires all three legs to agree.

## gdb recipe (WSL)

gdb -batch -ex run -ex bt --args ./apps/ninfer-serve <model> [flags] \
  > gdb.log 2>&1

Release builds keep function symbols (no line info). The crashing
FUNCTION is usually enough; rebuild with debug symbols only if it isn't.

## Pre-registered outcomes (oracle crash hunt, 2026-09-25)

- gdb1 (oracle on, one request, queue timeout raised): crash => fault is
  in the oracle's own request handling; gdb names it. Clean => the
  cancel path (req#1's 503 queue-timeout) is the prime suspect: a lane
  freed/recycled mid-oracle-step, touched by request #2.
- If clean: rerun deliberate cancel-then-request under gdb before
  calling it fixed.

## S-gate rules (2026-09-26 corrections)

- Run identity = pinned exe sha256 (preflight `hash=`); HEAD is info
  only (`head=`). Doc/code commits during arms do NOT void the window;
  only a binary swap does. No commit freeze needed: the code lane stays
  open during timing.
- nproc counts compute apps MINUS known display noise
  (explorer/SearchHost/Widgets/etc, fail-closed). The 5090 drives the
  desktop, so raw nproc is inherently noisy.
- Compare SM clocks UNDER LOAD (client LOADCLK line), not idle stamps.
  Idle clocks bounce hundreds of MHz and mean nothing.
- Mode compare on ms/step, not ms/token: slots-vs-legacy texts diverge
  (width-1 vs width-4 one-ULP tie flips), so ms/token compares different
  workloads. Slots-eager 62.5 vs legacy 64.6 ms/step: slots ~3% cheaper
  per step despite eager verify.
- Token-stream compare protocol: find the FIRST divergence, read the
  logit gap there. Tie = benign. Clear gap = bug, MTP cache first
  suspect. Post-split, divergence should get much rarer.
