# Serve ticket: SSE disconnect segfault (ship-blocker for any served use)

Status: PROPOSED 2026-09-25. Severity: ship-blocker (production: real
users cancel requests all the time).

## Repro

- Oracle-gated server (any recent build; crash handler in 3dc60be+
  prints the trace, gdb confirms without it).
- Send request #1; let it 503 (default `--pending-timeout-ms` is shorter
  than oracle-slowed steps, ~1s+/step with 3 verifies + host snapshots).
  Client closes the connection on 503.
- Send request #2. Server segfaults mid-stream (exit 139, no FATAL).
- Observed twice (sweep0, mtpA logs). Five clean requests with no
  cancel (gdb1) never crash. Cancel path is the trigger, not the oracle:
  the oracle only makes steps slow enough to hit the timeout.

## Trace

Crash-handler backtrace + addr2line both name the SSE write path:

- `HttpServer::handle_chat_completions(... understated lambda ...)`
  (`openai_chat_http.cpp`)
- via `ContentProviderAdapter` into `httplib::Server::write_response_core`
  → `process_request` → `process_and_close_socket` → `ThreadPool::worker`

The streaming lambda writes to a dead/disconnected sink (use-after-free
or null DataSink). stdout buffering hides the site: last log line is not
where it died (see scripts/harness_notes.md).

## Fix direction (not yet scoped)

Guard the SSE write path against client disconnect (check sink/request
state before each chunk; handle EPIPE/reset without touching freed
memory). Add a cancel-mid-stream regression test: abandon a request
after N chunks, follow with a clean request, assert the server survives
(scripts/cancel_repro.py already drives exactly this).

## Second crash (2026-09-25) — CLOSED 2026-09-25, was the FIRST SSE crash

- WSL dmesg `fatal signal 11` at boot+9298.33s = 17:35:51, matching the
  mtpA SSE crash to the second (req#1 503 at 17:35:50, crash after).
  The "idle death" never happened: PID 1110 was alive the whole time
  (idle in accept()); the pgrep probe failed silently and the conclusion
  was wrong. Correction kept in the record.
- Handler hardening (sigaltstack + SA_ONSTACK) stays — it was a real gap
  regardless. Stats-reporter chase dropped: no crash, no suspect.
- Liveness-probe rule (scripts/harness_notes.md): a check that declares
  a server dead must fail LOUDLY when it cannot tell (never trust a bare
  pgrep with stderr hidden; confirm via /proc/PID/exe + a socket check).
- S1 consequence: pin RECOVERED (/root/ninfer-serve-e28cacb,
  sha256 e71d9919c21d0e1c99235da832bfdc02bd0ed627559ca1656e0e8f61ab1d4605);
  all six arms run from that file with the sha stamped per S-line. The
  worktree rebuild is a spare.

## Workaround (gates only, hides nothing)

`--pending-timeout-ms 600000` on oracle-gated servers: no 503 => no
disconnect => no crash. Every oracle gate since gdb1 uses it.
