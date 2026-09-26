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

## Second crash (2026-09-25, different signature — NOT the same path)

- S1 arm-A server (e28cacb, slots-eager, `--pending-timeout-ms 600000`,
  15 clean sequential reqs, no 503, no cancel) died after going idle:
  WSL dmesg `fatal signal 11`, no FATAL, and NO crash-handler trace in
  the serve log (handler printed for both earlier crashes).
- Dead process's CUDA context stuck unreclaimed (31.5 GB held, compute
  app `[Not Found]`): WSL did not release VRAM on this death. Next
  server cannot start until it frees; if it sticks, only `wsl --shutdown`
  reclaims (kills builds + downloads too).
- Consequence for S1: /proc pin impossible (process gone AND build dir
  already relinked to the split). Arms B-F run a fresh e28cacb rebuild
  (isolated worktree, same flags/toolchain); S-lines stamp exe sha256 +
  build dir. Speed-risk of the rebuild is argued nil (Release, identical
  codegen inputs; only path strings differ) — recorded, not hidden.
- The idle-death itself is open: no trigger identified (no cancel, no
  timeout, handler silent). If it repeats on arms B-F, it becomes its own
  ticket ahead of the SSE one.

## Workaround (gates only, hides nothing)

`--pending-timeout-ms 600000` on oracle-gated servers: no 503 => no
disconnect => no crash. Every oracle gate since gdb1 uses it.
