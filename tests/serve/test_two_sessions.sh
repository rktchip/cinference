#!/usr/bin/env bash
# tests/serve/test_two_sessions.sh
# S5 proof: two concurrent HTTP sessions against one ninfer-serve instance.
#
# Linux-only (uses process groups, /proc-free liveness polling, and bash
# coproc-free background jobs; the served engine needs a CUDA device).
# Script only: never run from the shared-GPU Windows session.
#
# Layout: start server -> wait /health -> fire two concurrent
# POST /v1/chat/completions clients with distinct prompts -> both must
# answer HTTP 200 with a non-empty choice text -> kill server.
#
# Env:
#   NINFER_SERVE_BIN   server binary (default: ninfer-serve on PATH)
#   NINFER_TEST_MODEL  model artifact path (required to run; without it the
#                      script exits 77 so ctest records SKIP, not FAIL)
#   NINFER_TEST_HOST   bind host (default 127.0.0.1)
#   NINFER_TEST_PORT   bind port (default 18080)
set -u

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "SKIP: test_two_sessions is Linux-only (got $(uname -s))"
  exit 77
fi

BIN="${NINFER_SERVE_BIN:-ninfer-serve}"
HOST="${NINFER_TEST_HOST:-127.0.0.1}"
PORT="${NINFER_TEST_PORT:-18080}"
MODEL="${NINFER_TEST_MODEL:-}"

if [[ -z "$MODEL" ]]; then
  echo "SKIP: NINFER_TEST_MODEL is not set (no model artifact to serve)"
  exit 77
fi
if ! command -v "$BIN" >/dev/null 2>&1; then
  echo "SKIP: server binary '$BIN' not on PATH"
  exit 77
fi
for tool in curl python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "SKIP: required tool '$tool' not on PATH"
    exit 77
  fi
done

WORK="$(mktemp -d)"
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; rm -rf "$WORK"' EXIT

"$BIN" "$MODEL" --host "$HOST" --port "$PORT" --greedy >"$WORK/server.log" 2>&1 &
SERVER_PID=$!

# Wait for /health (30 s budget).
READY=0
for _ in $(seq 1 150); do
  if curl -fs -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited during startup; log:"
    cat "$WORK/server.log"
    exit 1
  fi
  sleep 0.2
done
if [[ "$READY" != "1" ]]; then
  echo "FAIL: server never became healthy; log:"
  cat "$WORK/server.log"
  exit 1
fi

# Two concurrent clients, distinct prompts. Greedy decode keeps the check
# deterministic: we only require HTTP 200 plus non-empty choice text.
client() {
  local name="$1" prompt="$2" out="$3"
  curl -s -m 120 -o "$out" -w "%{http_code}" \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"test\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"max_tokens\": 16}" \
    >"$out.code" 2>"$out.err" &
  echo $!
}

PID_A=$(client "a" "Session A: reply with the word ALPHA." "$WORK/a.json")
PID_B=$(client "b" "Session B: reply with the word BETA." "$WORK/b.json")
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?

FAIL=0
for tag in a b; do
  CODE="$(cat "$WORK/$tag.json.code")"
  if [[ "$CODE" != "200" ]]; then
    echo "FAIL: client $tag HTTP $CODE"
    cat "$WORK/$tag.json" "$WORK/$tag.json.err"
    FAIL=1
    continue
  fi
  TEXT="$(python3 -c "import json,sys; d=json.load(open('$WORK/$tag.json')); print(d['choices'][0]['message']['content'])" 2>/dev/null)"
  if [[ -z "$TEXT" ]]; then
    echo "FAIL: client $tag returned no choice text"
    cat "$WORK/$tag.json"
    FAIL=1
  else
    echo "client $tag ok (${#TEXT} chars)"
  fi
done
if [[ "$RC_A" != "0" || "$RC_B" != "0" ]]; then
  echo "FAIL: curl exit codes $RC_A/$RC_B"
  FAIL=1
fi

if [[ "$FAIL" == "0" ]]; then
  echo "two_sessions: PASS (two concurrent clients, one server)"
else
  echo "two_sessions: FAIL (see above; server log: $WORK/server.log)"
fi
exit "$FAIL"
