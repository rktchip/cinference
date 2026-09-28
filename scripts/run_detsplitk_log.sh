#!/bin/bash
# run_detsplitk_log.sh BIN LOG [DET] — spec-off :8902 conc-2 kv8192.
BIN="$1"; LOG="$2"; DET="${3:-0}"
export CUDA_EXL3_AUTOTUNE=0 SPLIT_TARGET=0
if [ "$DET" = "1" ]; then export NINFER_EXL3_DETERMINISTIC=1; else unset NINFER_EXL3_DETERMINISTIC; fi
# Self-gate (2026-09-28): never launch without a preflight GO (port/VRAM/idle/paths).
_GATE="$(bash "$(dirname "$0")/s_gate_preflight.sh" pre 2>&1 | tail -n 1)"
[ "$_GATE" = "GO" ] || { echo "LAUNCH-REFUSED (no preflight GO): $_GATE" >&2; exit 1; }
exec "$BIN" /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw --host 127.0.0.1 --port 8902 \
  --greedy --pending-timeout-ms 600000 --max-concurrency 2 --kv-capacity 8192 \
  --no-prefix-reuse >>"$LOG" 2>&1
