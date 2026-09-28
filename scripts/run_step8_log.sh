BIN=/root/ninfer-serve-step4-9b9b54b7
LOG="$2"
CAP="$1"
export NINFER_MTP_DEBUG=1
export CUDA_EXL3_AUTOTUNE=0
export SPLIT_TARGET=0
unset NINFER_SLOTS
if [ -n "$CAP" ]; then export NINFER_MTP_MAX_ACCEPT="$CAP"; fi
# Self-gate (2026-09-28): never launch without a preflight GO (port/VRAM/idle/paths).
_GATE="$(bash "$(dirname "$0")/s_gate_preflight.sh" pre 2>&1 | tail -n 1)"
[ "$_GATE" = "GO" ] || { echo "LAUNCH-REFUSED (no preflight GO): $_GATE" >&2; exit 1; }
exec "$BIN" /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --host 127.0.0.1 --port 8903 \
  --pending-timeout-ms 600000 --max-concurrency 2 \
  --kv-capacity 8192 --no-prefix-reuse --greedy >>"$LOG" 2>&1
