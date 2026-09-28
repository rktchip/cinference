export NINFER_MTP_DEBUG=1
export CUDA_EXL3_AUTOTUNE=0
export SPLIT_TARGET=0
unset NINFER_SLOTS
ulimit -c unlimited
# Self-gate (2026-09-28): never launch without a preflight GO (port/VRAM/idle/paths).
_GATE="$(bash "$(dirname "$0")/s_gate_preflight.sh" pre 2>&1 | tail -n 1)"
[ "$_GATE" = "GO" ] || { echo "LAUNCH-REFUSED (no preflight GO): $_GATE" >&2; exit 1; }
exec /root/ninfer-serve-e28cacb \
  /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --host 127.0.0.1 --port 8902 \
  --greedy \
  --pending-timeout-ms 600000 \
  --max-concurrency 2 --kv-capacity 8192 --no-prefix-reuse >>"$1" 2>&1
