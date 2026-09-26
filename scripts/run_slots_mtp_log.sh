#!/bin/bash
# run_slots_mtp_log.sh BIN LOG [DET] — slots MTP draft-3 :8902 conc-2 kv8192.
BIN="$1"; LOG="$2"; DET="${3:-0}"
export NINFER_SLOTS=1
export NINFER_MTP_DEBUG=1
export CUDA_EXL3_AUTOTUNE=0
export SPLIT_TARGET=0
if [ "$DET" = "1" ]; then export NINFER_EXL3_DETERMINISTIC=1; else unset NINFER_EXL3_DETERMINISTIC; fi
ulimit -c unlimited
exec "$BIN" /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw --host 127.0.0.1 --port 8902 \
  --greedy --pending-timeout-ms 600000 --max-concurrency 2 --kv-capacity 8192 \
  --no-prefix-reuse --spec mtp --draft-tokens 3 >>"$LOG" 2>&1
