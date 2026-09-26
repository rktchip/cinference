#!/bin/bash
# nsys slots-MTP capture: detsplitk binary, DET=0, :8902 conc-2 kv8192.
export NINFER_SLOTS=1
export NINFER_MTP_DEBUG=1
export CUDA_EXL3_AUTOTUNE=0
export SPLIT_TARGET=0
unset NINFER_EXL3_DETERMINISTIC
ulimit -c unlimited
exec nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -o /root/slotsmtp_node -f true \
  /root/ninfer-serve-detsplitk-528b29e \
  /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --host 127.0.0.1 --port 8902 \
  --greedy \
  --pending-timeout-ms 600000 \
  --max-concurrency 2 --kv-capacity 8192 --no-prefix-reuse \
  --spec mtp --draft-tokens 3 >>"$1" 2>&1
