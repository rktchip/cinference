export NINFER_MTP_DEBUG=1
export CUDA_EXL3_AUTOTUNE=0
export SPLIT_TARGET=0
unset NINFER_SLOTS
ulimit -c unlimited
exec nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -o /root/speccoff_node -f true \
  /root/ninfer-serve-e28cacb \
  /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --host 127.0.0.1 --port 8902 \
  --greedy \
  --pending-timeout-ms 600000 \
  --max-concurrency 2 --kv-capacity 8192 --no-prefix-reuse
