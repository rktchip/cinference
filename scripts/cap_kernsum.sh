nsys stats --report cuda_gpu_kernsum --format csv \
  --output /tmp/kernsum /root/speccoff_node.nsys-rep 2>&1 | tail -n 3
ls -la /tmp/kernsum*.csv 2>&1
head -n 25 /tmp/kernsum_cuda_gpu_kernsum.csv 2>/dev/null
