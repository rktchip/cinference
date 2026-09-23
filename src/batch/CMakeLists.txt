add_library(ninfer_batch STATIC
  paged_kv.cu
  request.cc
  scheduler.cc
  batch.cu
  cinference_hooks.cc
)
ninfer_internal_includes(ninfer_batch)
ninfer_cuda_archive(ninfer_batch)
target_link_libraries(ninfer_batch PUBLIC CUDA::cudart)
