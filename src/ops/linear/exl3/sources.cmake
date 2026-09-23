# EXL3 decode path (m <= 8): reference 1.5.1 pack layer (verbatim) + plain
# graph-safe carve + host launcher (had_in -> gemv -> had_out -> cast).
# Prefill path (m > 8): exl3_prefill.cu — verbatim 1.5.1 reconstruct kernels
# (fused + rotated) + had planes + SIMT fp16 GEMM + cast (numeric-gated vs
# 1.5.1 production, P12b).
# Materializer: exl3_materialize.cpp — flat+wrapped safetensors -> per-group
# geometry + base-vector non-finite census/sanitizer (P10/P13).
# Not RDC-safe in the cooperative version; plain carve uses regular launches.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/exl3_launcher.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_dispatch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_prefill.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_materialize.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_aln.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_op.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_bind.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_v3_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_v3_had.cu"
)