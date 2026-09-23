// exl3_prefill.h
// EXL3 prefill (m > 8) + reconstruct (K >= 5) path, raw-pointer host contract.
//
// Convention (1.5.1 LinearEXL3.reconstruct_hgemm, rows < 1024, verbatim):
//   x   -> had_r_128 with suh PRE-scale      (A_h)
//   W16 = naked unpacked [k, n] fp16 (exl3_reconstruct_rot = 1.5.1
//         reconstruct_slice; byte-verified vs the 1.5.1 pyd in P12)
//   y   = GEMM(A_h, W16) -> had_r_128 with svh POST-scale -> bf16
// The FUSED variant (exl3_reconstruct_had) bakes both H128 + suh/svh into
// W16 and consumes RAW x; do NOT chain had planes onto it (double-H).
//
// Had plane contract: 32 threads/128-plane, grid (rows, planes) -- the
// vendored inner reads its scale window at blockIdx.y*32+t (1.5.1
// hadamard.cu:143-147), so PLANE = dim 2 and the scale pointer is the
// full suh/svh vector. (P12b audit 2026-09-22 fixed the (planes, rows)
// bug that made every row beyond 0 read plane-0's window.)
#pragma once

#include "exl3_dispatch.h"
#include "exl3_reconstruct.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace ninfer {
namespace exl3 {

// Raw-pointer dequant dispatch over the vendored 1.5.1 instance arrays.
//   rot : out [k, n] naked (rot) basis  -- grid (n/128, k/16), 256 thr
//   had : out [k, n] original basis (H+suh/svh baked) -- grid (n/128, k/128)
bool exl3_reconstruct_rot(Exl3ReconstructArgs& a, cudaStream_t stream);
bool exl3_reconstruct_had(Exl3ReconstructArgs& a, cudaStream_t stream);

// One-shot prefill (m >= 9): A [m, k] fp16 contiguous -> A_h -> GEMM ->
// C16 -> had_out(svh) -> bf16 out [m, n]. W16 = naked [k, n] (caller
// dequantized via exl3_reconstruct_rot). scratch bytes: m*k*2 + m*n*2.
// Returns false on geometry failure (out untouched).
bool exl3_prefill_op(const half* A, const half* W16, uint16_t* outbf, half* scratch,
                     const Exl3Weight& w, std::int32_t m, cudaStream_t stream);

std::size_t exl3_prefill_workspace_bytes(std::int32_t m, std::int32_t k, std::int32_t n);

} // namespace exl3
} // namespace ninfer
