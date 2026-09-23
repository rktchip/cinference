// exl3_aln.h
// Cinference raw-pointer host rows for EXL3 (criterion-7 swap of the
// per-row gemv + hand-rolled prefill GEMM for cuda-exl3's M-tiled fused GEMM).
//
// The v3 kernel reads: A (Hadamard-transformed activations, fp16) x trellis
// B, with svh applied in its own epilogue — no cooperative launch, no tracking
// region, plain <<<...>>> (graph-safe). One row = one (row-tile, column-tile)
// unit; the M tile covers batch rows and the full k extent, so trellis bytes
// are read once per tile (criterion-1 leverage for concurrent decode).
//
// Code is from cuda-exl3 (MIT; strands leveraged into torchexl3/ with the ATen
// host layer stripped). Kernel bodies are unmodified except the documented
// seam (graph-safe capture gate, cross-TU linkage, hybrid scale fallback);
// see torchexl3/VENDORED_SHA256.txt for the per-file record.
//
// Workspaces (both caller-owned, sized via exl3_aln_ws_bytes):
//   ahad : [groups * m, k] fp16  — the suh-scaled Hadamard of x, one block per
//          group
//   acc  : [m, n] fp32           — split-k partials; invariant: zeroed before
//          first use. The epilogue zeroes every element it reads, so the
//          invariant holds again after a full row (inline split-k only; a
//          truly-complete row clears everything). Reserving, zeroing, and
//          self-maintaining are the engine's concern.
#pragma once

#include "exl3_dispatch.h"
#include "exl3_v3_alias.h"  // half / bfloat16: this header is v3-row interface only

#include <cstddef>
#include <cstdint>

namespace ninfer {
namespace exl3 {

// Row parameter pack. Fields mirror the front-of-row seam; nothing device-y.
struct Exl3AlnParams {
    const uint16_t* w16 = nullptr;  // [k/16, n/16, 16*bits] trellis (packed)
    const half* suh = nullptr;      // [k]  fp16
    const half* svh = nullptr;      // [n]  fp16
    std::int32_t k = 0;
    std::int32_t n = 0;
    std::int32_t bits = 0;          // K (1..8)
    std::int32_t cb = 0;            // 0=3inst 1=mcg 2=mul1
};

// Geometry probe. Returns the byte count the engine should hold for the
// ahad + acc workspaces (acc counted only when split-k actually triggers for
// that shape).
std::size_t exl3_aln_ws_bytes(const Exl3AlnParams& p, std::int32_t groups, std::int32_t m,
                              bool* need_acc);

// Single-shard row: x [m,k] -> C [m,n]. C dtype selects the overload. Returns
// false only on geometry failure (k or n not a 128-multiple, bits outside
// 1..8); out untouched then.
bool exl3_aln_row_bf16(const bfloat16* x, bfloat16* C, const Exl3AlnParams& p,
                       half* ahad, float* acc, std::int32_t m, cudaStream_t stream);
bool exl3_aln_row_half(const half* x, half* C, const Exl3AlnParams& p,
                       half* ahad, float* acc, std::int32_t m, cudaStream_t stream);

// Multi-group row: `groups` shards packed sequentially along n (qkv_proj
// pattern), each with its own suh row (suh is [groups, k]). group_n[g] is the
// shard width; w16/svh span the full n. Single-group calls use the row above.
bool exl3_aln_rows_bf16(const bfloat16* x, bfloat16* C, const Exl3AlnParams& p,
                        const int* group_n, std::int32_t groups, half* ahad,
                        float* acc, std::int32_t m, cudaStream_t stream);

} // namespace exl3
} // namespace ninfer
