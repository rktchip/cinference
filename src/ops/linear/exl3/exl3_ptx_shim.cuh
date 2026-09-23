// Compat shim for the vendored exllamav3 1.5.1 codebook.cuh / exl3_dq.cuh (MIT).
// Pulls in the minimum symbols those two files use, self-contained (no exllamav3 headers).
// Sources: exllamav3_ext/{ptx.cuh, compat.cuh} - ondefinitions copied verbatim.

#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace ninfer {
namespace exl3 {

// ptx.cuh:14
template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};
using FragB = Vec<half2, 2>;
using FragA = Vec<half2, 4>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

// compat.cuh:14 — approximate tanh (sm_75+ path is exact-asm; we are sm_120a)
__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__forceinline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

// ptx.cuh:304 (bfe across a 64-bit pair)
static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint32_t v;
    uint64_t x64 = ((uint64_t)hi << 32) | (uint64_t)lo;
    x64 = (x64 >> offset) & (((uint64_t)-1) >> (64 - length));
    v = (uint32_t)x64;
    return v;
}

// ptx.cuh:314-315
#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm)   asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// util.cuh bit-cast unions (verbatim)
union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};
union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

// util.cuh vector structs (verbatim)
typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
} half4;

typedef struct __align__(8) bfloat164
{
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __device__ bfloat164() = default;
    __device__ bfloat164(__nv_bfloat162 x_, __nv_bfloat162 y_) : x(x_), y(y_) {}
    __device__ bfloat164(__nv_bfloat16 b0, __nv_bfloat16 b1, __nv_bfloat16 b2, __nv_bfloat16 b3) :
         x(__halves2bfloat162(b0, b1)),
         y(__halves2bfloat162(b2, b3)) {}
} bfloat164;

} // namespace exl3
} // namespace ninfer