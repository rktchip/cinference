// exl3_v3_alias.h
// The cuda-exl3 (v3) device headers reference `half` and `bfloat16` unqualified.
// In their torch build those names come from ATen/GlobalState (torch defines
// global aliases); cinference has no libtorch, so we provide the same two
// aliases here, at GLOBAL scope.
//
// Scope discipline: include this ONLY in TUs that compile a torchexl3/ file
// (torchexl3/exl3_gemm.cu, torchexl3/exl3_hadamard.cu, the probe that links
// them). When cinference's own ninfer::exl3 namespace is in scope, `half` and
// `bfloat16` still resolve to the sim's ninfer::exl3 types (inner namespace
// wins) — the global aliases stay inert over cinference code.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

using half = __half;
using bfloat16 = __nv_bfloat16;
