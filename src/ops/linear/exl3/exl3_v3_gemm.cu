// exl3_v3_gemm.cu
// Linux-build wrapper: aliases + runtime first, then the vendored TU.
// (Probes use the same pattern via scratch wrappers.)
#include "exl3_v3_alias.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "torchexl3/exl3_gemm.cu"
