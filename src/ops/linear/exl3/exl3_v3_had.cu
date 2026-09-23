// exl3_v3_had.cu
// Linux-build wrapper: aliases + runtime first, then the vendored TU.
#include "exl3_v3_alias.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "torchexl3/exl3_hadamard.cu"
