// exl3_weights.h
// EXL3 serve-startup weight seam for qwen3_5 (single slot).
//
// The standard artifact path (Binder -> materialize -> native_weight) cannot
// see an HF EXL3 checkpoint directory: there is no .ninfer Reader, and the
// fused EXL3 side-cars (attention/qkv groups=3, mlp/gate_up groups=2) have no
// single-parent byte image. This header bridges the two sides with no CUDA
// types, so host-compiled model TUs (parameters.cpp) and the host-compiled
// runtime builder (engine/exl3_program.cpp) share one lookup:
//
//   * the device side-cars live in the process-lifetime Exl3EngineStore
//     (ops/linear/exl3/exl3_bind.cu, Linux deployment path), keyed by the
//     binder logical names (text/layers/...);
//   * the declarations below name that store opaquely. Signatures must match
//     exl3_bind.h exactly (same pattern as serve/generation_service.cpp).
//   * exl3_make_weight turns (logical name, n, k) into an execution Weight
//     whose payload borrows the store side-car. Geometry comes from the
//     caller (config-derived widths, cross-checked against the fusion plan
//     at bind time), because the store struct itself is __CUDACC__-only.
//
// GDN convention (single slot): GdnParameters.projection is a Single whose
// Weight covers the qkv side-car (n = q+k+v rows); its qdata field (unused
// by exl3_dispatch, which reads only payload/n/k) carries the sibling z
// side-car pointer, so the gdn_input_proj EXL3 arm can serve qkv and z with
// two dispatches and no workspace temp. Non-GDN EXL3 Weights leave qdata
// null.
//
// Lifetime: side-car pointers borrow the process store (freed only at
// teardown, same as the store itself). Weights built here must not outlive
// the store; in practice they live inside the serving Model, which is
// shorter-lived than the process.
#pragma once

#include "core/weight.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace exl3 {

struct Exl3EngineStore;
struct Exl3Weight;

const Exl3Weight* exl3_engine_find(const Exl3EngineStore& store,
                                   const std::string& name) noexcept;
Exl3EngineStore& exl3_process_store() noexcept;

} // namespace exl3

namespace models::qwen3_5::loading {

// Payload side-car for one EXL3 logical linear, or nullptr when the name is
// not an EXL3 linear in the process store. Never throws.
[[nodiscard]] inline const void* exl3_find_payload(const std::string& logical)
{
    const exl3::Exl3Weight* found =
        exl3::exl3_engine_find(exl3::exl3_process_store(), logical);
    return static_cast<const void*>(found);
}

// Execution Weight for one EXL3 logical linear. The payload borrows the
// process store (see above); n/k mirror the side-car geometry. Throws
// std::invalid_argument when the name is not in the store or the extents
// are not positive.
[[nodiscard]] inline Weight exl3_make_weight(const std::string& logical, std::int32_t n,
                                            std::int32_t k)
{
    const void* payload = exl3_find_payload(logical);
    if (payload == nullptr || n <= 0 || k <= 0) {
        throw std::invalid_argument("EXL3 linear is missing from the process store: " +
                                    logical);
    }
    Weight out;
    out.payload = payload;
    out.qtype   = QType::EXL3;
    out.layout  = QuantLayout::Contiguous;
    out.ndim    = 2;
    out.n = out.shape[0] = out.padded_shape[0] = n;
    out.k = out.shape[1] = out.padded_shape[1] = k;
    return out;
}

} // namespace models::qwen3_5::loading
} // namespace ninfer
