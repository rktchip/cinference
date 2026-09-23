// exl3_materialize.h
// EXL3 safetensors materializer for cinference: parses
// quantization_config.json (HF EXL3 family layout) and the safetensors data
// offsets for a Qwen3.8-EXL3 checkpoint, and exposes a per-group layout that
// the loader maps onto device memory. Host-only, raw pointers, no libtorch.
//
// On-disk layout per EXL3 linear group <name> (verified on C:/models/
// Qwen3.8-27B-EXL3-3.5bpw): quantization_config.json holds
//   "tensor_storage": { "<group>";
//     "stored_tensors": {
//        "<group>.suh"     : {shape:[k],   n_bytes, dtype:f16},
//        "<group>.svh"     : {shape:[n],   n_bytes, dtype:f16},
//        "<group>.trellis" : {shape:[k/16,n/16,16K], n_bytes, dtype:i16},
//        optional "<group>.mcg"/"<group>.mul1" : {shape:[], n_bytes:4, dtype:i32} },
//     "quant_format": "exl3", "bits_per_weight": K,
//     optional "mul1_multiplier": int32 }
// Trellis column order is [k/16, n/16, 16K] (dim0 = k/16): proven via lm_head
// (b*16 == 248320 == vocab == n). suh is the k-side (input) Hadamard
// pre-scale, svh is the n-side (output) post-scale — exactly the 1.5.1
// contract (xh = had128(x * suh[k])); see buun
// docs/development/exl3-format-plan.md and 1.5.1 quant/exl3_gemm.cu.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer {
namespace exl3 {

struct Exl3GroupLayout {
    std::string name;
    std::int64_t bits          = 0; // K = bits_per_weight (3/4/5/6 here)
    std::int64_t k             = 0; // feature dim (activations) = trellis[0]*16
    std::int64_t n             = 0; // output dim                     = trellis[1]*16
    std::int64_t trellis_u16   = 0; // trellis.shape[2] = 16*K
    std::int64_t suh_bytes     = 0;
    std::int64_t svh_bytes     = 0;
    std::int64_t trellis_bytes = 0;
    bool has_mcg              = false; // this checkpoint: mcg=0, mul1=409
    bool has_mul1             = false;
    std::int32_t mul1_multiplier = 0;  // checkpoint pins 2212286765
    // Per-sub-tensor byte offsets (safetensors data region) + byte counts.
    // The loader copies each range independently; no contiguity is assumed.
    std::int64_t suh_off       = 0;
    std::int64_t svh_off       = 0;
    std::int64_t tr_off        = 0;
    std::int64_t mcg_off       = -1;   // -1 when absent
    std::int64_t mul1_off      = -1;   // -1 when absent
};

struct Exl3CheckpointMeta {
    long bits_target = 0;      // quantization_config.head_bits (6 here)
    double bpw        = 0;      // configured bpw (3.5 here)
    std::vector<Exl3GroupLayout> groups;
    std::size_t file_count = 0;      // safetensors files named by index.json
    std::vector<std::string> files;
};

// Parses one EXL3 checkpoint directory. Throws std::runtime_error with a
// diagnostic on structural mismatch. `path` is the directory holding
// quantization_config.json, model.safetensors.index.json, and the
// .safetensors payloads. Does NOT read the payload bytes (only offsets +
// extents for each group).
Exl3CheckpointMeta load_exl3_layout(const std::string& path);

// ------ serve-startup probe + packed upload plan (S3, LINUX-ONLY) ------
// exl3_is_exl3_checkpoint_dir is the Engine-construction front door: true
// when `path` is an HF EXL3 checkpoint directory (quantization_config.json
// with tensor_storage + model.safetensors.index.json with weight_map).
// Host-only, never throws (false on any mismatch); reads only the two small
// JSON files, no payload bytes. The "exl3" spelling it looks for is the one
// registered in src/artifact/formats.cpp (QType::EXL3).
bool exl3_is_exl3_checkpoint_dir(const std::string& path);

// Packed device-upload plan for one EXL3 group: byte ranges straight out of
// the shards (trellis [k/16,n/16,16K] int16 + suh [k] fp16 + svh [n] fp16)
// plus the codebook selector (cb: 0=3inst, 1=mcg, 2=mul1) and its 4-byte
// scalar. Materialize uploads these ranges VERBATIM: NO dense unpack of the
// trellis, NO NVFP4/FP8 convert. Sub-tensors of one group may live in
// different shard files, so each range carries its own file name.
struct Exl3PackedUpload {
    std::string name;
    std::int64_t k    = 0;
    std::int64_t n    = 0;
    std::int64_t bits = 0; // K = bits_per_weight
    int cb           = 0;  // 0=3inst 1=mcg 2=mul1
    std::int32_t mul1_multiplier = 0;
    bool has_mcg  = false;
    bool has_mul1 = false;
    std::string suh_file;
    std::int64_t suh_off   = 0;
    std::int64_t suh_bytes = 0;
    std::string svh_file;
    std::int64_t svh_off   = 0;
    std::int64_t svh_bytes = 0;
    std::string tr_file;
    std::int64_t tr_off        = 0;
    std::int64_t trellis_bytes = 0;
    std::string cb_file;      // empty when cb == 0
    std::int64_t cb_off = -1; // -1 when cb == 0
};

// Builds the packed upload plan for every EXL3 group in the checkpoint
// directory `dir` (same directory layout accepted by load_exl3_layout).
// Host-only, throws std::runtime_error on structural mismatch.
std::vector<Exl3PackedUpload> exl3_packed_uploads(const std::string& dir);

// True when every stored tensor of the group maps to an index.json
// data_offset inside the named .safetensors (per weight_map).
// Materializer builds the caller's byte plan from this: suh/svh are fp16
// (0..k, 0..n * 2 B), trellis is fp16 as int16 (trellis_u16 * 2 B),
// in index.json tensor order.
std::vector<const Exl3GroupLayout*> exl3_groups_by_size(const Exl3CheckpointMeta& meta);

// ------ load-time base-vector sanitization (host, byte-exact) ------
// exl3_sanize_range replaces every non-finite fp16 with +0 (dead channel
// -> scale 0 = the conservative identity). NOTE (2026-09-22): the
// Qwen3.8-27B-EXL3-3.5bpw file recounts 0 non-finite suh/svh in all 409
// groups (an earlier 367/401 census read fp16 BITS as int16 VALUES), so
// the sanitizer is currently a no-op safeguard -- it stays because a
// future dirty file would otherwise NaN-poison whole 128-block Hadamard
// outputs, and clean groups pass through byte-identical.
std::int64_t exl3_sanize_range(std::uint16_t* v, std::int64_t count);
std::int64_t exl3_count_nondefinite(const std::uint16_t* v, std::int64_t count);
// Per-group summary over a checkpoint (host byte reads of suh/svh only):
struct Exl3SanitizeSummary {
    std::int64_t groups_total = 0;
    std::int64_t groups_bad_suh = 0;
    std::int64_t groups_bad_svh = 0;
    std::int64_t entries_bad_suh = 0;
    std::int64_t entries_bad_svh = 0;
};
Exl3SanitizeSummary exl3_checkpoint_summary(const std::string& dir);

} // namespace exl3
} // namespace ninfer
