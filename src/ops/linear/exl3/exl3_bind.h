// exl3_bind.h
// EXL3 side-car builder: the materializer's call-down.
//
// Census decision (2026-09-22, 401 EXL3 groups on
// C:/models/Qwen3.8-27B-EXL3-3.5bpw: K3x137, K4x262, K5x1 o_proj L63, K6x1
// lm_head; every group uniform-K, all mul1). S3c (2026-09-22): the scan
// covers 409 trellis groups, not 401 -- quantization_config tensor_storage
// omits the 8 mtp.* groups (mtp.fc draft head + mtp.layers.0.mlp.{down,
// gate,up} + mtp.layers.0.self_attn.{k,o,q,v}; all K4 mul1), which
// exl3_append_mtp_groups harvests from weight_map + shard headers below.
// The binder name-map (S3b,
// host-only section below) fuses self_attn q/k/v into one logical linear
// per full-attention layer (groups=3, group_n = [12288,1024,1024]) and mlp
// gate/up into one logical linear per layer (groups=2, group_n halves
// [17408,17408]), so fused layers serve multi-group one-launch via the v3
// fused multi row. Everything else stays dense (groups=1): the linear-attn
// in_proj_qkv single group, in_proj_z, out_proj, down_proj, o_proj, lm_head.
// K>=5 (o_proj K5, lm_head K6): v3 instantiates BITS 1..8 and the K5/K6
// gate (2026-09-22) proves finite serving at production amplitudes
// (ramp + randn s=2, maxAbs <=0.04), so dense K5/K6 entries carry ok=true
// (v3-servable; plan stays empty -- the legacy gemv template caps at K4
// and the dispatch bits gate keeps K5/K6 off that route). Fused entries
// likewise carry ok=true with an empty plan: detail::exl3_dispatch routes
// groups>1 to the v3 multi row for any m and never consults the plan.
//
// The file is split for toolchain reasons: the name-map + fusion plan are
// pure host C++ (visible in every TU, including cl-compiled ones and the
// host-only unit test), while the side-car/store section needs CUDA device
// types plus the v3 launcher chain, so it is visible only under nvcc
// (__CUDACC__). Host TUs (e.g. model_instance.cpp) drive construction
// through exl3_engine_construct_from_dir, which takes the stream opaque.
#pragma once

#include "ops/linear/exl3/exl3_materialize.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef __CUDACC__
#include "ops/linear/exl3/exl3_dispatch.h"

#include <cuda_runtime.h>
#endif

namespace ninfer {
namespace exl3 {

// ------ binder name-map + fusion plan (S3b, HOST-ONLY) ------
// Maps HF tensor_storage groups to logical .ninfer linears. A logical entry
// is either dense (one HF group, groups=1) or fused (q/k/v or gate/up
// members concatenated along n, groups=2..3, served one-launch by the v3
// fused multi row: trellis spans the full n, suh stacks [groups,k], svh
// spans the full n, group_n[g] widths sum to n). The planner is total: a
// fusion set that is incomplete (missing member) or non-uniform (k, K, or
// codebook differ) falls back to dense singles, preserving the pre-S3b
// 1:1 load behavior for that set instead of failing the whole checkpoint.
struct Exl3LogicalEntry {
    std::string logical;                     // .ninfer name (engine store key)
    std::vector<std::string> members;        // HF checkpoint groups, serve order
    std::vector<std::string> member_logical; // .ninfer name per member
    std::int64_t k    = 0;
    std::int64_t n    = 0;
    std::int64_t bits = 0; // K = bits_per_weight
    int cb           = 0;  // 0=3inst 1=mcg 2=mul1
    bool has_mcg     = false;
    bool has_mul1    = false;
    std::int32_t mul1_multiplier = 0;
    std::int32_t groups     = 1;
    std::int32_t group_n[8] = {};
    // Staged bytes (empty until exl3_stage_fused_host runs): fused trellis
    // [k/16, n/16, 16K] u16, suh [groups*k], svh [n].
    std::vector<std::uint16_t> trellis;
    std::vector<std::uint16_t> suh;
    std::vector<std::uint16_t> svh;
};

// HF checkpoint group -> logical .ninfer linear. "model.language_model.
// layers.{i}." becomes "text/layers/{i}/" with the projection suffix mapped
// (self_attn.q_proj -> attention/query, mlp.gate_proj -> mlp/gate, ...);
// lm_head -> text/output_head, mtp.fc (draft head) -> text/mtp/draft_head,
// mtp.layers.{i}.* -> text/mtp/layers/{i}/* (same projection suffix table
// as text/layers). Fused pseudo-groups use the briefing names:
// ".self_attn.qkv_proj" -> "attention/qkv", ".mlp.gate_up_proj" ->
// "mlp/gate_up". Unknown spellings fall back to the HF name verbatim (the
// loader still serves them dense); never throws.
inline std::string exl3_map_group_to_logical(const std::string& hf)
{
    if (hf == "lm_head") return "text/output_head";
    if (hf == "mtp.fc") return "text/mtp/draft_head"; // MTP draft head
    // MTP draft layer mirrors the text/layers scheme under text/mtp. This
    // branch runs before the generic ".layers." branch below: "mtp.layers."
    // contains ".layers." and would otherwise collide with the real
    // text/layers/{i} names.
    const std::string mtpfx = "mtp.layers.";
    if (hf.compare(0, mtpfx.size(), mtpfx) == 0) {
        const std::string tail = hf.substr(mtpfx.size()); // "<i>.<rest...>"
        const std::string::size_type dot = tail.find('.');
        if (dot == std::string::npos) return hf;
        const std::string idx = tail.substr(0, dot);
        if (idx.empty()) return hf;
        for (char c : idx) {
            if (c < '0' || c > '9') return hf;
        }
        const std::string rest = tail.substr(dot + 1);
        const std::string base = "text/mtp/layers/" + idx + "/";
        if (rest == "self_attn.q_proj") return base + "attention/query";
        if (rest == "self_attn.k_proj") return base + "attention/key";
        if (rest == "self_attn.v_proj") return base + "attention/value";
        if (rest == "self_attn.o_proj") return base + "attention/output";
        if (rest == "self_attn.qkv_proj") return base + "attention/qkv";
        if (rest == "mlp.gate_proj") return base + "mlp/gate";
        if (rest == "mlp.up_proj") return base + "mlp/up";
        if (rest == "mlp.down_proj") return base + "mlp/down";
        if (rest == "mlp.gate_up_proj") return base + "mlp/gate_up";
        return hf;
    }
    const std::string tok = ".layers.";
    const std::string::size_type p = hf.find(tok);
    if (p == std::string::npos) return hf;
    const std::string tail = hf.substr(p + tok.size()); // "<i>.<rest...>"
    const std::string::size_type dot = tail.find('.');
    if (dot == std::string::npos) return hf;
    const std::string idx = tail.substr(0, dot);
    if (idx.empty()) return hf;
    for (char c : idx) {
        if (c < '0' || c > '9') return hf;
    }
    const std::string rest = tail.substr(dot + 1);
    const std::string base = "text/layers/" + idx + "/";
    if (rest == "self_attn.q_proj") return base + "attention/query";
    if (rest == "self_attn.k_proj") return base + "attention/key";
    if (rest == "self_attn.v_proj") return base + "attention/value";
    if (rest == "self_attn.o_proj") return base + "attention/output";
    if (rest == "self_attn.qkv_proj") return base + "attention/qkv";
    if (rest == "self_attn.q_norm") return base + "attention/query_norm";
    if (rest == "self_attn.k_norm") return base + "attention/key_norm";
    if (rest == "linear_attn.in_proj_qkv") return base + "gdn/qkv";
    if (rest == "linear_attn.in_proj_z") return base + "gdn/z";
    if (rest == "linear_attn.out_proj") return base + "gdn/output";
    if (rest == "mlp.gate_proj") return base + "mlp/gate";
    if (rest == "mlp.up_proj") return base + "mlp/up";
    if (rest == "mlp.down_proj") return base + "mlp/down";
    if (rest == "mlp.gate_up_proj") return base + "mlp/gate_up";
    if (rest == "input_layernorm") return base + "input_norm";
    if (rest == "post_attention_layernorm") return base + "post_attention_norm";
    return hf;
}

// Builds the logical plan for a parsed checkpoint (meta order preserved:
// fused entries emit at their first member's position). Fuses complete,
// uniform self_attn q/k/v triples (serve order q,k,v) and mlp gate/up
// pairs (serve order gate,up); everything else is dense. Never throws on
// names (unknown spellings map verbatim); callers validate geometry at
// stage/upload time with byte-exact diagnostics.
inline std::vector<Exl3LogicalEntry> exl3_plan_fusion(const Exl3CheckpointMeta& meta)
{
    struct SetInfo {
        int kind = 0; // 3 = qkv, 2 = gate_up
        const Exl3GroupLayout* slot[3] = {};
    };
    std::map<std::string, SetInfo> sets;
    std::map<const Exl3GroupLayout*, std::pair<std::string, int>> role; // group -> (set key, slot)
    for (const auto& g : meta.groups) {
        const std::string::size_type dot = g.name.rfind('.');
        if (dot == std::string::npos) continue;
        const std::string head = g.name.substr(0, dot);
        const std::string comp = g.name.substr(dot + 1);
        const bool is_attn = head.size() >= 10 &&
                             head.compare(head.size() - 10, 10, ".self_attn") == 0;
        const bool is_mlp = head.size() >= 4 &&
                            head.compare(head.size() - 4, 4, ".mlp") == 0;
        int kind = 0, slot = -1;
        std::string fkey;
        if (is_attn && (comp == "q_proj" || comp == "k_proj" || comp == "v_proj")) {
            kind = 3;
            slot = (comp == "q_proj") ? 0 : ((comp == "k_proj") ? 1 : 2);
            fkey = head + ".qkv_proj";
        } else if (is_mlp && (comp == "gate_proj" || comp == "up_proj")) {
            kind = 2;
            slot = (comp == "gate_proj") ? 0 : 1;
            fkey = head + ".gate_up_proj";
        } else {
            continue;
        }
        SetInfo& s = sets[fkey];
        s.kind = kind;
        if (s.slot[slot] == nullptr) {
            s.slot[slot] = &g;
            role[&g] = {fkey, slot};
        }
        // Duplicate group names (corrupt layout): keep the first, leave the
        // duplicate dense. The stage step re-validates bytes per member.
    }
    // A set fuses only when complete AND uniform (k, K, codebook, multiplier).
    std::map<std::string, bool> fuse;
    for (const auto& kv : sets) {
        const SetInfo& s = kv.second;
        bool ok = true;
        for (int i = 0; i < s.kind; ++i) {
            if (s.slot[i] == nullptr) { ok = false; break; }
        }
        if (ok) {
            const Exl3GroupLayout* m0 = s.slot[0];
            for (int i = 1; i < s.kind; ++i) {
                const Exl3GroupLayout* m = s.slot[i];
                if (m->k != m0->k || m->bits != m0->bits || m->has_mcg != m0->has_mcg ||
                    m->has_mul1 != m0->has_mul1 ||
                    m->mul1_multiplier != m0->mul1_multiplier) {
                    ok = false;
                    break;
                }
            }
        }
        fuse[kv.first] = ok;
    }
    std::vector<Exl3LogicalEntry> plan;
    plan.reserve(meta.groups.size());
    auto emit_dense = [&](const Exl3GroupLayout& g) {
        Exl3LogicalEntry e;
        e.logical = exl3_map_group_to_logical(g.name);
        e.members.push_back(g.name);
        e.member_logical.push_back(e.logical);
        e.k     = g.k;
        e.n     = g.n;
        e.bits  = g.bits;
        e.cb    = g.has_mul1 ? 2 : (g.has_mcg ? 1 : 0);
        e.has_mcg  = g.has_mcg;
        e.has_mul1 = g.has_mul1;
        e.mul1_multiplier = g.mul1_multiplier;
        e.groups = 1;
        plan.push_back(std::move(e));
    };
    for (const auto& g : meta.groups) {
        const auto rit = role.find(&g);
        if (rit == role.end()) {
            emit_dense(g);
            continue;
        }
        const std::string& fkey = rit->second.first;
        const int slot = rit->second.second;
        const SetInfo& s = sets[fkey];
        if (!fuse[fkey]) {
            emit_dense(g); // incomplete/non-uniform set: dense fallback
            continue;
        }
        if (slot != 0) continue; // fused entry emits at its first member
        Exl3LogicalEntry e;
        // fkey is "<hf head>.qkv_proj" / "<hf head>.gate_up_proj": the map
        // function knows those fused spellings.
        e.logical = exl3_map_group_to_logical(fkey);
        e.k    = s.slot[0]->k;
        e.bits = s.slot[0]->bits;
        e.cb   = s.slot[0]->has_mul1 ? 2 : (s.slot[0]->has_mcg ? 1 : 0);
        e.has_mcg  = s.slot[0]->has_mcg;
        e.has_mul1 = s.slot[0]->has_mul1;
        e.mul1_multiplier = s.slot[0]->mul1_multiplier;
        e.groups = (std::int32_t) s.kind;
        e.n = 0;
        for (int i = 0; i < s.kind; ++i) {
            e.members.push_back(s.slot[i]->name);
            e.member_logical.push_back(exl3_map_group_to_logical(s.slot[i]->name));
            e.group_n[i] = (std::int32_t) s.slot[i]->n;
            e.n += s.slot[i]->n;
        }
        plan.push_back(std::move(e));
    }
    return plan;
}

// Appends checkpoint trellis groups that load_exl3_layout misses because
// they have no quantization_config tensor_storage entry, derived straight
// from model.safetensors.index.json + the shard headers with the same
// strictness (trellis [k/16,n/16,16K] i16, suh [k] f16, svh [n] f16,
// mcg/mul1 i32-scalar xor, byte counts match, every sub-tensor resolved in
// weight_map and its shard header). On C:/models/Qwen3.8-27B-EXL3-3.5bpw
// that is exactly the 8 mtp.* groups (mtp.fc draft head +
// mtp.layers.0.mlp.{down,gate,up} + mtp.layers.0.self_attn.{k,o,q,v}),
// lifting the scan 401 -> 409; groups already in `meta` are skipped, so
// calling this on a fully-covered layout is a no-op returning 0.
// mul1_multiplier stays 0: the file scalar is adopted at stage time
// (exl3_stage_fused_host tolerates a zero seed and enforces member
// agreement), and a zero seed is uniform across the appended set so the
// fusion planner treats mtp triples/pairs exactly like text ones.
// Appends in weight_map order. Throws std::runtime_error with a diagnostic
// on structural mismatch. Host-only file I/O, no CUDA calls.
inline std::size_t exl3_append_mtp_groups(const std::string& dirpath, Exl3CheckpointMeta& meta)
{
    const std::string dir = dirpath + (dirpath.empty() || dirpath.back() == '/' ? "" : "/");
    std::ifstream fi(dir + "model.safetensors.index.json", std::ios::binary);
    if (!fi) throw std::runtime_error("exl3 mtp scan: cannot open " + dir + "model.safetensors.index.json");
    nlohmann::json index;
    try {
        fi >> index;
    } catch (...) {
        throw std::runtime_error("exl3 mtp scan: bad JSON in model.safetensors.index.json");
    }
    if (!index.is_object() || !index.contains("weight_map") || !index["weight_map"].is_object())
        throw std::runtime_error("exl3 mtp scan: weight_map missing");
    const nlohmann::json& weight_map = index["weight_map"];

    std::set<std::string> known;
    for (const auto& g : meta.groups) known.insert(g.name);

    struct ShardTensors {
        std::map<std::string, std::pair<std::int64_t, std::int64_t>> offsets; // name -> (off, bytes)
        std::map<std::string, nlohmann::json> recs; // name -> header record
    };
    std::map<std::string, ShardTensors> shards;
    auto shard_of = [&](const std::string& tensor_name) -> const ShardTensors& {
        const auto wit = weight_map.find(tensor_name);
        if (wit == weight_map.end() || !wit.value().is_string())
            throw std::runtime_error("exl3 mtp scan: " + tensor_name + " missing in weight_map");
        const std::string fname = wit.value().get<std::string>();
        const auto sit = shards.find(fname);
        if (sit != shards.end()) return sit->second;
        std::ifstream f(dir + fname, std::ios::binary);
        if (!f) throw std::runtime_error("exl3 mtp scan: cannot open " + dir + fname);
        std::uint64_t hdr = 0;
        f.read(reinterpret_cast<char*>(&hdr), 8);
        if (!f || hdr == 0 || hdr > (1ull << 31))
            throw std::runtime_error("exl3 mtp scan: bad safetensors header size in " + fname);
        std::string text(static_cast<std::size_t>(hdr), '\0');
        f.read(text.data(), static_cast<std::streamsize>(hdr));
        if (!f) throw std::runtime_error("exl3 mtp scan: short safetensors header in " + fname);
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(text);
        } catch (...) {
            throw std::runtime_error("exl3 mtp scan: bad safetensors header JSON in " + fname);
        }
        if (!j.is_object()) throw std::runtime_error("exl3 mtp scan: header not an object in " + fname);
        const nlohmann::json* tensors = &j;
        if (j.contains("tensors") && j["tensors"].is_object()) tensors = &j["tensors"];
        ShardTensors st;
        for (const auto& e : tensors->items()) {
            const nlohmann::json& rec = e.value();
            if (!rec.is_object() || !rec.contains("data_offsets") || !rec["data_offsets"].is_array() ||
                rec["data_offsets"].size() != 2)
                continue;
            const std::int64_t off = rec["data_offsets"][0].get<std::int64_t>();
            const std::int64_t end = rec["data_offsets"][1].get<std::int64_t>();
            if (end < off)
                throw std::runtime_error("exl3 mtp scan: bad data_offsets for " + e.key());
            st.offsets[e.key()] = {off, end - off};
            st.recs[e.key()] = rec;
        }
        return shards.emplace(fname, std::move(st)).first->second;
    };
    auto shape_len = [](const nlohmann::json& shape) -> std::int64_t {
        std::int64_t n = 1;
        for (const auto& dim : shape) n *= dim.get<std::int64_t>();
        return n;
    };
    auto is_dtype = [](const nlohmann::json& rec, const char* short_sp, const char* torch_sp) -> bool {
        if (!rec.contains("dtype") || !rec["dtype"].is_string()) return false;
        const std::string d = rec["dtype"].get<std::string>();
        return d == short_sp || d == torch_sp;
    };

    std::size_t appended = 0;
    for (const auto& e : weight_map.items()) {
        const std::string& txn = e.key();
        const std::string suf = ".trellis";
        if (txn.size() <= suf.size() || txn.compare(txn.size() - suf.size(), suf.size(), suf) != 0)
            continue;
        const std::string grp = txn.substr(0, txn.size() - suf.size());
        if (known.count(grp)) continue;
        const std::string ctx = "group " + grp;
        const std::string t_suh = grp + ".suh";
        const std::string t_svh = grp + ".svh";
        if (weight_map.find(t_suh) == weight_map.end() || !weight_map[t_suh].is_string())
            throw std::runtime_error("exl3 mtp scan: " + ctx + " suh missing in weight_map");
        if (weight_map.find(t_svh) == weight_map.end() || !weight_map[t_svh].is_string())
            throw std::runtime_error("exl3 mtp scan: " + ctx + " svh missing in weight_map");
        const bool mg_in = weight_map.find(grp + ".mcg") != weight_map.end();
        const bool ml_in = weight_map.find(grp + ".mul1") != weight_map.end();
        if (mg_in && ml_in) throw std::runtime_error("exl3 mtp scan: " + ctx + " both mcg and mul1 present");

        const ShardTensors& tr_sh = shard_of(txn);
        const ShardTensors& su_sh = shard_of(t_suh);
        const ShardTensors& sv_sh = shard_of(t_svh);
        auto lookup = [&](const ShardTensors& st, const std::string& t) -> const nlohmann::json& {
            const auto it = st.recs.find(t);
            if (it == st.recs.end())
                throw std::runtime_error("exl3 mtp scan: " + ctx + " " + t + " missing in shard header");
            return it->second;
        };
        const nlohmann::json& tr = lookup(tr_sh, txn);
        const nlohmann::json& su = lookup(su_sh, t_suh);
        const nlohmann::json& sv = lookup(sv_sh, t_svh);
        if (!is_dtype(tr, "I16", "torch.int16")) throw std::runtime_error("exl3 mtp scan: " + ctx + " trellis dtype");
        if (!is_dtype(su, "F16", "torch.float16")) throw std::runtime_error("exl3 mtp scan: " + ctx + " suh dtype");
        if (!is_dtype(sv, "F16", "torch.float16")) throw std::runtime_error("exl3 mtp scan: " + ctx + " svh dtype");
        if (!tr.contains("shape") || !tr["shape"].is_array() || tr["shape"].size() != 3)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " trellis shape");
        const std::int64_t d0 = tr["shape"][0].get<std::int64_t>();
        const std::int64_t d1 = tr["shape"][1].get<std::int64_t>();
        const std::int64_t d2 = tr["shape"][2].get<std::int64_t>();
        if (d0 < 1 || d1 < 1 || d2 % 16 != 0)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " trellis dims");
        const std::int64_t bits = d2 / 16;
        if (bits < 1 || bits > 8) throw std::runtime_error("exl3 mtp scan: " + ctx + " K must be 1..8");

        Exl3GroupLayout l;
        l.name = grp;
        l.bits = bits;
        l.mul1_multiplier = 0; // file scalar adopted at stage time
        l.k = d0 * 16;
        l.n = d1 * 16;
        l.trellis_u16 = d2;
        l.trellis_bytes = d0 * d1 * d2 * 2;
        if (!su.contains("shape") || shape_len(su["shape"]) != l.k)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " suh size != k");
        l.suh_bytes = l.k * 2;
        if (!sv.contains("shape") || shape_len(sv["shape"]) != l.n)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " svh size != n");
        l.svh_bytes = l.n * 2;
        auto byte_count = [&](const ShardTensors& st, const std::string& t) -> std::int64_t {
            const auto it = st.offsets.find(t);
            if (it == st.offsets.end())
                throw std::runtime_error("exl3 mtp scan: " + ctx + " " + t + " missing in shard header");
            return it->second.second;
        };
        if (byte_count(tr_sh, txn) != l.trellis_bytes)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " trellis file bytes mismatch");
        if (byte_count(su_sh, t_suh) != l.suh_bytes)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " suh file bytes mismatch");
        if (byte_count(sv_sh, t_svh) != l.svh_bytes)
            throw std::runtime_error("exl3 mtp scan: " + ctx + " svh file bytes mismatch");
        l.suh_off = su_sh.offsets.find(t_suh)->second.first;
        l.svh_off = sv_sh.offsets.find(t_svh)->second.first;
        l.tr_off = tr_sh.offsets.find(txn)->second.first;
        l.has_mcg = mg_in;
        l.has_mul1 = ml_in;
        if (mg_in) {
            const ShardTensors& mg_sh = shard_of(grp + ".mcg");
            const nlohmann::json& mg = lookup(mg_sh, grp + ".mcg");
            if (!is_dtype(mg, "I32", "torch.int32") || !mg.contains("shape") ||
                shape_len(mg["shape"]) != 1 || byte_count(mg_sh, grp + ".mcg") != 4)
                throw std::runtime_error("exl3 mtp scan: " + ctx + " mcg must be a 4-byte scalar");
            l.mcg_off = mg_sh.offsets.find(grp + ".mcg")->second.first;
        }
        if (ml_in) {
            const ShardTensors& ml_sh = shard_of(grp + ".mul1");
            const nlohmann::json& ml = lookup(ml_sh, grp + ".mul1");
            if (!is_dtype(ml, "I32", "torch.int32") || !ml.contains("shape") ||
                shape_len(ml["shape"]) != 1 || byte_count(ml_sh, grp + ".mul1") != 4)
                throw std::runtime_error("exl3 mtp scan: " + ctx + " mul1 must be a 4-byte scalar");
            l.mul1_off = ml_sh.offsets.find(grp + ".mul1")->second.first;
        }
        meta.groups.push_back(std::move(l));
        known.insert(grp);
        ++appended;
    }
    return appended;
}

// Fills every plan entry's staged byte vectors with packed shard reads
// (trellis + suh + svh + 4-byte codebook scalar per member, concatenated
// along n for fused entries). Host-only file I/O, no CUDA calls; throws
// std::invalid_argument on geometry/byte mismatch, std::runtime_error on
// I/O failure. Sanitization stays in the side-car builders (proven path).
void exl3_stage_fused_host(const std::string& dir, std::vector<Exl3LogicalEntry>& plan);

// Engine-construction driver for an HF EXL3 checkpoint directory: host
// name-map + fusion plan, then (Linux deployment path) the packed upload
// through exl3_engine_load_dir into the process-lifetime store. The stream
// is opaque (cudaStream_t, which is void*) so host-compiled callers such as
// model_instance.cpp need no CUDA headers. Staging terminal (never returns):
// the EXL3 program binding that turns the store into a serving Engine lands
// outside S3b, so every path ends in an accurate diagnostic, never in the
// removed name-map stop-line and never in a half-built Engine.
[[noreturn]] void exl3_engine_construct_from_dir(const std::string& dir, void* stream_or_null);

#ifdef __CUDACC__

struct Exl3HostParts {
    const std::uint16_t* trellis = nullptr; // [k/16, n/16, 16K] int16
    std::uint16_t* suh           = nullptr; // [k] fp16 bits (sanitized in place)
    std::uint16_t* svh           = nullptr; // [n] fp16 bits (sanitized in place)
    std::int64_t k               = 0;
    std::int64_t n               = 0;
    std::int64_t K               = 0; // bits, 1..8
    bool has_mcg                 = false;
    bool has_mul1                = false;
    std::int32_t mul1_multiplier = 0;
};

struct Exl3Loaded {
    Exl3Weight weight; // trellis/suh/svh point at the device buffers below
    void* trellis_dev = nullptr;
    void* suh_dev     = nullptr;
    void* svh_dev     = nullptr;
    std::int64_t suh_fixed = 0; // sanitizer fix counts (load-time census)
    std::int64_t svh_fixed = 0;
};

// Throws std::invalid_argument on geometry/pointer mismatch, std::runtime_error
// on CUDA failure. ok=false (not throw) when K is not gemv-plannable.
Exl3Loaded exl3_build_sidecar(const Exl3HostParts& parts, cudaStream_t stream);

// Fused one-launch side-car: staged entry (bytes filled by
// exl3_stage_fused_host) -> sanitize (in place) -> upload -> weight with
// groups>1 + group_n. ok=true with an empty plan (the dispatch routes
// groups>1 to the v3 multi row for any m and any K in 1..8; the legacy
// gemv template is never consulted for fused entries). Throws like the
// dense builder on geometry mismatch (group_n widths must sum to n, each a
// 128-multiple; staged byte counts must match the fused geometry).
Exl3Loaded exl3_build_fused_sidecar(Exl3LogicalEntry& entry, cudaStream_t stream);

void exl3_free_sidecar(Exl3Loaded& loaded) noexcept;

// ------ serve-startup engine load (S3, LINUX-ONLY deployment) ------
// Process-lifetime EXL3 state for serve: the side-car store owns every
// logical linear's device trellis/suh/svh (built once at Engine
// construction, freed at teardown), and the workspace owns the v3 row's
// Hadamard scratch plus split-k accumulator (reserved once, NOT per
// request). The loader reads packed shard ranges verbatim (trellis + suh +
// svh + 4-byte codebook scalar per member): NO dense unpack, NO NVFP4
// convert, no scratch disk. Per-(m,K) routing is NOT duplicated here: the
// side-cars feed detail::exl3_dispatch unchanged (groups>1 -> v3 fused
// multi row; m==1 K3/K4 -> legacy gemv; otherwise -> v3 fused single row;
// see ops/linear/exl3/exl3_op.cu).
//
// Deployment note: the code below is portable C++/CUDA, but serve startup
// runs on Linux, so the device upload entry is exercised Linux-only.
struct Exl3EngineStore {
    std::vector<Exl3Loaded> groups; // device-owned side-cars, layout order
    std::vector<std::string> names; // names[i] is groups[i]'s LOGICAL .ninfer
                                    // name (serve lookup key, e.g.
                                    // text/layers/3/attention/qkv)
    std::vector<std::vector<std::string>> members; // members[i] lists the HF
                                    // checkpoint groups fused into names[i]
                                    // (exactly one entry when dense)
};

struct Exl3EngineWorkspace {
    void* had       = nullptr; // [groups*m_max, k_max] fp16 Hadamard scratch
    void* acc       = nullptr; // [m_max, n_max] fp32 split-k partials (null when no group splits)
    std::size_t had_bytes = 0;
    std::size_t acc_bytes = 0;
    std::int32_t m_max    = 0;
};

// Loads every logical EXL3 linear of the checkpoint directory `dir` (same
// layout accepted by load_exl3_layout): name-map -> packed shard reads ->
// sanitize -> upload -> plan. Clears `out` first; throws
// std::invalid_argument on geometry mismatch, std::runtime_error on I/O or
// CUDA failure (out is empty then). `stream` may be 0 (default stream) or
// the Engine transfer stream.
void exl3_engine_load_dir(const std::string& dir, cudaStream_t stream, Exl3EngineStore& out);

// Reserves the process-lifetime v3 workspace for the worst group in `store`
// at batch `m_max`, sized by the same exl3_aln_ws_bytes the dispatch carves
// (acc counted only when split-k actually triggers for that shape).
// Reserves ONCE at load; throws on empty store, non-positive m_max, or CUDA
// failure (ws is empty then).
void exl3_engine_reserve_workspace(const Exl3EngineStore& store, std::int32_t m_max,
                                   Exl3EngineWorkspace& ws);

void exl3_engine_free_workspace(Exl3EngineWorkspace& ws) noexcept;
void exl3_engine_free_store(Exl3EngineStore& store) noexcept;

// QType-path lookup: side-car for one logical linear, or nullptr when the
// name is not in the store. Returned pointer borrows the store.
const Exl3Weight* exl3_engine_find(const Exl3EngineStore& store,
                                   const std::string& name) noexcept;

// Process-lifetime serve store + workspace (LINUX-ONLY deployment):
// populated once by exl3_engine_construct_from_dir; serve consumes entries
// via exl3_engine_find without re-entering the Engine flow.
Exl3EngineStore& exl3_process_store() noexcept;
Exl3EngineWorkspace& exl3_process_workspace() noexcept;

#endif // __CUDACC__

} // namespace exl3
} // namespace ninfer
