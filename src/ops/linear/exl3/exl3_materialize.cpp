// exl3_materialize.cpp
// See exl3_materialize.h for the on-disk contract. Host-only; reads only the
// JSON directories (quantization_config.json, .index.json, safetensors
// headers) — no payload bytes — and validates every field strictly:
// trellis.c == 16*K, suh size == k, svh size == n, k*16 and n*16 divide
// trellis dims evenly, every EXL3 sub-tensor present in the index.json
// weight map and the containing file's header, dtype f16/i16/i32, n_bytes
// matches the derived shape size. The output layout feeds the device-view
// builder and the ops/linear dispatch (cb via mcg/mul1 per the 1.5.1 rule).
#include "exl3_materialize.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace exl3 {

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("exl3 layout: " + msg);
}

void load_json_file(const std::string& path, json& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    f >> out;
    if (f.fail() || !out.is_object()) fail("bad JSON object in " + path);
}

void require_arr(const json& j, const char* what, std::size_t ndim, const std::string& ctx) {
    if (!j.is_array() || j.size() != ndim) fail(ctx + ": " + what + " wrong shape");
}

std::int64_t shape_len(const json& shape) {
    std::int64_t n = 1;
    for (const auto& dim : shape) { n *= dim.get<std::int64_t>(); }
    return n;
}

void require_dtype(const json& t, const char* want, const std::string& ctx) {
    if (!t.contains("dtype") || t["dtype"].get<std::string>() != want) {
        fail(ctx + ": unexpected dtype (want " + std::string(want) + ")");
    }
    if (!t.contains("n_bytes")) fail(ctx + ": missing n_bytes");
    if (t["n_bytes"].get<std::int64_t>() != shape_len(t["shape"]) * (std::string(want) == "torch.bfloat16" ? 2 : (std::string(want) == "torch.int16" ? 2 : (std::string(want) == "torch.float16" ? 2 : 4)))) {
        fail(ctx + ": n_bytes disagrees with shape/dtype");
    }
}

struct FileTensorMap {
    std::map<std::string, std::pair<std::int64_t, std::int64_t>> offsets; // name -> (off, bytes)
};

// Parses the small 8-byte + JSON header of a .safetensors file.
FileTensorMap load_safetensors_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    std::uint64_t hdr = 0;
    f.read(reinterpret_cast<char*>(&hdr), 8);
    if (!f || hdr == 0 || hdr > (1ull << 31)) fail(path + ": bad safetensors header size");
    std::string text(static_cast<std::size_t>(hdr), '\0');
    f.read(text.data(), static_cast<std::streamsize>(hdr));
    if (!f) fail(path + ": short safetensors header");
    json j;
    try { j = json::parse(text); } catch (...) { fail(path + ": bad safetensors header JSON"); }
    if (!j.is_object()) fail(path + ": header is not a JSON object");
    FileTensorMap m;
    // Accept both the wrapped layout ({"version":..., "tensors":{...}}) and the
    // flat layout where every tensor record sits at the JSON root.
    const json* tensors = nullptr;
    if (j.contains("tensors") && j["tensors"].is_object()) {
        tensors = &j["tensors"];
    } else {
        tensors = &j;
        (void) 0;
    }
    int n_tensor = 0;
    for (const auto& e : tensors->items()) {
        const json& rec = e.value();
        if (!rec.is_object() || !rec.contains("data_offsets") || !rec["data_offsets"].is_array() || rec["data_offsets"].size() != 2) continue;
        const std::int64_t off = rec["data_offsets"][0].get<std::int64_t>();
        const std::int64_t end = rec["data_offsets"][1].get<std::int64_t>();
        if (end < off) fail(path + ": bad data_offsets for " + e.key());
        m.offsets[e.key()] = {off, end - off};
        ++n_tensor;
    }
    if (n_tensor == 0) fail(path + ": no tensor records found in header");
    return m;
}

} // namespace

Exl3CheckpointMeta load_exl3_layout(const std::string& path) {
    const std::string dir = path + (path.empty() || path.back() == '/' ? "" : "/");
    Exl3CheckpointMeta meta;

    json qcfg;
    load_json_file(dir + "quantization_config.json", qcfg);
    meta.bits_target = qcfg.value<int>("head_bits", 0);
    meta.bpw = qcfg.value<double>("bpw", 0.0);
    const auto& tmap = qcfg.at("tensor_storage"); // map group -> {stored_tensors, quant_format, ...}
    if (!tmap.is_object()) fail("tensor_storage is not an object");

    json index;
    load_json_file(dir + "model.safetensors.index.json", index);
    const auto& weight_map = index.at("weight_map");
    if (!weight_map.is_object()) fail("weight_map missing");

    // Open + parse each referenced safetensors header once.
    std::map<std::string, FileTensorMap> file_maps;
        auto file_for = [&](const std::string& tensor_name) -> const FileTensorMap& {
            auto wit = weight_map.find(tensor_name);
            if (wit == weight_map.end()) { fail("tensor not in weight_map: " + tensor_name); }
            const std::string fname = wit.value().get<std::string>();
            auto found = file_maps.find(fname);
            if (found == file_maps.end()) {
                found = file_maps.emplace(fname, load_safetensors_header(dir + fname)).first;
            }
            return found->second;
        };

    std::size_t exl3_count = 0;
    for (const auto& e : tmap.items()) {
        const std::string key = e.key();
        const std::string ctx = "group " + key;
        const json& v = e.value();
        if (!v.contains("quant_format")) continue; // non-EXL3 (bf16 norms / embeds)
        if (v["quant_format"].get<std::string>() != "exl3") continue;
        if (!v.contains("bits_per_weight") || !v["bits_per_weight"].is_number_integer()) {
            fail(ctx + ": missing bits_per_weight");
        }
        const auto& sts = v.at("stored_tensors");
        if (!sts.is_object()) fail(ctx + ": stored_tensors not an object");

        Exl3GroupLayout l;
        l.name = key;
        l.bits = v["bits_per_weight"].get<std::int64_t>();
        l.mul1_multiplier = v.value("mul1_multiplier", 0);

        json tr_t{}, su_t{}, sv_t{}, mg_t{}, ml_t{};
        bool has_tr = false, has_su = false, has_sv = false;
        for (const auto& se : sts.items()) {
            const std::string k = se.key();
            if (k == l.name + ".trellis") { tr_t = se.value(); has_tr = true; }
            else if (k == l.name + ".suh") { su_t = se.value(); has_su = true; }
            else if (k == l.name + ".svh") { sv_t = se.value(); has_sv = true; }
            else if (k == l.name + ".mcg") mg_t = se.value();
            else if (k == l.name + ".mul1") ml_t = se.value();
        }
        if (!has_tr || !has_su || !has_sv) fail(ctx + ": missing trellis/suh/svh sub-tensor");
        const json* tr = &tr_t; const json* su = &su_t; const json* sv = &sv_t;
        const bool mg_present = !mg_t.is_null();
        const bool ml_present = !ml_t.is_null();
        const json* mg = mg_present ? &mg_t : nullptr;
        const json* ml = ml_present ? &ml_t : nullptr;

        require_arr((*tr)["shape"], "trellis", 3, ctx);
        require_dtype(*tr, "torch.int16", ctx + "+trellis");
        std::int64_t d0 = (*tr)["shape"][0].get<std::int64_t>();
        std::int64_t d1 = (*tr)["shape"][1].get<std::int64_t>();
        std::int64_t d2 = (*tr)["shape"][2].get<std::int64_t>();
        if (d2 != 16 * l.bits) fail(ctx + ": trellis dim2 should be 16*K");
        if (d0 < 1 || d1 < 1) fail(ctx + ": degenerate trellis dims");
        l.k = d0 * 16;
        l.n = d1 * 16;
        l.trellis_u16 = d2;
        l.trellis_bytes = d0 * d1 * d2 * 2;

        require_dtype(*su, "torch.float16", ctx + "+suh");
        if (shape_len((*su)["shape"]) != l.k) fail(ctx + ": suh size != k");
        l.suh_bytes = l.k * 2;

        require_dtype(*sv, "torch.float16", ctx + "+svh");
        if (shape_len((*sv)["shape"]) != l.n) fail(ctx + ": svh size != n");
        l.svh_bytes = l.n * 2;

        l.has_mcg = (mg != nullptr);
        l.has_mul1 = (ml != nullptr);
        if (l.has_mcg && l.has_mul1) fail(ctx + ": both mcg and mul1 present");
        if (l.has_mcg) {
            require_dtype(*mg, "torch.int32", ctx + "+mcg");
            if (shape_len((*mg)["shape"]) != 1) fail(ctx + ": mcg not a scalar");
        }
        if (l.has_mul1) {
            require_dtype(*ml, "torch.int32", ctx + "+mul1");
            if (shape_len((*ml)["shape"]) != 1) fail(ctx + ": mul1 not a scalar");
        }

        // Resolve off-table per sub-tensor.
        struct R { std::int64_t off, bytes; };
        auto fetch = [&](const std::string& txn) -> R {
            const FileTensorMap& fm = file_for(txn);
            auto o = fm.offsets.find(txn);
            if (o == fm.offsets.end()) fail(ctx + ": " + txn + " missing in " + std::string(weight_map[txn].get<std::string>()));
            return {o->second.first, o->second.second};
        };
        const R rs = fetch(l.name + ".suh");
        if (rs.bytes != l.suh_bytes) fail(ctx + ": suh file bytes != suh_bytes");
        const R rv = fetch(l.name + ".svh");
        if (rv.bytes != l.svh_bytes) fail(ctx + ": svh file bytes != svh_bytes");
        const R rt = fetch(l.name + ".trellis");
        if (rt.bytes != l.trellis_bytes) fail(ctx + ": trellis file bytes != trellis_bytes");
        l.suh_off = rs.off; l.svh_off = rv.off; l.tr_off = rt.off;
        if (l.has_mcg) {
            const R rm = fetch(l.name + ".mcg");
            if (rm.bytes != 4) fail(ctx + ": mcg must be 4 bytes");
            l.mcg_off = rm.off;
        }
        if (l.has_mul1) {
            const R rx = fetch(l.name + ".mul1");
            if (rx.bytes != 4) fail(ctx + ": mul1 must be 4 bytes");
            l.mul1_off = rx.off;
        }
        meta.groups.push_back(std::move(l));
        ++exl3_count;
    }
    for (const auto& e : weight_map.items()) { meta.files.push_back(e.value().get<std::string>()); }
    std::sort(meta.files.begin(), meta.files.end());
    meta.files.erase(std::unique(meta.files.begin(), meta.files.end()), meta.files.end());
    meta.file_count = meta.files.size();
    if (exl3_count == 0) fail("no EXL3 groups found");
    return meta;
}

std::int64_t exl3_count_nondefinite(const std::uint16_t* v, std::int64_t count) {
    std::int64_t bad = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        const std::uint16_t b = v[i];
        const int sign = static_cast<int>((b >> 15) & 1u);
        const int stiff = static_cast<int>((b >> 10) & 0x1Fu);
        const int mant = static_cast<int>(b & 0x3FFu);
        const bool nan = (stiff == 0x1F) && (mant != 0);
        const bool inf = (stiff == 0x1F) && (mant == 0);
        bad += (nan || inf) ? 1 : 0;
    }
    return bad;
}

std::int64_t exl3_sanize_range(std::uint16_t* v, std::int64_t count) {
    std::int64_t bad = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        const std::uint16_t b = v[i];
        const int stiff = static_cast<int>((b >> 10) & 0x1Fu);
        const int mant = static_cast<int>(b & 0x3FFu);
        if ((stiff == 0x1F) && (mant != 0)) { v[i] = 0x0000u; ++bad; }        // NaN -> +0
        else if (stiff == 0x1F) { v[i] = 0x0000u; ++bad; }                    // Inf -> +0 (not saturate: keeps Hadamard block well-defined, no magnitude blow-up)
    }
    return bad;
}

Exl3SanitizeSummary exl3_checkpoint_summary(const std::string& dirpath) {
    const std::string dir = dirpath + (dirpath.empty() || dirpath.back() == '/' ? "" : "/");
    json index;
    load_json_file(dir + "model.safetensors.index.json", index);
    const auto& weight_map = index.at("weight_map");
    if (!weight_map.is_object()) fail("weight_map missing");
    // Re-run the full layout parse for group geometry (offsets + byte counts).
    const Exl3CheckpointMeta meta = load_exl3_layout(dir);

    std::map<std::string, FileTensorMap> file_maps;
    auto file_for = [&](const std::string& tensor_name) -> const FileTensorMap& {
        auto wit = weight_map.find(tensor_name);
        if (wit == weight_map.end()) { fail("tensor not in weight_map: " + tensor_name); }
        const std::string fname = wit.value().get<std::string>();
        auto found = file_maps.find(fname);
        if (found == file_maps.end()) {
            found = file_maps.emplace(fname, load_safetensors_header(dir + fname)).first;
        }
        return found->second;
    };
    auto read_range = [&](const std::string& tensor_name, std::int64_t off, std::int64_t bytes, std::vector<char>& out) {
        const FileTensorMap& fm = file_for(tensor_name);
        const auto it = fm.offsets.find(tensor_name);
        if (it == fm.offsets.end()) fail("no offsets for " + tensor_name);
        const std::int64_t fname_off = it->second.first;
        if (off != fname_off) fail("materializer/ckpt offset drift for " + tensor_name);
        const std::string fn = weight_map.at(tensor_name).get<std::string>();
        std::ifstream f(dir + fn, std::ios::binary);
        if (!f) fail("cannot open " + fn);
        f.seekg(fname_off);
        out.resize(static_cast<std::size_t>(bytes));
        f.read(out.data(), static_cast<std::streamsize>(bytes));
        if (!f || f.gcount() != static_cast<std::streamsize>(bytes)) fail("short read for " + tensor_name);
    };

    Exl3SanitizeSummary s;
    s.groups_total = static_cast<std::int64_t>(meta.groups.size());
    for (const auto& g : meta.groups) {
        std::vector<char> buf;
        if (g.suh_bytes > 0) {
            read_range(g.name + ".suh", g.suh_off, g.suh_bytes, buf);
            const std::int64_t bad = exl3_count_nondefinite(reinterpret_cast<const std::uint16_t*>(buf.data()), g.suh_bytes / 2);
            s.entries_bad_suh += bad;
            if (bad) ++s.groups_bad_suh;
        }
        if (g.svh_bytes > 0) {
            read_range(g.name + ".svh", g.svh_off, g.svh_bytes, buf);
            const std::int64_t bad = exl3_count_nondefinite(reinterpret_cast<const std::uint16_t*>(buf.data()), g.svh_bytes / 2);
            s.entries_bad_svh += bad;
            if (bad) ++s.groups_bad_svh;
        }
    }
    return s;
}

std::vector<const Exl3GroupLayout*> exl3_groups_by_size(const Exl3CheckpointMeta& meta) {
    std::vector<const Exl3GroupLayout*> out;
    for (const auto& g : meta.groups) out.push_back(&g);
    std::sort(out.begin(), out.end(),
              [](const Exl3GroupLayout* a, const Exl3GroupLayout* b) {
                  const auto as = a->suh_bytes + a->svh_bytes + a->trellis_bytes;
                  const auto bs = b->suh_bytes + b->svh_bytes + b->trellis_bytes;
                  return (as < bs) || (as == bs && a->name < b->name);
              });
    return out;
}

bool exl3_is_exl3_checkpoint_dir(const std::string& path)
{
    try {
        const std::string dir = path + (path.empty() || path.back() == '/' ? "" : "/");
        json qcfg;
        load_json_file(dir + "quantization_config.json", qcfg);
        json index;
        load_json_file(dir + "model.safetensors.index.json", index);
        if (!qcfg.contains("tensor_storage") || !qcfg["tensor_storage"].is_object())
            return false;
        if (!index.contains("weight_map") || !index["weight_map"].is_object()) return false;
        for (const auto& e : qcfg["tensor_storage"].items()) {
            const json& v = e.value();
            if (v.is_object() && v.contains("quant_format") && v["quant_format"].is_string() &&
                v["quant_format"].get<std::string>() == "exl3")
                return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

std::vector<Exl3PackedUpload> exl3_packed_uploads(const std::string& dirpath)
{
    const Exl3CheckpointMeta meta = load_exl3_layout(dirpath);
    const std::string dir = dirpath + (dirpath.empty() || dirpath.back() == '/' ? "" : "/");
    json index;
    load_json_file(dir + "model.safetensors.index.json", index);
    const auto& weight_map = index.at("weight_map");
    auto fname = [&](const std::string& txn, const std::string& ctx) -> std::string {
        const auto it = weight_map.find(txn);
        if (it == weight_map.end() || !it.value().is_string())
            fail(ctx + ": " + txn + " missing in weight_map");
        return it.value().get<std::string>();
    };
    std::vector<Exl3PackedUpload> out;
    out.reserve(meta.groups.size());
    for (const auto& g : meta.groups) {
        Exl3PackedUpload u;
        u.name             = g.name;
        u.k                = g.k;
        u.n                = g.n;
        u.bits             = g.bits;
        u.has_mcg          = g.has_mcg;
        u.has_mul1         = g.has_mul1;
        u.mul1_multiplier  = g.mul1_multiplier;
        u.cb               = g.has_mul1 ? 2 : (g.has_mcg ? 1 : 0);
        u.suh_file         = fname(g.name + ".suh", u.name);
        u.suh_off          = g.suh_off;
        u.suh_bytes        = g.suh_bytes;
        u.svh_file         = fname(g.name + ".svh", u.name);
        u.svh_off          = g.svh_off;
        u.svh_bytes        = g.svh_bytes;
        u.tr_file          = fname(g.name + ".trellis", u.name);
        u.tr_off           = g.tr_off;
        u.trellis_bytes    = g.trellis_bytes;
        if (g.has_mcg) {
            u.cb_file = fname(g.name + ".mcg", u.name);
            u.cb_off  = g.mcg_off;
        } else if (g.has_mul1) {
            u.cb_file = fname(g.name + ".mul1", u.name);
            u.cb_off  = g.mul1_off;
        }
        out.push_back(std::move(u));
    }
    return out;
}

} // namespace exl3
} // namespace ninfer