// exl3_bind.cu
// EXL3 side-car builder (see exl3_bind.h). Dense path: sanitize (host) ->
// upload (async) -> plan. Fused path: stage (packed member reads +
// concat along n, host) -> sanitize -> upload once -> weight with
// groups>1 + group_n for the v3 one-launch multi row. Compiled with nvcc
// (full header view); the host-only name-map section of the header stays
// cl-safe for host TUs and the standalone unit test.
#include "ops/linear/exl3/exl3_bind.h"
#include "ops/linear/exl3/exl3_materialize.h"
#include "ops/linear/exl3/exl3_aln.h"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ninfer {
namespace exl3 {
namespace {

void check_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("exl3 side-car: ") + what);
}

} // namespace

Exl3Loaded exl3_build_sidecar(const Exl3HostParts& parts, cudaStream_t stream)
{
    if (!parts.trellis || !parts.suh || !parts.svh)
        throw std::invalid_argument("exl3 side-car: null parts");
    if (parts.k % 128 != 0 || parts.n % 128 != 0 || parts.k <= 0 || parts.n <= 0)
        throw std::invalid_argument("exl3 side-car: k/n must be +128-multiples");
    if (parts.K < 1 || parts.K > 8)
        throw std::invalid_argument("exl3 side-car: K must be 1..8");
    if (parts.has_mcg && parts.has_mul1)
        throw std::invalid_argument("exl3 side-car: mcg+mul1 exclusive");

    Exl3Loaded out{};
    out.suh_fixed = exl3_sanize_range(parts.suh, parts.k);
    out.svh_fixed = exl3_sanize_range(parts.svh, parts.n);

    const std::size_t ntr = (std::size_t)(parts.k / 16) * (std::size_t)(parts.n / 16) *
                            (std::size_t)(16 * parts.K);
    check_cuda(cudaMalloc(&out.trellis_dev, ntr * 2), "trellis alloc");
    check_cuda(cudaMalloc(&out.suh_dev, (std::size_t) parts.k * 2), "suh alloc");
    check_cuda(cudaMalloc(&out.svh_dev, (std::size_t) parts.n * 2), "svh alloc");
    check_cuda(cudaMemcpyAsync(out.trellis_dev, parts.trellis, ntr * 2,
                               cudaMemcpyHostToDevice, stream), "trellis upload");
    check_cuda(cudaMemcpyAsync(out.suh_dev, parts.suh, (std::size_t) parts.k * 2,
                               cudaMemcpyHostToDevice, stream), "suh upload");
    check_cuda(cudaMemcpyAsync(out.svh_dev, parts.svh, (std::size_t) parts.n * 2,
                               cudaMemcpyHostToDevice, stream), "svh upload");

    Exl3Weight& w = out.weight;
    w.trellis = static_cast<const std::uint16_t*>(out.trellis_dev);
    w.suh = static_cast<const half*>(out.suh_dev);
    w.svh = static_cast<const half*>(out.svh_dev);
    w.k = (std::int32_t) parts.k;
    w.n = (std::int32_t) parts.n;
    w.bits = (std::int32_t) parts.K;
    w.has_mcg = parts.has_mcg;
    w.has_mul1 = parts.has_mul1;
    w.mul1_multiplier = parts.mul1_multiplier;
    w.groups = 1; // dense single (fused entries use exl3_build_fused_sidecar)
    // K3/K4 earn a gemv plan (single-token decode route); K5/K6 are
    // v3-servable (gate-proven) with no gemv plan -- the dispatch bits
    // gate keeps them off the legacy route, so ok=true is honest.
    w.ok = (w.bits <= 4) ? exl3_plan_for(w, 1) : true;
    return out;
}

Exl3Loaded exl3_build_fused_sidecar(Exl3LogicalEntry& entry, cudaStream_t stream)
{
    if (entry.trellis.empty() || entry.suh.empty() || entry.svh.empty())
        throw std::invalid_argument("exl3 fused side-car: entry not staged for " + entry.logical);
    if (entry.groups < 2 || entry.groups > 8)
        throw std::invalid_argument("exl3 fused side-car: groups must be 2..8 for " +
                                    entry.logical);
    if (entry.k % 128 != 0 || entry.n % 128 != 0 || entry.k <= 0 || entry.n <= 0)
        throw std::invalid_argument("exl3 fused side-car: k/n must be +128-multiples for " +
                                    entry.logical);
    if (entry.bits < 1 || entry.bits > 8)
        throw std::invalid_argument("exl3 fused side-car: K must be 1..8 for " + entry.logical);
    if (entry.has_mcg && entry.has_mul1)
        throw std::invalid_argument("exl3 fused side-car: mcg+mul1 exclusive for " +
                                    entry.logical);
    std::int64_t n_sum = 0;
    for (std::int32_t g = 0; g < entry.groups; ++g) {
        if (entry.group_n[g] <= 0 || entry.group_n[g] % 128 != 0)
            throw std::invalid_argument("exl3 fused side-car: group_n must be +128-multiples for " +
                                        entry.logical);
        n_sum += entry.group_n[g];
    }
    if (n_sum != entry.n)
        throw std::invalid_argument("exl3 fused side-car: group_n must sum to n for " +
                                    entry.logical);
    const std::size_t ntr = (std::size_t)(entry.k / 16) * (std::size_t)(entry.n / 16) *
                            (std::size_t)(16 * entry.bits);
    if (entry.trellis.size() != ntr)
        throw std::invalid_argument("exl3 fused side-car: trellis bytes mismatch for " +
                                    entry.logical);
    if (entry.suh.size() != (std::size_t) entry.groups * (std::size_t) entry.k ||
        entry.svh.size() != (std::size_t) entry.n)
        throw std::invalid_argument("exl3 fused side-car: suh/svh bytes mismatch for " +
                                    entry.logical);

    Exl3Loaded out{};
    out.suh_fixed = exl3_sanize_range(entry.suh.data(),
                                      (std::int64_t) entry.groups * entry.k);
    out.svh_fixed = exl3_sanize_range(entry.svh.data(), entry.n);

    check_cuda(cudaMalloc(&out.trellis_dev, ntr * 2), "fused trellis alloc");
    check_cuda(cudaMalloc(&out.suh_dev, entry.suh.size() * 2), "fused suh alloc");
    check_cuda(cudaMalloc(&out.svh_dev, entry.svh.size() * 2), "fused svh alloc");
    check_cuda(cudaMemcpyAsync(out.trellis_dev, entry.trellis.data(), ntr * 2,
                               cudaMemcpyHostToDevice, stream), "fused trellis upload");
    check_cuda(cudaMemcpyAsync(out.suh_dev, entry.suh.data(), entry.suh.size() * 2,
                               cudaMemcpyHostToDevice, stream), "fused suh upload");
    check_cuda(cudaMemcpyAsync(out.svh_dev, entry.svh.data(), entry.svh.size() * 2,
                               cudaMemcpyHostToDevice, stream), "fused svh upload");

    Exl3Weight& w = out.weight;
    w.trellis = static_cast<const std::uint16_t*>(out.trellis_dev);
    w.suh = static_cast<const half*>(out.suh_dev);
    w.svh = static_cast<const half*>(out.svh_dev);
    w.k = (std::int32_t) entry.k;
    w.n = (std::int32_t) entry.n;
    w.bits = (std::int32_t) entry.bits;
    w.has_mcg = entry.has_mcg;
    w.has_mul1 = entry.has_mul1;
    w.mul1_multiplier = entry.mul1_multiplier;
    w.groups = entry.groups;
    for (std::int32_t g = 0; g < entry.groups; ++g) w.group_n[g] = entry.group_n[g];
    // Fused entries serve the v3 multi row for any m (the dispatch never
    // consults the gemv plan when groups>1), so ok=true with an empty plan.
    w.ok = true;
    return out;
}

void exl3_free_sidecar(Exl3Loaded& loaded) noexcept
{
    if (loaded.trellis_dev) cudaFree(loaded.trellis_dev);
    if (loaded.suh_dev) cudaFree(loaded.suh_dev);
    if (loaded.svh_dev) cudaFree(loaded.svh_dev);
    loaded = Exl3Loaded{};
}

// ------ serve-startup engine load (see exl3_bind.h) ------
namespace {

// LINUX-ONLY deployment path (serve startup); portable C++/CUDA below.
std::uint64_t exl3_shard_hdr_len(const std::string& file)
{
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("exl3 engine load: cannot open " + file);
    std::uint64_t hdr = 0;
    f.read(reinterpret_cast<char*>(&hdr), 8);
    if (!f || hdr == 0 || hdr > (1ull << 31))
        throw std::runtime_error("exl3 engine load: bad safetensors header size in " + file);
    return hdr;
}

void exl3_read_range(const std::string& file, std::uint64_t hdr, std::int64_t off,
                     std::int64_t bytes, void* dst)
{
    if (off < 0 || bytes <= 0)
        throw std::invalid_argument("exl3 engine load: bad range in " + file);
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("exl3 engine load: cannot open " + file);
    f.seekg((std::streamoff)(8 + hdr + (std::uint64_t) off));
    if (!f) throw std::runtime_error("exl3 engine load: cannot seek in " + file);
    f.read(static_cast<char*>(dst), (std::streamsize) bytes);
    if (!f || f.gcount() != (std::streamsize) bytes)
        throw std::runtime_error("exl3 engine load: short read in " + file);
}

} // namespace

void exl3_stage_fused_host(const std::string& dir, std::vector<Exl3LogicalEntry>& plan)
{
    const std::string root = dir + (dir.empty() || dir.back() == '/' ? "" : "/");
    // Uploads over the augmented layout: exl3_packed_uploads re-derives the
    // tensor_storage-only layout (401 groups) and would miss staged mtp.*
    // members, so the plan is rebuilt here from load + mtp append.
    Exl3CheckpointMeta stage_meta = load_exl3_layout(root);
    exl3_append_mtp_groups(root, stage_meta);
    nlohmann::json index;
    {
        std::ifstream fi(root + "model.safetensors.index.json", std::ios::binary);
        if (!fi) throw std::runtime_error("exl3 stage: cannot open weight_map");
        fi >> index;
    }
    const auto& weight_map = index.at("weight_map");
    auto fname = [&](const std::string& txn, const std::string& ctx) -> std::string {
        const auto it = weight_map.find(txn);
        if (it == weight_map.end() || !it.value().is_string())
            throw std::runtime_error("exl3 stage: " + ctx + ": " + txn + " missing in weight_map");
        return it.value().get<std::string>();
    };
    std::vector<Exl3PackedUpload> uploads;
    uploads.reserve(stage_meta.groups.size());
    for (const auto& g : stage_meta.groups) {
        Exl3PackedUpload u;
        u.name            = g.name;
        u.k               = g.k;
        u.n               = g.n;
        u.bits            = g.bits;
        u.has_mcg         = g.has_mcg;
        u.has_mul1        = g.has_mul1;
        u.mul1_multiplier = g.mul1_multiplier;
        u.cb              = g.has_mul1 ? 2 : (g.has_mcg ? 1 : 0);
        u.suh_file        = fname(g.name + ".suh", u.name);
        u.suh_off         = g.suh_off;
        u.suh_bytes       = g.suh_bytes;
        u.svh_file        = fname(g.name + ".svh", u.name);
        u.svh_off         = g.svh_off;
        u.svh_bytes       = g.svh_bytes;
        u.tr_file         = fname(g.name + ".trellis", u.name);
        u.tr_off          = g.tr_off;
        u.trellis_bytes   = g.trellis_bytes;
        if (g.has_mcg) {
            u.cb_file = fname(g.name + ".mcg", u.name);
            u.cb_off  = g.mcg_off;
        } else if (g.has_mul1) {
            u.cb_file = fname(g.name + ".mul1", u.name);
            u.cb_off  = g.mul1_off;
        }
        uploads.push_back(std::move(u));
    }
    std::map<std::string, const Exl3PackedUpload*> by_name;
    for (const auto& u : uploads) by_name[u.name] = &u;
    std::map<std::string, std::uint64_t> hdrs;
    auto hdr_of = [&](const std::string& shard) -> std::uint64_t {
        const auto it = hdrs.find(shard);
        if (it != hdrs.end()) return it->second;
        const std::uint64_t h = exl3_shard_hdr_len(root + shard);
        hdrs.emplace(shard, h);
        return h;
    };
    for (auto& e : plan) {
        if (e.members.empty())
            throw std::invalid_argument("exl3 stage: logical entry with no members: " + e.logical);
        if ((std::size_t) e.groups != e.members.size())
            throw std::invalid_argument("exl3 stage: groups/member count mismatch for " + e.logical);
        if (e.k <= 0 || e.n <= 0 || e.k % 128 != 0 || e.n % 128 != 0)
            throw std::invalid_argument("exl3 stage: bad k/n for " + e.logical);
        if (e.bits < 1 || e.bits > 8)
            throw std::invalid_argument("exl3 stage: bad K for " + e.logical);
        const std::int64_t d0 = e.k / 16;
        const std::int64_t d2 = 16 * e.bits;
        std::int64_t d1 = 0;
        for (const auto& m : e.members) {
            const auto it = by_name.find(m);
            if (it == by_name.end())
                throw std::runtime_error("exl3 stage: member not in upload plan: " + m);
            const Exl3PackedUpload* u = it->second;
            // The planner guarantees uniformity inside a fusion set; re-check
            // against the upload plan (the files are the authority).
            if (u->k != e.k || u->bits != e.bits || u->has_mcg != e.has_mcg ||
                u->has_mul1 != e.has_mul1)
                throw std::invalid_argument("exl3 stage: member geometry drift for " + m);
            if (u->n % 128 != 0 || u->n <= 0)
                throw std::invalid_argument("exl3 stage: bad member n for " + m);
            const std::size_t ntr = (std::size_t)(u->k / 16) * (std::size_t)(u->n / 16) *
                                    (std::size_t)(16 * u->bits);
            if ((std::int64_t)(ntr * 2) != u->trellis_bytes)
                throw std::invalid_argument("exl3 stage: trellis bytes mismatch for " + m);
            if (u->suh_bytes != u->k * 2 || u->svh_bytes != u->n * 2)
                throw std::invalid_argument("exl3 stage: suh/svh bytes mismatch for " + m);
            d1 += u->n / 16;
        }
        if (d1 != e.n / 16)
            throw std::invalid_argument("exl3 stage: member widths do not sum to n for " +
                                        e.logical);
        // Packed reads straight out of the shards: trellis + suh + svh +
        // the 4-byte codebook scalar per member. No dense unpack, no convert.
        std::vector<std::uint16_t> tr((std::size_t) d0 * (std::size_t) d1 * (std::size_t) d2);
        std::vector<std::uint16_t> suh((std::size_t) e.groups * (std::size_t) e.k);
        std::vector<std::uint16_t> svh((std::size_t) e.n);
        std::int64_t col = 0; // fused dim1 block cursor (n/16 units)
        std::int32_t mul = e.mul1_multiplier;
        for (std::size_t g = 0; g < e.members.size(); ++g) {
            const Exl3PackedUpload* u = by_name[e.members[g]];
            const std::int64_t md1 = u->n / 16;
            std::vector<std::uint16_t> mtr((std::size_t)(u->k / 16) * (std::size_t) md1 *
                                           (std::size_t)(16 * u->bits));
            exl3_read_range(root + u->suh_file, hdr_of(u->suh_file), u->suh_off, u->suh_bytes,
                            suh.data() + g * (std::size_t) e.k);
            exl3_read_range(root + u->svh_file, hdr_of(u->svh_file), u->svh_off, u->svh_bytes,
                            svh.data() + (std::size_t) col * 16);
            exl3_read_range(root + u->tr_file, hdr_of(u->tr_file), u->tr_off, u->trellis_bytes,
                            mtr.data());
            if (u->cb != 0) {
                std::int32_t scalar = 0;
                exl3_read_range(root + u->cb_file, hdr_of(u->cb_file), u->cb_off, 4, &scalar);
                if (u->cb == 2) {
                    if (mul != 0 && mul != scalar)
                        throw std::runtime_error("exl3 stage: mul1 scalar mismatch for " +
                                                 e.members[g]);
                    mul = scalar;
                }
            }
            // Trellis is [d0, md1, d2] C-order: splice each k-block row's
            // md1*d2 words at the fused dim1 cursor.
            for (std::int64_t i = 0; i < d0; ++i) {
                std::uint16_t* dst = tr.data() + ((std::size_t) i * (std::size_t) d1 +
                                                  (std::size_t) col) * (std::size_t) d2;
                const std::uint16_t* src = mtr.data() + ((std::size_t) i * (std::size_t) md1) *
                                                              (std::size_t) d2;
                for (std::int64_t w = 0; w < md1 * d2; ++w) dst[w] = src[w];
            }
            col += md1;
        }
        e.mul1_multiplier = mul;
        e.trellis = std::move(tr);
        e.suh     = std::move(suh);
        e.svh     = std::move(svh);
    }
}

void exl3_engine_load_dir(const std::string& dir, cudaStream_t stream, Exl3EngineStore& out)
{
    const std::string root = dir + (dir.empty() || dir.back() == '/' ? "" : "/");
    Exl3CheckpointMeta load_meta = load_exl3_layout(root);
    exl3_append_mtp_groups(root, load_meta);
    auto plan = exl3_plan_fusion(load_meta);
    if (plan.empty()) throw std::runtime_error("exl3 engine load: no EXL3 groups in " + dir);
    exl3_stage_fused_host(root, plan);
    Exl3EngineStore fresh;
    fresh.groups.reserve(plan.size());
    fresh.names.reserve(plan.size());
    fresh.members.reserve(plan.size());
    try {
        for (auto& e : plan) {
            Exl3Loaded loaded{};
            try {
                if (e.groups > 1) {
                    loaded = exl3_build_fused_sidecar(e, stream);
                } else {
                    // Dense singles ride the proven 1:1 builder verbatim.
                    Exl3HostParts parts;
                    parts.trellis         = e.trellis.data();
                    parts.suh             = e.suh.data();
                    parts.svh             = e.svh.data();
                    parts.k               = e.k;
                    parts.n               = e.n;
                    parts.K               = e.bits;
                    parts.has_mcg         = e.has_mcg;
                    parts.has_mul1        = e.has_mul1;
                    parts.mul1_multiplier = e.mul1_multiplier;
                    loaded = exl3_build_sidecar(parts, stream);
                }
            } catch (...) {
                exl3_free_sidecar(loaded);
                throw;
            }
            // Fence the async upload before the host buffers die (load-time
            // only; serve never pays this per request).
            check_cuda(cudaStreamSynchronize(stream), "side-car upload fence");
            fresh.groups.push_back(loaded);
            fresh.names.push_back(e.logical);
            fresh.members.push_back(e.members);
        }
    } catch (...) {
        exl3_engine_free_store(fresh);
        throw;
    }
    exl3_engine_free_store(out);
    out = std::move(fresh);
}

void exl3_engine_reserve_workspace(const Exl3EngineStore& store, std::int32_t m_max,
                                   Exl3EngineWorkspace& ws)
{
    if (m_max <= 0)
        throw std::invalid_argument("exl3 engine workspace: m_max must be positive");
    if (store.groups.empty())
        throw std::invalid_argument("exl3 engine workspace: empty store");
    exl3_engine_free_workspace(ws);
    std::size_t had = 0, acc = 0;
    for (const auto& g : store.groups) {
        const Exl3Weight& w = g.weight;
        Exl3AlnParams p{};
        p.k    = w.k;
        p.n    = w.n;
        p.bits = w.bits;
        p.cb   = (w.has_mul1 ? 2 : (w.has_mcg ? 1 : 0));
        bool need_acc = false;
        (void) exl3_aln_ws_bytes(p, w.groups, m_max, &need_acc);
        const std::size_t h =
            (std::size_t) w.groups * (std::size_t) m_max * (std::size_t) w.k * 2;
        if (h > had) had = h;
        std::size_t a = 0;
        if (need_acc) {
            a = (std::size_t) m_max * (std::size_t) w.n * 4;
        }
        if (exl3_aln_det_enabled()) {
            // Deterministic planes: the split factor is a function of the
            // runtime batch, which can be smaller than m_max with a larger
            // split, so cover every servable batch (powers of two plus m_max
            // itself). planes(mm)*mm*n*4 is the exact plane footprint there;
            // the kernel's per-candidate splits never exceed the bm=128 value
            // this query returns, so the max covers tuning as well.
            for (std::int32_t mm = m_max; mm >= 1; mm /= 2) {
                int pl = exl3_aln_det_planes(p, mm);
                if (pl > 0) {
                    const std::size_t cand =
                        (std::size_t) pl * (std::size_t) mm * (std::size_t) w.n * 4;
                    if (cand > a) a = cand;
                }
                if (mm == 1) break;  // mm /= 2 would stick at 1 for signed ints
            }
        }
        if (a > acc) acc = a;
    }
    Exl3EngineWorkspace fresh;
    check_cuda(cudaMalloc(&fresh.had, had), "engine workspace had alloc");
    try {
        if (acc) check_cuda(cudaMalloc(&fresh.acc, acc), "engine workspace acc alloc");
    } catch (...) {
        exl3_engine_free_workspace(fresh);
        throw;
    }
    fresh.had_bytes = had;
    fresh.acc_bytes = acc;
    fresh.m_max     = m_max;
    ws = fresh;
}

void exl3_engine_free_workspace(Exl3EngineWorkspace& ws) noexcept
{
    if (ws.had) cudaFree(ws.had);
    if (ws.acc) cudaFree(ws.acc);
    ws = Exl3EngineWorkspace{};
}

void exl3_engine_free_store(Exl3EngineStore& store) noexcept
{
    for (auto& g : store.groups) exl3_free_sidecar(g);
    store = Exl3EngineStore{};
}

const Exl3Weight* exl3_engine_find(const Exl3EngineStore& store,
                                   const std::string& name) noexcept
{
    for (std::size_t i = 0; i < store.names.size() && i < store.groups.size(); ++i) {
        if (store.names[i] == name) return &store.groups[i].weight;
    }
    return nullptr;
}

Exl3EngineStore& exl3_process_store() noexcept
{
    static Exl3EngineStore store;
    return store;
}

Exl3EngineWorkspace& exl3_process_workspace() noexcept
{
    static Exl3EngineWorkspace ws;
    return ws;
}

void exl3_engine_construct_from_dir(const std::string& dir, void* stream_or_null)
{
    Exl3CheckpointMeta construct_meta = load_exl3_layout(dir);
    exl3_append_mtp_groups(dir, construct_meta);
    const auto plan = exl3_plan_fusion(construct_meta);
    std::size_t fused = 0;
    for (const auto& e : plan) fused += (e.groups > 1) ? 1 : 0;
    const std::size_t dense = plan.size() - fused;
    const std::string census = std::to_string(plan.size()) + " logical linears (" +
                               std::to_string(fused) + " fused multi-group, " +
                               std::to_string(dense) + " dense)";
#if defined(_WIN32)
    (void) stream_or_null;
    // LINUX-ONLY: the packed trellis/suh/svh device upload below runs at
    // serve startup on Linux. This host validated the name-map; no upload
    // is attempted here.
    throw std::invalid_argument("EXL3 checkpoint directory with " + census +
                                ": side-car device upload is LINUX-ONLY (serve startup); " +
                                "host name-map validated, no upload attempted on this host");
#else
    // LINUX-ONLY deployment path (serve startup): packed trellis/suh/svh +
    // codebook-scalar upload, one side-car per logical linear, into the
    // process-lifetime store that serve consumes via exl3_engine_find.
    // (Workspace reservation stays with serve startup, which owns the batch
    // bound; see exl3_engine_reserve_workspace.)
    cudaStream_t stream = static_cast<cudaStream_t>(stream_or_null);
    Exl3EngineStore store;
    exl3_engine_load_dir(dir, stream, store);
    Exl3EngineStore& live = exl3_process_store();
    exl3_engine_free_store(live);
    live = std::move(store);
    throw std::invalid_argument("EXL3 checkpoint directory with " + census +
                                ": packed side-car store loaded into the process store; " +
                                "EXL3 program binding lands outside S3b (no .ninfer program " +
                                "for a checkpoint directory)");
#endif
}

} // namespace exl3
} // namespace ninfer
