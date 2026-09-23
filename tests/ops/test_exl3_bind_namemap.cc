// tests/ops/test_exl3_bind_namemap.cc
// S3b proof: EXL3 binder name-map + fusion plan, host-only.
//
// What this pins (src/ops/linear/exl3/exl3_bind.h):
//   1. HF -> logical .ninfer names (self_attn.q/k/v/o, mlp.gate/up/down,
//      linear_attn in_proj_*, lm_head; fused qkv_proj / gate_up_proj
//      spellings; unknown spellings fall back verbatim).
//   2. Fusion: complete uniform q/k/v triples -> one entry groups=3 with
//      group_n in serve order (q,k,v); gate/up pairs -> groups=2 halves.
//   3. Dense singles stay groups=1 (incl. K5 o_proj, K6 lm_head).
//   4. Incomplete/non-uniform sets fall back to dense singles (never break
//      the load); every input group is consumed exactly once.
//
// Host-only: no CUDA headers, no device, no JSON, no checkpoint bytes. The
// meta here is synthetic; the real-checkpoint census (409 groups -> 310
// logical linears, 82 fused: 401 tensor_storage groups at 305/80 plus 8
// mtp.* groups contributing one qkv triple, one gate_up pair, and 3 dense)
// is verified by a scratch probe, not by this TU.

#include "ops/linear/exl3/exl3_bind.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            ++failures;                                                              \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                     \
        }                                                                            \
    } while (0)

ninfer::exl3::Exl3GroupLayout mk(const std::string& name, std::int64_t bits, std::int64_t k,
                                 std::int64_t n)
{
    ninfer::exl3::Exl3GroupLayout g;
    g.name = name;
    g.bits = bits;
    g.k = k;
    g.n = n;
    g.trellis_u16 = 16 * bits;
    g.suh_bytes = k * 2;
    g.svh_bytes = n * 2;
    g.trellis_bytes = (k / 16) * (n / 16) * 16 * bits * 2;
    g.has_mcg = false;
    g.has_mul1 = true;
    g.mul1_multiplier = 2212286765;
    return g;
}

const ninfer::exl3::Exl3LogicalEntry* find_by_logical(
    const std::vector<ninfer::exl3::Exl3LogicalEntry>& plan, const std::string& logical)
{
    for (const auto& e : plan) {
        if (e.logical == logical) return &e;
    }
    return nullptr;
}

} // namespace

int main()
{
    using ninfer::exl3::Exl3CheckpointMeta;

    // --- name map spot checks ---
    CHECK(ninfer::exl3::exl3_map_group_to_logical("lm_head") == "text/output_head");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.3.self_attn.q_proj") == "text/layers/3/attention/query");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.3.self_attn.o_proj") == "text/layers/3/attention/output");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.0.linear_attn.in_proj_qkv") == "text/layers/0/gdn/qkv");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.0.mlp.down_proj") == "text/layers/0/mlp/down");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.3.self_attn.qkv_proj") == "text/layers/3/attention/qkv");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "model.language_model.layers.0.mlp.gate_up_proj") == "text/layers/0/mlp/gate_up");
    CHECK(ninfer::exl3::exl3_map_group_to_logical("some.unknown.tensors") ==
          "some.unknown.tensors");

    // --- mtp.* name map (S3c): text/mtp mirror, never collides with text/layers ---
    CHECK(ninfer::exl3::exl3_map_group_to_logical("mtp.fc") == "text/mtp/draft_head");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.self_attn.q_proj") == "text/mtp/layers/0/attention/query");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.self_attn.k_proj") == "text/mtp/layers/0/attention/key");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.self_attn.v_proj") == "text/mtp/layers/0/attention/value");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.self_attn.o_proj") == "text/mtp/layers/0/attention/output");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.self_attn.qkv_proj") == "text/mtp/layers/0/attention/qkv");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.mlp.gate_proj") == "text/mtp/layers/0/mlp/gate");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.mlp.up_proj") == "text/mtp/layers/0/mlp/up");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.mlp.down_proj") == "text/mtp/layers/0/mlp/down");
    CHECK(ninfer::exl3::exl3_map_group_to_logical(
              "mtp.layers.0.mlp.gate_up_proj") == "text/mtp/layers/0/mlp/gate_up");
    // Collision guard: "mtp.layers." contains ".layers." and must not map
    // into the real text/layers/{i} namespace.
    CHECK(ninfer::exl3::exl3_map_group_to_logical("mtp.layers.0.self_attn.q_proj") !=
          "text/layers/0/attention/query");
    CHECK(ninfer::exl3::exl3_map_group_to_logical("mtp.layers.0.mlp.gate_proj") !=
          "text/layers/0/mlp/gate");
    // Unknown mtp spellings (and non-numeric layer indices) fall back verbatim.
    CHECK(ninfer::exl3::exl3_map_group_to_logical("mtp.layers.0.self_attn.q_norm") ==
          "mtp.layers.0.self_attn.q_norm");
    CHECK(ninfer::exl3::exl3_map_group_to_logical("mtp.layers.x.self_attn.q_proj") ==
          "mtp.layers.x.self_attn.q_proj");

    // --- fusion plan over a synthetic checkpoint ---
    Exl3CheckpointMeta meta;
    const std::string a5 = "model.language_model.layers.5.self_attn";
    const std::string m5 = "model.language_model.layers.5.mlp";
    meta.groups.push_back(mk(a5 + ".q_proj", 4, 256, 384));
    meta.groups.push_back(mk(a5 + ".k_proj", 4, 256, 128));
    meta.groups.push_back(mk(a5 + ".v_proj", 4, 256, 128));
    meta.groups.push_back(mk(a5 + ".o_proj", 5, 384, 256)); // K5 dense
    meta.groups.push_back(mk(m5 + ".gate_proj", 3, 256, 256));
    meta.groups.push_back(mk(m5 + ".up_proj", 3, 256, 256));
    meta.groups.push_back(mk(m5 + ".down_proj", 4, 256, 256));
    meta.groups.push_back(mk("lm_head", 6, 256, 1024)); // K6 dense
    meta.groups.push_back(
        mk("model.language_model.layers.0.linear_attn.in_proj_qkv", 4, 256, 512));
    // Layer 9: K-mismatched triple -> dense fallback (k uniform, K differs).
    const std::string a9 = "model.language_model.layers.9.self_attn";
    meta.groups.push_back(mk(a9 + ".q_proj", 4, 256, 384));
    meta.groups.push_back(mk(a9 + ".k_proj", 3, 256, 128));
    meta.groups.push_back(mk(a9 + ".v_proj", 4, 256, 128));

    const auto plan = ninfer::exl3::exl3_plan_fusion(meta);
    // 12 input groups -> 9 logical entries (2 fused + 7 dense).
    CHECK(plan.size() == 9);

    const auto* qkv = find_by_logical(plan, "text/layers/5/attention/qkv");
    CHECK(qkv != nullptr);
    if (qkv != nullptr) {
        CHECK(qkv->groups == 3);
        CHECK(qkv->group_n[0] == 384);
        CHECK(qkv->group_n[1] == 128);
        CHECK(qkv->group_n[2] == 128);
        CHECK(qkv->k == 256);
        CHECK(qkv->n == 640);
        CHECK(qkv->bits == 4);
        CHECK(qkv->cb == 2);
        CHECK(qkv->has_mul1 && !qkv->has_mcg);
        CHECK(qkv->members.size() == 3);
        if (qkv->members.size() == 3) {
            CHECK(qkv->members[0] == a5 + ".q_proj");
            CHECK(qkv->members[1] == a5 + ".k_proj");
            CHECK(qkv->members[2] == a5 + ".v_proj");
        }
        CHECK(qkv->member_logical.size() == 3);
        if (qkv->member_logical.size() == 3) {
            CHECK(qkv->member_logical[0] == "text/layers/5/attention/query");
            CHECK(qkv->member_logical[1] == "text/layers/5/attention/key");
            CHECK(qkv->member_logical[2] == "text/layers/5/attention/value");
        }
    }

    const auto* gu = find_by_logical(plan, "text/layers/5/mlp/gate_up");
    CHECK(gu != nullptr);
    if (gu != nullptr) {
        CHECK(gu->groups == 2);
        CHECK(gu->group_n[0] == 256);
        CHECK(gu->group_n[1] == 256);
        CHECK(gu->k == 256);
        CHECK(gu->n == 512);
        CHECK(gu->bits == 3);
        CHECK(gu->members.size() == 2);
        if (gu->members.size() == 2) {
            CHECK(gu->members[0] == m5 + ".gate_proj");
            CHECK(gu->members[1] == m5 + ".up_proj");
        }
    }

    // Dense singles keep groups=1 with their K intact (K5/K6 route via v3).
    const auto* o = find_by_logical(plan, "text/layers/5/attention/output");
    CHECK(o != nullptr && o->groups == 1 && o->bits == 5 && o->members.size() == 1);
    const auto* head = find_by_logical(plan, "text/output_head");
    CHECK(head != nullptr && head->groups == 1 && head->bits == 6 && head->n == 1024);
    const auto* down = find_by_logical(plan, "text/layers/5/mlp/down");
    CHECK(down != nullptr && down->groups == 1 && down->bits == 4);
    const auto* gqkv = find_by_logical(plan, "text/layers/0/gdn/qkv");
    CHECK(gqkv != nullptr && gqkv->groups == 1 && gqkv->bits == 4);

    // Mismatched layer-9 triple falls back to three dense singles.
    const auto* q9 = find_by_logical(plan, "text/layers/9/attention/query");
    const auto* k9 = find_by_logical(plan, "text/layers/9/attention/key");
    const auto* v9 = find_by_logical(plan, "text/layers/9/attention/value");
    CHECK(q9 != nullptr && q9->groups == 1 && q9->bits == 4);
    CHECK(k9 != nullptr && k9->groups == 1 && k9->bits == 3);
    CHECK(v9 != nullptr && v9->groups == 1 && v9->bits == 4);
    CHECK(find_by_logical(plan, "text/layers/9/attention/qkv") == nullptr);

    // --- mtp fusion over synthetic mtp groups (real S3c geometry, K4 mul1) ---
    // The generic planner treats mtp triples/pairs exactly like text ones:
    // 8 groups -> 5 logical (qkv groups=3 thirds, gate_up groups=2 halves,
    // fc/o/down dense).
    Exl3CheckpointMeta mtp_meta;
    const std::string mtpa = "mtp.layers.0.self_attn";
    const std::string mtpm = "mtp.layers.0.mlp";
    mtp_meta.groups.push_back(mk(mtpa + ".q_proj", 4, 5120, 12288));
    mtp_meta.groups.push_back(mk(mtpa + ".k_proj", 4, 5120, 1024));
    mtp_meta.groups.push_back(mk(mtpa + ".v_proj", 4, 5120, 1024));
    mtp_meta.groups.push_back(mk(mtpa + ".o_proj", 4, 6144, 5120));
    mtp_meta.groups.push_back(mk(mtpm + ".gate_proj", 4, 5120, 17408));
    mtp_meta.groups.push_back(mk(mtpm + ".up_proj", 4, 5120, 17408));
    mtp_meta.groups.push_back(mk(mtpm + ".down_proj", 4, 17408, 5120));
    mtp_meta.groups.push_back(mk("mtp.fc", 4, 10240, 5120));
    const auto mtp_plan = ninfer::exl3::exl3_plan_fusion(mtp_meta);
    CHECK(mtp_plan.size() == 5);
    const auto* mtp_qkv = find_by_logical(mtp_plan, "text/mtp/layers/0/attention/qkv");
    CHECK(mtp_qkv != nullptr);
    if (mtp_qkv != nullptr) {
        CHECK(mtp_qkv->groups == 3);
        CHECK(mtp_qkv->group_n[0] == 12288);
        CHECK(mtp_qkv->group_n[1] == 1024);
        CHECK(mtp_qkv->group_n[2] == 1024);
        CHECK(mtp_qkv->k == 5120);
        CHECK(mtp_qkv->n == 14336);
        CHECK(mtp_qkv->bits == 4);
        CHECK(mtp_qkv->members.size() == 3);
        if (mtp_qkv->members.size() == 3) {
            CHECK(mtp_qkv->members[0] == mtpa + ".q_proj");
            CHECK(mtp_qkv->members[1] == mtpa + ".k_proj");
            CHECK(mtp_qkv->members[2] == mtpa + ".v_proj");
        }
    }
    const auto* mtp_gu = find_by_logical(mtp_plan, "text/mtp/layers/0/mlp/gate_up");
    CHECK(mtp_gu != nullptr);
    if (mtp_gu != nullptr) {
        CHECK(mtp_gu->groups == 2);
        CHECK(mtp_gu->group_n[0] == 17408);
        CHECK(mtp_gu->group_n[1] == 17408);
        CHECK(mtp_gu->k == 5120);
        CHECK(mtp_gu->n == 34816);
    }
    const auto* mtp_fc = find_by_logical(mtp_plan, "text/mtp/draft_head");
    CHECK(mtp_fc != nullptr && mtp_fc->groups == 1 && mtp_fc->k == 10240 && mtp_fc->n == 5120);
    const auto* mtp_o = find_by_logical(mtp_plan, "text/mtp/layers/0/attention/output");
    CHECK(mtp_o != nullptr && mtp_o->groups == 1 && mtp_o->k == 6144);
    const auto* mtp_down = find_by_logical(mtp_plan, "text/mtp/layers/0/mlp/down");
    CHECK(mtp_down != nullptr && mtp_down->groups == 1 && mtp_down->k == 17408);
    {
        std::size_t mtp_total = 0;
        for (const auto& e : mtp_plan) mtp_total += e.members.size();
        CHECK(mtp_total == mtp_meta.groups.size());
    }

    // Coverage: every input group consumed exactly once.
    std::size_t member_total = 0;
    for (const auto& e : plan) member_total += e.members.size();
    CHECK(member_total == meta.groups.size());
    for (const auto& g : meta.groups) {
        int hits = 0;
        for (const auto& e : plan) {
            for (const auto& m : e.members) hits += (m == g.name) ? 1 : 0;
        }
        if (hits != 1) {
            ++failures;
            std::cout << "FAIL coverage: " << g.name << " hits=" << hits << "\n";
        }
    }

    if (failures == 0) std::cout << "PASS exl3_bind_namemap\n";
    return (failures == 0) ? 0 : 1;
}
