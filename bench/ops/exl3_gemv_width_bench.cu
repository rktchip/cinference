// exl3_gemv_width_bench.cu — same-shape EXL3 GEMV width microbench.
//
// Times the repo's own m=1 gemv kernel entry point
// (ninfer::exl3::exl3_gemv_run, the exl3_gemv_plain kernel without the
// Hadamard in/out + cast stages) at the real Qwen3.8-27B per-layer linear
// shapes, for bit-widths {3,4}. Synthetic random quantized weights, no model
// files, timing only, no numerics checks.
//
// Shapes (H=5120; Q:24x256, KV:4x256, GDN:16x128k/48x128v, MLP dense 17408):
//   attn_q    K=5120  N=6144   (also covers gdn_v / gdn_z, same KxN)
//   attn_kv   K=5120  N=1024
//   attn_o    K=6144  N=5120
//   gdn_qk    K=5120  N=2048
//   mlp_gateup K=5120 N=17408
//   mlp_down  K=17408 N=5120
//
// Plan mirrors the checkpoint: mcg=false, mul1=true -> cb2 (the Qwen3.8-27B
// EXL3 checkpoint is 100% mul1), c_fp32=false, mmode=0 (m=1), cc/num_sms from
// the live device. Both widths use cb2 so the comparison isolates bit-width.
//
// Bits {5,6,8} are NOT covered: the vendored gemv kernel static_asserts
// bits<=4 and the dispatch routes K>=5 to the reconstruct/dequant path, so a
// same-kernel width sweep is impossible there (not a trivial format switch).
//
// Bytes (read denominator, per shape+width):
//   trellis = K*N*bits/8            (TWORDS=8*bits u32 per 16x16 tile)
//   scales  = 2*K + 2*N             (fp16 suh [K] + svh [N])
//   total   = trellis + scales
// Effective GB/s = total / median_us / 1e3 (decimal GB).
// NOTE: the timed entry point (exl3_gemv_run) takes only the trellis B; the
// scales are counted in the denominator per the bench spec as the full
// decode-read cost, not because this launch touches them.
//
// Example:
//   ./build/bench/ninfer_exl3_gemv_width_bench
//   ./build/bench/ninfer_exl3_gemv_width_bench --bits 3 --repeat 500 --csv-out /tmp/w.csv

#include "ops/linear/exl3/exl3_launcher.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

struct Shape {
    const char* label;
    int k;
    int n;
};

// Distinct per-layer KxN at H=5120 (see file header for provenance).
constexpr Shape kShapes[] = {
    {"attn_q", 5120, 6144},      // + gdn_v / gdn_z (same KxN)
    {"attn_kv", 5120, 1024},     // k_proj + v_proj (same KxN)
    {"attn_o", 6144, 5120},
    {"gdn_qk", 5120, 2048},      // q_proj + k_proj (same KxN)
    {"mlp_gateup", 5120, 17408}, // gate_proj + up_proj (same KxN)
    {"mlp_down", 17408, 5120},
};

struct Options {
    int warmup = 20;
    int repeat = 200;
    std::vector<int> bits{3, 4};
    std::string csv_out;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_exl3_gemv_width_bench [--bits 3,4|3|4] "
                 "[--warmup N] [--repeat N] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

int parse_int(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum) usage(flag);
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::puts("usage: ninfer_exl3_gemv_width_bench [--bits 3,4|3|4] "
                      "[--warmup N] [--repeat N] [--csv-out PATH]");
            std::exit(0);
        }
        if (++i >= argc) usage("missing option value");
        const std::string_view value(i < argc ? argv[i] : "");
        if (arg == "--warmup") {
            options.warmup = parse_int(value, 0, 100000, "--warmup");
        } else if (arg == "--repeat") {
            options.repeat = parse_int(value, 1, 1000000, "--repeat");
        } else if (arg == "--bits") {
            options.bits.clear();
            std::string_view rest(value);
            while (!rest.empty()) {
                const std::size_t comma = rest.find(',');
                const int b = parse_int(rest.substr(0, comma), 3, 4, "--bits (only 3,4 have gemv kernels)");
                if (std::find(options.bits.begin(), options.bits.end(), b) == options.bits.end())
                    options.bits.push_back(b);
                if (comma == std::string_view::npos) break;
                rest.remove_prefix(comma + 1);
            }
            if (options.bits.empty()) usage("--bits");
        } else if (arg == "--csv-out") {
            if (value.empty()) usage("--csv-out");
            options.csv_out = std::string(value);
        } else {
            usage("unknown flag");
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    const int cc = prop.major * 10 + prop.minor;
    const int num_sms = static_cast<int>(prop.multiProcessorCount);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::mt19937 rng(0xC0FFEEu);

    std::printf("# device=%s cc=%d sms=%d m=1 cb=2(mul1) c_fp32=0 warmup=%d repeat=%d\n", prop.name,
                cc, num_sms, options.warmup, options.repeat);
    std::printf("# bytes = trellis(K*N*bits/8) + scales(2*K+2*N); gbs = bytes/median_us/1e3\n");
    std::printf("%-12s %6s %6s %4s %12s %10s %12s %10s %8s %4s %5s\n", "shape", "K", "N", "bits",
                "trellis_B", "scales_B", "total_B", "median_us", "GB/s", "cfg", "grid");

    struct Row {
        std::string label;
        int k, n, bits, cfg, grid;
        std::uint64_t trellis, scales, total;
        double median_us, gbs;
    };
    std::vector<Row> rows;

    for (const Shape& shape : kShapes) {
        for (const int bits : options.bits) {
            exl3::Exl3GemvPlan plan{};
            // Checkpoint-faithful: mcg=false, mul1=true -> cb2 for both widths.
            const bool planned = exl3::exl3_gemv_try_plan(plan, /*size_m=*/1, shape.k, shape.n,
                                                         bits, /*mcg=*/false, /*mul1=*/true,
                                                         /*c_fp32=*/false, cc, num_sms);
            const std::uint64_t trellis =
                static_cast<std::uint64_t>(shape.k) * static_cast<std::uint64_t>(shape.n) *
                static_cast<std::uint64_t>(bits) / 8ULL;
            const std::uint64_t scales =
                2ULL * static_cast<std::uint64_t>(shape.k) + 2ULL * static_cast<std::uint64_t>(shape.n);
            if (!planned || !plan.ok) {
                std::printf("%-12s %6d %6d %4d %12llu %10llu %12llu %10s %8s %4s %5s\n",
                            shape.label, shape.k, shape.n, bits, (unsigned long long)trellis,
                            (unsigned long long)scales, (unsigned long long)(trellis + scales),
                            "NO_PLAN", "-", "-", "-");
                continue;
            }
            // Synthetic random weights: trellis B as random u16 words (TWORDS =
            // 8*bits u32 per 16x16 tile -> K*N*bits/8 bytes), A as random fp16.
            const std::size_t b_words = static_cast<std::size_t>(trellis / 2ULL);
            std::vector<std::uint16_t> host_b(b_words);
            for (auto& w : host_b) w = static_cast<std::uint16_t>(rng());
            std::vector<std::uint16_t> host_a(static_cast<std::size_t>(shape.k));
            for (auto& w : host_a) w = static_cast<std::uint16_t>(0x3800u | (rng() & 0x3FFu));

            DeviceBuffer a_buf(static_cast<std::size_t>(shape.k) * 2ULL);
            DeviceBuffer b_buf(trellis);
            DeviceBuffer c_buf(static_cast<std::size_t>(shape.n) * 2ULL);
            a_buf.copy_from_host(host_a.data(), a_buf.bytes);
            b_buf.copy_from_host(host_b.data(), b_buf.bytes);
            c_buf.fill(0);

            const auto launch = [&](cudaStream_t launch_stream) {
                if (!exl3::exl3_gemv_run(plan, reinterpret_cast<const half*>(a_buf.p),
                                        reinterpret_cast<const std::uint16_t*>(b_buf.p), c_buf.p,
                                        launch_stream)) {
                    throw std::runtime_error("exl3_gemv_run failed");
                }
            };
            const bench::ColdTiming timing =
                bench::measure_launch(launch, stream, options.warmup, options.repeat);
            const std::uint64_t total = trellis + scales;
            const double gbs = static_cast<double>(total) / timing.median_us / 1.0e3;
            std::printf("%-12s %6d %6d %4d %12llu %10llu %12llu %10.3f %8.1f %4d %5d\n",
                        shape.label, shape.k, shape.n, bits, (unsigned long long)trellis,
                        (unsigned long long)scales, (unsigned long long)total, timing.median_us,
                        gbs, plan.cfg, plan.grid);
            rows.push_back({shape.label, shape.k, shape.n, bits, plan.cfg, plan.grid, trellis,
                            scales, total, timing.median_us, gbs});
        }
    }

    if (!options.csv_out.empty()) {
        FILE* csv = std::fopen(options.csv_out.c_str(), "w");
        if (csv == nullptr) {
            std::fprintf(stderr, "error: cannot open %s\n", options.csv_out.c_str());
            return 1;
        }
        std::fputs("shape,K,N,bits,trellis_bytes,scales_bytes,total_bytes,median_us,gbs,cfg,grid\n",
                   csv);
        for (const Row& row : rows) {
            std::fprintf(csv, "%s,%d,%d,%d,%llu,%llu,%llu,%.4f,%.3f,%d,%d\n", row.label.c_str(),
                         row.k, row.n, row.bits, (unsigned long long)row.trellis,
                         (unsigned long long)row.scales, (unsigned long long)row.total,
                         row.median_us, row.gbs, row.cfg, row.grid);
        }
        std::fclose(csv);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
