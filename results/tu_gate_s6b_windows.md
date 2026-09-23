# S6b Windows TU gate -- verdict: OK (21/21)

Date (UTC): 2026-09-22. Box: Windows 11, MSVC 19.44.35228 for x64
(BuildTools 2022, vcvarsall x64 from the x86 path), nvcc release 13.3
V13.3.73, sm_120a.

Scope: the 19 TUs from results/tu_gate_s6_windows.md recompiled against
current on-disk content (wave-1 landings included), PLUS the S2
hook_loop TUs that postdated that table: src/serve/hook_loop.cpp and
its consumer src/serve/generation_service.cpp (which includes
serve/hook_loop.h). Headers (weight.h, dispatch.h, hadamard.cuh,
materialize.h, batch/*.h, hook_loop.h) are covered via including TUs.
Objects in Hermes scratch s6b_gate/obj (NOT in repo, never in-source).

## cl /nologo /std:c++20 /c, includes: include/ src/ third_party/ CUDA/include

| TU | result | note |
|----|--------|------|
| src/ops/linear/linear.cpp | OK | still green after wave-1 edits |
| src/artifact/formats.cpp | OK | benign C4530 only |
| src/batch/request.cc | OK | |
| src/batch/scheduler.cc | OK | |
| src/batch/cinference_hooks.cc | OK | |
| tests/test_continuous_batch.cc | OK | |
| tests/ops/test_exl3_dequant.cc | OK | benign C4530 only |
| tests/ops/test_exl3_launch_m47.cc | OK | |
| src/serve/hook_loop.cpp (S2, new) | OK | benign C4530 (chrono header) only |
| src/serve/generation_service.cpp (S2 consumer, new) | OK | benign C4530 (chrono header) only |

## nvcc -std=c++20 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler "/Zc:preprocessor" -c, includes: include/ src/ third_party/ exl3/

| TU | result | note |
|----|--------|------|
| src/ops/linear/exl3/exl3_launcher.cu | OK | known benign 20012-D shim warnings only |
| src/ops/linear/exl3/exl3_dispatch.cu | OK | same 20012-D only |
| src/ops/linear/exl3/exl3_prefill.cu | OK | same 20012-D only |
| src/ops/linear/exl3/exl3_materialize.cpp | OK | |
| src/ops/linear/exl3/exl3_aln.cu | OK | |
| src/ops/linear/exl3/exl3_op.cu | OK | |
| src/ops/linear/exl3/exl3_bind.cu | OK | |
| src/ops/linear/exl3/exl3_v3_gemm.cu | OK | includes torchexl3/exl3_gemm.cu via wrapper |
| src/ops/linear/exl3/exl3_v3_had.cu | OK | includes torchexl3/exl3_hadamard.cu via wrapper |
| src/batch/batch.cu | OK | |
| src/batch/paged_kv.cu | OK | |

Excluded (not a CMake TU): src/batch/paged_kv.cu.txt (reference copy,
device code in .txt, never compiled by the build).

## Dep absence (unchanged, re-verified this run)

- WSL2: NOT INSTALLED. `wsl --status` -> `The Windows Subsystem for
  Linux is not installed...` (exit 50). No install attempted (admin +
  reboot = user action). Branch (a) taken; see
  results/linux_build_runbook.md for the exact post-WSL2 sequence.
- Full Windows configure still cannot proceed: `Could NOT find
  PkgConfig` is the first fatal error (pkg-config, FFmpeg dev, and
  libcurl dev all absent). No Windows link of serve binaries is
  claimed; the Linux-only gate at apps/CMakeLists.txt:1-5 stands.

## Staleness guarantee

Final sweep compared every gated source mtime against its .obj: all 21
CURRENT at table time. Per-TU logs kept alongside the objects in
s6b_gate/obj.
