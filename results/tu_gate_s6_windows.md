# S6 Windows TU gate -- verdict: OK (19/19)

Date (UTC): 2026-09-22. Box: Windows 11, MSVC 19.44 (BuildTools 2022),
nvcc 13.3, sm_120a. vcvarsall used (x86 path):
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat x64
(%ProgramFiles% does NOT contain BuildTools on this box; x86 path is correct.)

Scope: every TU other swarms touched (staged + unstaged + untracked at gate
time), plus S5 test TUs that landed mid-run. Headers (weight.h, dispatch.h,
hadamard.cuh, materialize.h, batch/*.h) are covered via including TUs.
Objects in scratch s6_gate/obj (NOT in repo, never in-source).

## cl /nologo /std:c++20 /c, includes: include/ src/ third_party/ CUDA/include

| TU | result | note |
|----|--------|------|
| src/ops/linear/linear.cpp | OK | QType::EXL3 dispatch + capacity cases |
| src/artifact/formats.cpp | OK | exl3 spelling; first pass failed on missing third_party include (harness flag, not repo); re-run green. Benign C4530 only |
| src/batch/request.cc | OK | |
| src/batch/scheduler.cc | OK | recompiled after S1/S4/S5 landing (src newer than first obj); green |
| src/batch/cinference_hooks.cc | OK | recompiled after landing; green |
| tests/test_continuous_batch.cc | OK | recompiled after batch-header landing; green |
| tests/ops/test_exl3_dequant.cc (S5, landed mid-run) | OK | host-only; benign C4530 only |
| tests/ops/test_exl3_launch_m47.cc (S5, landed mid-run) | OK | host-only, no device handle; benign C4530 only |

## nvcc -std=c++20 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler "/Zc:preprocessor" -c, includes: include/ src/ third_party/ exl3/torchexl3/

| TU | result | note |
|----|--------|------|
| src/ops/linear/exl3/exl3_launcher.cu | OK | known benign 20012-D shim warnings only |
| src/ops/linear/exl3/exl3_dispatch.cu | OK | same 20012-D only |
| src/ops/linear/exl3/exl3_prefill.cu | OK | same 20012-D only |
| src/ops/linear/exl3/exl3_materialize.cpp | OK | first pass failed on missing third_party include (harness flag); re-run green |
| src/ops/linear/exl3/exl3_aln.cu | OK | S4 cross-check: compiles clean (claim confirmed) |
| src/ops/linear/exl3/exl3_op.cu | OK | S4 cross-check: compiles clean (claim confirmed) |
| src/ops/linear/exl3/exl3_bind.cu | OK | S4 cross-check: compiles clean (claim confirmed) |
| src/ops/linear/exl3/exl3_v3_gemm.cu | OK | includes torchexl3/exl3_gemm.cu via wrapper; clean |
| src/ops/linear/exl3/exl3_v3_had.cu | OK | includes torchexl3/exl3_hadamard.cu via wrapper; clean |
| src/batch/batch.cu | OK | object newer than landed edit; current |
| src/batch/paged_kv.cu | OK | object newer than landed edit; current |

Excluded (not a CMake TU): src/batch/paged_kv.cu.txt (reference copy, device
code in .txt, never compiled by the build).

## Dep absence (task a)

- pkg-config: ABSENT. `which pkg-config` -> not found (exit 1);
  `pkg-config --version` -> `command not found` (exit 127).
- FFmpeg dev: ABSENT. No avformat/libav headers under /usr/include,
  /usr/local/include, /mingw64/include; zero .pc files on disk. (An
  HP-bundled ffmpeg.exe runtime binary exists; it is not dev material and
  cmake cannot see it without pkg-config.)
- libcurl pkg-config: ABSENT. No curl-config, no /usr/include/curl, no
  /mingw64/include/curl. (msys curl 8.19 runtime binary exists; no dev
  headers, no .pc, so cmake pkg_check_modules(LIBCURL) cannot succeed.)
- Full configure proof (VS-bundled cmake 3.31, explicit -S/-B to scratch,
  -DNINFER_BUILD_APPS=ON): compilers detect fine (MSVC 19.44, nvcc 13.3,
  CUDAToolkit 13.3.73), then FIRST FATAL: `Could NOT find PkgConfig
  (missing: PKG_CONFIG_EXECUTABLE)` (cmake exit 1). FFmpeg/LIBCURL checks
  sit behind PkgConfig, so configure stops here. Stop line honored.
- WSL2: NOT INSTALLED. `wsl --status` -> `The Windows Subsystem for Linux
  is not installed...` (exit 50). No install attempted (admin + reboot =
  user action). Expected blocker, stands.

## Linux-ready CMake wiring (tasks b + d)

Single location: C:/src/cinference/apps/CMakeLists.txt (modified, +12,
unstaged, NOT committed per swarm rules).
- Platform statement: line 5 message(STATUS ...) + comment lines 1-4. This
  is the ONE authoritative place; nothing added to README or top-level
  CMakeLists (do not scatter).
- Binary names: lines 31-32, `add_executable(cinference ALIAS ninfer)` and
  `add_executable(cinference-serve ALIAS ninfer-serve)`. Both share the one
  engine entry (ninfer_engine, linked by ninfer at lines 11-14; serve stack
  via ninfer_serve at line 18). Linux-ready wiring only; no Windows link
  of serve binaries is claimed (configure cannot reach apps/ on this box).

## Staleness guarantee

Final sweep compared every gated source mtime against its .obj: all 19
CURRENT at table time. Mid-run landings (batch scheduler/hooks headers +
sources, S5 tests) were recompiled; table reflects on-disk content.
