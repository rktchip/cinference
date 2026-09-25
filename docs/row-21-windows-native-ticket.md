# Row 21 — Native Windows 11 build + parity gate

Status: PROPOSED 2026-09-25. NEXT-1 in execution order (first when hold lifts). Needs Chip sign-off. No GPU spent. 5090 HELD.

## Goal

`ninfer-serve.exe` built with MSVC (VS2022 BuildTools, `vcvarsall` + `cl`,
CUDA 13.3, Ninja, `-j24`, single `sm_120`), gated to parity with the WSL2
binary that produced 14 / 23. No WSL at runtime.

## Why

All measured numbers today (14 spec-off, 23 spec-on, conc 3.4x, 8/8) come
from `/root/cinference-graph-build` running under `wsl.exe` (Linux binary,
GPU passthrough, weights via `/mnt/c`). Source is portable (no NCCL /
NVSHMEM / fork / epoll in `src/`; only `cpp-httplib` mmap which already has
a Windows path). Remaining risk is build-system + WDDM, not architecture.

## Build spec

- Toolchain: VS2022 BuildTools `vcvarsall x64`, `nvcc` CUDA 13.3,
  `cmake -S/-B` (never in-source), Ninja, `-j24`, `sm_120` only.
- Env (same as WSL gates): `CUDA_EXL3_AUTOTUNE=0`,
  `CUDA_EXL3_SPLIT_TARGET=0`. No other env vars for the gate.
- Checkpoint: `C:/models/Qwen3.8-27B-EXL3-3.5bpw`, `--greedy`.
- Artifacts: `apps/ninfer-serve.exe` + build log + `cmake --build` tail.
  Keep WSL binary untouched as the reference.

## Expected WDDM deltas (not failures)

- TDR watchdog (~2s): long captures/kernels must finish inside it.
- WDDM pin quota smaller than Linux: watch for pin/clone failures at load.
- Graph capture may behave host-side (not WSL record-only): keep the
  immediate-replay + fail-closed rules; re-prove with `[cgraph]/[vgraph]`
  log lines on the exe.
- Display contention: gate with display idle; note any stutter separately.

## Bars (exe must meet ALL, same prompts)

- Health 200, no FATAL on warmup, spec-off or spec-on.
- Spec-off FOX-64 streaming: ~14 ms/tok excl tok1 (allow +10% vs WSL for
  WDDM; anything worse is a blocker with log).
- Paris frozen 6511/314/9338/369 accept-3 (spec-on) + accept-1 rewinds.
- Spec-on FOX-64: ~23 band (same +10% allowance).
- 8/8 quality matrix green (solo/conc P16/R16/P24/R24).
- Conc-1/2/4/8: aggregate scaling healthy, no regression vs WSL numbers
  (1.13/2.3/2.4/2.7s, 44→150 words/s).

## Kill / revert criteria

- Any rewind mismatch vs WSL binary → stop, file log, keep WSL binary.
- Pin/TDR/graph failure that needs arch change → stop, ticket stays open,
  WSL binary remains the product. No engine rewrite inside this ticket.
- Perf miss beyond +10% → profile one nsys session (cuda-only), file top
  ranges, then stop. No fusion kernels in this ticket.

## Cost

One GPU session for build-verify + one for the full gate, serial.
No default flip (already default-on). No 20b dependency, either order.
Needs Chip release of 5090 hold.
