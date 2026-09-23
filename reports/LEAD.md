# LEAD — wave-3 integration status

## File ownership (active wave)

| Swarm | Owns | Must not edit |
|---|---|---|
| A serve path | src/serve/generation_service.cpp, src/serve/hook_loop.*, apps/serve/main.cpp | EXL3 kernels, pager |
| B one forward | src/batch/cinference_hooks.cc, runtime set_prefill_lane / transaction loop | serve HTTP |
| C EXL3 load | DONE (S3b/S3c verified) — exl3_bind.*, model_instance.cpp load branch | — |
| D gemv dtype | DONE (S1b EXACT) — exl3_op.cu epilogue read-only verify | — |
| E attention tables | attention call site + batch.cu consumer | packer itself |
| F MTP glue | program/speculative/mtp.cpp into hook_loop (after B) | DFlash2, window 10 |
| G proof | tests/* only | production code |
| H linux link | WSL2/CMake/apps/CMakeLists.txt (blocked: no WSL2) | feature code |

## Blocked-on map

- F starts after B confirms one-forward (B substance already proven by S1/S1b/S5b; F briefed to proceed on the stable StepDispatch contract).
- H blocked on human WSL2 install (runbook at results/linux_build_runbook.md).
- E2E two-client blocked on H/WSL2.

## Last litmus (lead-verified 2026-09-23)

1. Engine construct on EXL3 dir: PASS (diagnostic removed; src/runtime/engine/model_instance.cpp:174-175 routes to exl3_engine_construct_from_dir; device run Linux-gated).
2. Token loop in generation_service: BOTH (engine_->submit executes :370, hook_loop admits :369; schedule_step has zero live serve callers).
3. Mixed StepPlan M + qkv launches: YES (M==9 host green; qkv==1/gate_up==1; out-dtype EXACT maxAbs 0.0).
4. Two HTTP clients: linux-gated (null until WSL2).
