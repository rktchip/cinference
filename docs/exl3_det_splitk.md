# Deterministic split-K for the EXL3 v3 GEMM (default ON; `NINFER_EXL3_DETERMINISTIC=0` opts out)

Default behavior (flag unset, empty, or anything but exactly `0`) is the
deterministic path: bit-identical across runs. Opt out with
`NINFER_EXL3_DETERMINISTIC=0` for bit-for-bit the historical kernel. Do NOT
use default-on numbers for timing comparisons against the old path without
noting the extra reduce pass (verify-only cost).

## Problem

`exl3_gemm_m_kernel<SPLIT>` accumulates cross-block partials with `atomicAdd`
into one shared fp32 plane. Block completion order varies run to run, and fp
addition is not associative, so outputs carry ULP noise that flips 0.125-gap
argmaxes in MTP verify at m<=4. `NINFER_EXL3_FORCE_SPLIT1=1` removes the noise
but collapses the grid (kv m=4: 8 blocks vs ~88 with split) -- an occupancy
cliff unusable for anything but debugging.

## Mechanism (flag on)

Same split factors, same grid, same occupancy -- only the accumulation path
changes:

1. **Where partials live.** `acc` grows from 1 plane of `(m, ldc)` fp32 to S
   planes (`S` = the split factor pick_split returns for the shape; the
   autotuner is pinned to that value, see below). Plane `s` =
   `acc[s * (m*ldc) .. ]`, i.e. bytes `S*m*n*4` (ldc == n on the dense path).
2. **Who writes.** GEMM block `(bx, by, s)` plain-stores (no atomics) its fp32
   partials into plane `s` only (`exl3_gemm_m_kernel<..., DET=true>`). Every
   plane element has exactly one writer block (column/row tiles are disjoint
   across blocks; cooperative threads inside a block are disjoint), so block
   completion order cannot affect bits. Out-of-range rows are skipped, never
   stored, preserving the acc zero invariant the reduce relies on.
3. **Who reduces, in what order.** `exl3_det_reduce_epilogue_kernel` runs on
   the same stream right after the GEMM (kernel-boundary ordering, same grid
   as the historical epilogue: one warp per `(row, 128-col-group)`). Each warp
   loads planes `s = 0..S-1` in fixed slab order and folds them with the same
   left fold every run (`((p0+p1)+p2)...`), zeroes each plane element as it
   reads it (zero invariant maintained), then runs the identical Hadamard+svh
   tail via `had128_warp_acc_val` (the arithmetic factored out of
   `had128_warp_acc`; the flag-off epilogue delegates to it with the same op
   order, so flag-off bits are unchanged).
4. **Run-to-run stability of the split itself.** The autotuner otherwise tries
   `base/2` and `base*2` splits and picks by timing -- timing noise would pick
   different splits (different summation orders, different bits) on different
   runs. With the flag on the tuner tries only the cost-model split, a pure
   shape function. The tune-cache key additionally carries the flag, so a
   flag-off entry (possibly a doubled split the plane buffer was not sized
   for) can never replay under flag-on, or vice versa.

## Memory cost

Per split call: `S*m*n*4` bytes vs `m*n*4` historically. Per-call arenas
(`exl3_op.cu`) size exactly; the process-lifetime reserve (`exl3_bind.cu`)
takes the max of `planes(mm)*mm*n*4` over `mm` = powers of two up to `m_max`
plus `m_max` itself, because the split factor varies with runtime batch.

- **lm_head (n=248320): never splits.** Grid is `(1940, ceil(m/128), S)` --
   1940 column blocks already exceed the ~564-block split target at every m,
   so `pick_split` returns 1 and the deterministic path adds **0 bytes**
   (single plane, plain kernel, no reduce). Hypothetical S=16 bound at m=4:
   60.6 MiB total (16 x 3.79 MiB/plane).
- **Small-n split example (attn_kv-like, m=4, S~11, n=2048):** 0.34 MiB total
   (11 planes x 32 KiB).

## ULP note (read before comparing)

Flag-on outputs **differ from flag-off at ULP level by design**: fixed slab
order `((p0+p1)+p2)...` vs completion-order atomic accumulation are different
summation orders of the same terms. Do not try to match bits across the flag.
The guarantee is within flag-on: identical inputs give bit-identical outputs
across runs. Residual nondeterminism outside this path is out of scope (e.g.
the MoE fused-combine `had128_warp_out_acc` atomics, a different path).

## Files

- `src/ops/linear/exl3/torchexl3/exl3_gemm.cu` -- DET kernel variant, ordered
  reduce kernel + launcher, dispatch/tuner wiring, `exl3_det_planes_row` /
  `exl3_det_enabled` exports.
- `src/ops/linear/exl3/torchexl3/exl3_had.cuh` -- `had128_warp_acc_val`
  factor-out (flag-off delegates, same op order).
- `src/ops/linear/exl3/exl3_aln.{h,cu}` -- flag-aware `exl3_aln_ws_bytes`,
  new `exl3_aln_acc_bytes` / `exl3_aln_det_planes` / `exl3_aln_det_enabled`.
- `src/ops/linear/exl3/exl3_op.cu` -- memset uses `exl3_aln_acc_bytes`.
- `src/ops/linear/exl3/exl3_bind.{h,cu}` -- reserve covers S planes over the
  servable batch grid (flag off: historical size).

## Build + verify (operator, GPU machine)

```bash
# from repo root, on branch det-splitk
cmake --preset dev          # or: cmake -S . -B build
cmake --build build -j
```

```bash
# 1. default on: bit-identical across runs
./build/<mtp-verify-binary> --reps 5 > /tmp/det_on_1.log 2>&1
./build/<mtp-verify-binary> --reps 5 > /tmp/det_on_2.log 2>&1
cmp /tmp/det_on_1.log /tmp/det_on_2.log && echo DETERMINISTIC

# 2. opted out: historical behavior (sanity, zero-change gate vs pre-port)
NINFER_EXL3_DETERMINISTIC=0 ./build/<mtp-verify-binary> --reps 5 2>&1 | tee /tmp/det_off.log

# 3. occupancy spot-check (flag on must keep split grid, e.g. kv m=4 ~88 blocks,
#    not the 8-block split=1 cliff): Nsight Compute `sm__warps_active` /
#    grid-size counter on the exl3_gemm_m_kernel launch, or
#    CUDA_EXL3_SPLIT_TARGET / verbose launch logging if wired.

# 4. cross-flag diff: expect ULP-level-only differences, no argmax flips at
#    0.125 gaps beyond the pre-existing noise envelope going away.
```

Do not combine with `NINFER_EXL3_FORCE_SPLIT1=1` for verification (it forces
S=1, which trivially bypasses the plane path). Never use flag-on numbers for
timing -- the extra reduce pass and S-times acc traffic are verify-only cost.
