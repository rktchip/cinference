# cuda-exl3 (Zeuss5) — what we took and what we left

- Source: https://github.com/Zeuss5/cuda-exl3 at `6a1ffc3` ("Record the provenance of the fat kernels")
- Author: cuda-exl3 contributors. License: MIT.
- Role in this tree: the EXL3 GEMM vehicle (criterion 7). Vendored under
  `src/ops/linear/exl3/torchexl3/`, sha-pinned in `VENDORED_SHA256.txt`.

## Borrowed (6 files)

| File | Upstream name | Status |
|---|---|---|
| `exl3_codebook.cuh` | same | byte-identical |
| `exl3_common.cuh` | same | byte-identical |
| `exl3_dq.cuh` | same | byte-identical |
| `exl3_had.cuh` | same | MODIFIED (pin-recorded, see below) |
| `exl3_gemm.cu` | `gemm.cu` | device byte-identical; ATen host layer replaced with raw-pointer seam |
| `exl3_hadamard.cu` | `hadamard.cu` | kernels byte-identical; ATen wrappers replaced with `_row` seams |

What this buys: single M-tiled fused EXL3 GEMM (`exl3_gemm_m_kernel`),
multi-shard single-launch rows via `ShardMap`, epilogue Hadamard, and the
per-shape autotuner (`autotune_cfg` over BM candidates with `pick_bm`
fallback — verified intact after the host replacement, not assumed).
Bench: 8x/25x/26x over the SIMT prefill at m=16/64/128.

Seam-only changes: `namespace cuda_exl3` wrap, `v3_capturing()` helper,
`exl3_pick_split_row` de-inline, `exl3_gemm_row(s)` multi-shard entries.

## Our one device-adjacent edit (inside the vendored file)

`had_hybrid_mul1/mul2` + 10 scale call sites in `exl3_had.cuh`: in-range
fp16 product verbatim (bit-exact), exact fp32 fallback only on
overflow-with-finite-scale. Recorded in the pin file as
`MODIFIED ... (pristine was dfa6f331)`. Post-two-curl slot decides
revert-vs-fork (see README Hybrid note). The 10 sites span 5 warp
functions feeding both had_in kernels AND the GEMM epilogue
(`had128_warp_acc/out`), so this is load-bearing, not a wrapper.

## Not borrowed

- `bindings.cpp` (torch/extension pybind + op registration) — would drag
  libtorch into a server with no Python on the step. Our `exl3_op` /
  `exl3_bind` layer is the replacement binding.
- `mla_decode.cu` (fused sparse-MLA decode, SM120) — Qwen has no MLA.
  Revisit only for a DeepSeek-shaped model.
- Python package (`linear.py`, `moe.py`, `ops.py`, `attention.py`,
  `config.py`, `parameter.py`) — torch `nn.Module` scaffolding; `moe.py`
  only matters if an MoE model ever lands.
- `bench/`, `tests/`, `docs/`, `docker/` — their `test_exl3_gemm.py` is
  torch-only (`safe_open framework="pt"`, cuda tensors); no
  bytes-in/bytes-out fixtures to lift.
- Their bench numbers — quoted nowhere as ours. Our numbers are our
  microbenches (P16/MG/OP/BIND/K5K6/K56-BENCH).

## Contract

`VENDORED_SHA256.txt` is drift detection. Any re-vendor diffs against
upstream commit `6a1ffc3`; expected differences are the seam + hybrid
lines only.
