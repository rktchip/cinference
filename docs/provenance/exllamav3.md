# exllamav3 1.5.1 (turboderp) — reference only, never linked

- Source: https://github.com/turboderp/exllamav3 (pip tree, 1.5.1)
- Author: turboderp (ExLlamaV3).
- Role in this tree: ground-truth oracle. Every numeric gate (P10–P16,
  MG, OP, BIND, K5K6) oracles against it. Not vendored, not linked,
  not shipped.

## Borrowed (knowledge, not code)

- Trellis contract: int16 `[n/16, k/16, 16K]` tiles, k-words fastest;
  per-tensor float bitrate K; odd .5 rates use the mul1 codebook (cb2).
- Codebook rule: `mcg -> cb1, mul1 -> cb2` (not presence-based).
- Dequant semantics: `reconstruct_slice` / `reconstruct_had` byte-exact
  behavior, `had_r_128` grid convention (`gridDim.y` = plane index).
- QTIP gemv lineage (m16n8k16 MMA, fp16 accum) behind the m=1 legacy path.
- `LinearEXL3.forward` as the full-chain reference for P12b/P12c/K5K6.

## Not borrowed (and why)

- The entire implementation — our kernels are cuda-exl3's, our host is
  our seam. 1.5.1 stays across the process boundary as a test oracle.
- The Python loader — our `exl3_materialize` + `exl3_bind` replace it in
  C++/CUDA (flat + wrapped safetensors headers, index.json, geometry
  invariants).
- The quantizer — the checkpoint is pre-quantized; we never quantize.
