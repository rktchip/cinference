# Base: NInfer + Cinference fork + checkpoint

## NInfer (Neroued)

- Source: https://github.com/Neroued/ninfer
- What we kept: the native C++/CUDA engine shape — CLI, serve skeleton,
  MTP/draft paths, CUDA-graph machinery, artifact system.
- What we changed: essentially the execution path (EXL3 linears, batch
  pager, serve pump, sampler). Upstream concepts, our execution.

## Cinference fork changes (satellitedown)

- MTP-10 window (narrowed to decode-only window 3 in our pump),
  capture-based graph reuse, enlarged CPU/GPU round buffers, Huihui NVFP4
  installer path. Retained; see repo history (`b74044f` MTP-10 publish).

## Checkpoint Qwen3.8-27B-EXL3-3.5bpw

- Base model Alibaba Qwen (`Qwen/Qwen3.8-27B`); local EXL3 quant via
  exllamav3 (`library_name: exllamav3`).
- 409 trellis groups (270 K4 / 137 K3 / 1 K5 o_proj L63 / 1 K6 lm_head),
  all mul1-only; fused qkv (n=10240); ships its own MTP draft head.
- Layout contract: `weight_map` (409) is the group universe;
  `tensor_storage` (707) omits the 8 `mtp.*` groups. Safetensors headers
  are per-file truth for byte offsets; `data_offsets` are (start, end).
- Servability: every group serves (K5/K6 via v3, Gate K5K6); file is
  clean (0 non-finite scales after the corrected recount — the 367/401
  census read fp16 bits as int16 values).
