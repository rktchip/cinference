# HyperQwen (rktchip) — designs only, zero code

- Source: https://github.com/rktchip/HyperQwen.git at `fe09615`
- Author: rktchip. A vLLM 0.28 fork (43 Python/Triton patches + KVarN
  glue). Verified hard negative twice: zero `.cu`/`.cuh`/`.cpp`/`.rs`
  repo-wide — no portable C++/CUDA exists to take.

## Borrowed (methodology)

- Continuous-batching mechanism (interleaved mixed-step descriptors)
  behind `src/batch/` + the serve pump. Concept port, our code.
- DFlash2 draft algorithm (n-gram lookup + quantized candidate chains);
  informs `draft.cpp` wiring.
- KVarN concept (Hadamard-rotated 4-bit K / 2-bit V per 128-token tile);
  top long-context play, scoped as backend work.
- Measured anchors: 1,094 tok/s e2e at 64-concurrent; draft-vocab
  coverage 92.1 -> 97.5% (+10 tok/s method); GPTQ int4 lm_head
  (KL 0.0068 -> 0.0029, -1.8 ms/step).
- The MTP-finetune NEGATIVE (7M tokens, no needle movement) — saved a 6h
  GPU burn; ship the checkpoint MTP as-is.

## Not borrowed (could still take, all need hand-rolled CUDA backends)

1. KVarN KV compression (1.6–4x pool) — backend-level change.
2. DFlash2 k>1 port + degenerate-distribution guard.
3. int8-QK prefill attention (head_dim 256).
4. Sort-free small-k top-k/top-p + multi-block row softmax.
5. Gumbel-salt draft temperature + spec-rung prewarm.

## Never taking

Python speculator, Triton kernels, kvarn-over-`files/vllm` glue,
`offload-wsl2-devptr.patch` (different stack).
