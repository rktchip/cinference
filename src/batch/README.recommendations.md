# Recommended 5090 concurrency settings — Qwen3.8-27B EXL3 3.5bpw

Portable: numbers derived from `C:/models/Qwen3.8-27B-EXL3-3.5bpw/config.json`
geometry + HyperQwen's documented batch measurement tables (batch/README.md),
retargeted to this 32.6 GB card. Tolerance band: verify against the
pool-size print at first Linux boot; the pool self-corrects the estimates here.

## Model geometry (real, from local config.json)

- Hybrid 64 layers, `full_attention_interval: 4` → **16 attention / 48 GDN**.
- Attention KV per token: 4 KV heads × 256 dim × (K+V) × 16 = 32K elements
  → 64 KiB fp16 / 16 KiB fp8 / 8 KiB per token (int4).
- Recurrent (GDN) state per sequence: 48 × (16 heads × 128 × 128)
  → ~25 MB fp16 per request. Confirm against the state-size print
  on first boot; arithmetic from config, not yet measured.
- Weights: 15.0 GB (3.5bpw main) + ~1.3 GB (DFlash2 draft companion).

## VRAM budget (32.6 GB, single card)

    weights          ~15.0 GB
    draft + buffers  ~1.5–2.5 GB
    activations      ~1.5 GB (width-8, chunk 2048)
    ------------------------------
    KV + state pool  ~ 14–15 GB

## Settings

| Knob | Value | Rationale |
|---|---|---|
| KV dtype | 4-bit | 8 KiB/token → ~1.7M resident tokens in a 14 GB pool. fp8 halves that to ~850k; fp16 is not viable at this context |
| `MAX_SEQS` | 64 | Per-request state is ~25 MB, slots are cheap; memory is the real ceiling |
| `MAX_LEN` | 200k | 1.7M-token pool ÷ 200k × 64 = exactly at the pool ceiling; 262k only for 1–2 concurrent sessions |
| Chunk | 2048 | HyperQwen's prefill budget basis (~1.8k tok/s/lane). Staged port tests at 512 for latency probes; prod flag raises it |
| Recurrent state | fp16 | 64 × 200k ≈ 1.6 GB worst-case state total; fp8 path not in the port |
| `PREFIX_CACHE` | 1 (shared-prompt workloads) | HyperQwen measured 13× on 64 reqs sharing a 5.8k prompt; costs ~14% of pool (one extra state page per request). 0 for disjoint one-off traffic |
| GPU utilization | 0.97 (0.93 with prompt logprobs) | Documented ceiling from their gotchas — do not raise |

## Expected capacity (4-bit, 64 slots, 1.75M-token pool)

- **64 × ~26k avg** (128/512-type workload): 600–1000 tok/s aggregate
  (their W4A16-default and int8-gate rows; the int8-all row is not
  reproducible on EXL3 weights)
- **8 × 200k** heavy-context concurrent (1.6M tokens)
- **1 × 150k** single session with ~15 slots of headroom

## Integration anchors (for the Linux round)

- Reuse/state-resume boundary: `KVPrefixForkReservation` in
  `src/models/qwen3_5/program/storage/kv_store.h` is the upstream anchor the
  batch-admission loop binds to; the batch port's table maps into it after
  the 4:1 logical→physical bridge lands.
- Per-step cost at width stays flat with the device-side gather: the host
  assembles tokens/offsets/rows only (O(total_tokens + num_seqs), not
  O(num_seqs × max_blocks)); the compact matrix is built only on the
  host-only fallback path where no gather kernel exists (`AssembleBatch`).
- Tolerance check at first boot: confirm the KV dtype and pool size prints;
  if only fp8 is exposed, resplit `MAX_SEQS`↔`MAX_LEN` against the 850k-token
  pool (64 × ~12k or 8 × 100k).