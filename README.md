<!-- Modified by satellitedown for Cinference. Upstream attribution is retained in NOTICE. -->

# Cinference

> **Current contract (2026-09-25, Tier 0).** EXL3 Qwen 3.8 27B serve on one
> RTX 5090: default context/KV **8192**, max concurrency **8 seqs**, spec
> **off**, graphs **ON by default** (`NINFER_SERVE_GRAPH=0` opts out),
> `CUDA_EXL3_AUTOTUNE=0` + `CUDA_EXL3_SPLIT_TARGET=0` required. Measured
> serve, no env vars: spec-off **~14 ms/tok** (FOX-64 streaming, excl tok1,
> replay after warmup; TTFT unchanged; 8/8 matrix green default-on);
> spec-on MTP **~35 ms/tok** (verify graphed under the same flag,
> drafts/bonus/commit rows eager; frozen Paris 6511/314/9338/369 accept-3).
> Old eager numbers (GRAPH=0): spec-off ~27–30, spec-on ~39. No 180: that
> band needs spec that beats one weight pass plus DFlash/graphs
> we do not have. Gate: two-curl T=0 **Paris / Rome**, 24/24.
> 262k context is a future planner + compressed-KV project, not a flag.
> MTP target is decode-only **window 3** (not MTP-10). Numbers from other
> artifacts (NVFP4 tables, scratch microbench projections, HyperQwen
> docs) are marked where they appear and are **not** this engine's.

Built from [NInfer](https://github.com/Neroued/ninfer): the native C++/CUDA
engine for Qwen on a single RTX 5090. Cinference adds EXL3 weight support
and a multi-client serve pump on top of it.

## Purpose

Serve **one EXL3-quantized Qwen 3.8 27B** (`Qwen3.8-27B-EXL3-3.5bpw`, 3.5 bpw) on a **single RTX 5090** to **multiple concurrent HTTP clients**, with iteration-level continuous batching: mixed prefill+decode steps run as one forward at `m = live tokens`, paged KV, no Python on the step. Default context/KV is 8192 (Paris/Rome profile); 262K needs a planner + compressed KV and is a separate project. Performance is judged head-to-head against local baselines by tokens/sec per watt — no borrowed bench numbers are quoted as ours.

## Where we started, where we are

- **Start:** NInfer upstream (single-request decode path, `kMaximumConcurrency=8` cap, MTP<=3, fp16 KV) as modified by the Cinference fork (MTP-10, capture-based graph reuse, enlarged round buffers, Huihui NVFP4 installer path — fork history, not our serve config).
- **Then:** EXL3 weight support built from scratch against the exllamav3 1.5.1 reference: trellis/codebook contract, decode gemv, prefill reconstruct, numeric gates P10-P16/MG/OP/BIND/K5K6 (bit-exact or bounded 1-ulp vs the reference chain).
- **Then:** the cuda-exl3 v3 vehicle swap (criterion 7): ATen-stripped M-tiled fused GEMM + autotuner, multi-group single-launch rows, hybrid Hadamard guard; per-m kernel microbench 8x/25x/26x over the SIMT prefill at m=16/64/128 (kernel timings, not e2e serve).
- **Now:** serve pump drives two clients (triplet mutex, CV backoff, workspace reserve, default concurrency 8), Engine-owned forward (embed, ragged attn, EXL3 linears, decode-rows sampler), per-seq tables/GDN, think/content split, real TTFT, working CLI. Two-curl T=0 Paris/Rome is the frozen gate. Graphs reverted (stale replay); MTP-in-pump blocked on bind + program-list + serve state (engine ticket running).

## Status

- **Early history (collapsed):** first pump tokens streamed multilingual
  soup with cross-run divergence (block_tables/GDN binding + fused-scatter
  + conv-transpose + q/gate-interleave defects, fixed in order). Details
  are run-record below; the freeze is the truth.
- **Two-curl freeze PASSED (2026-09-23, lead-run on the same binary):**
  solo-A == conc-A (`'____'`, 16/16 tokens), conc-B 16 tokens,
  `output 16` on every req. Root cause was the EXL3 fused-QKV
  scatter (column-major temp copied as row-major; exact only at T==1)
  — fixed in `text.cpp` (+43/-1, column-wise 2D D2D). Determinism
  gate closed; fluency (flat-distribution soup) stays open.
- **Conv transpose fix (2026-09-23):** loader copied conv1d weights
  flat while kernels index tap-major — mixer then oracle-exact
  (15.69 vs 15.65). The remaining global gap was found by the
  layer-walk: per-head q/gate interleave (next entry), not the
  conv path.
- **COHERENT (2026-09-23, lead-run):** per-head q/gate interleave fix
  (`q_proj` 12288 rows are `[q-head; gate-head]` per 512-group, not
  halves) — bread NLL 15.37 → 1.21, 5-tok 3.57 (oracle 3.45).
  Two concurrent clients: **Paris / Rome**, output 24/24. Serve works.
  (Think/content split + real TTFT landed in `66ae8c7`; the nits noted
  at freeze time are fixed.)
- **FROZEN binary (2026-09-23):** `apps/ninfer-serve`, 216731704 B,
  sha256 `2746a95d…94aa17f46b`, env `CUDA_EXL3_AUTOTUNE=0
  CUDA_EXL3_SPLIT_TARGET=0`. Graphs reverted (stale replay — ticket 2
  needs the capture-stream fix first); MTP-in-pump blocked on the
  engine ticket (Work queue §5); fused attention queued behind those.

## Work queue (point of truth for /goal loops)

Status words: DONE / RUNNING / QUEUED / BLOCKED. Update this table as
work lands; the loop reads here, not chat.

| # | Item | Status | Mode | Gate / exit |
|---|---|---|---|---|
| 0 | Freeze: coherent two-curl T=0 (Paris/Rome) | DONE | - | 66ae8c7; binary sha 2746a95d |
| 1 | Think/content split | DONE | swarm | Test green, wire clean |
| 2 | Real TTFT | DONE | swarm | Build + timestamp path |
| 3 | CLI deadlock | DONE | swarm | CLI prints 24 toks, exit 0 |
| 4 | Graph capture single-seq | REVERTED | swarm | Replay ids == eager ids (stale, reverted cleanly) |
| 5 | MTP-in-pump (window 3, decode-only) | DONE on landed C:/src binary | Lead-verified: spec-off frozen gate holds (solo + conc Paris/Rome, 24/24, rewind True); spec-on accept 3 / commit 5 (Paris F=19 drafts=[6511 314 9338] verify=[6511 314 9338 369]); Paris/Rome rewind True; cross-path first-24 exact prefix both prompts. Re-score at landing: 03 groups > 0 (bind 979/81 fused), 04 fatal = false, 05 hard_off = false, 02 decode_only = true / mixed = off / window = 3, 09 accept > 1 (3), 14 Paris/Rome good. Known limits: serve ignores EOS (both paths run to max_tokens); final-step overshoot up to +4 over max_tokens; spec-on ~45-55 ms/tok wall vs ~25 ms/tok spec-off (syncs dominate, follow-up). Eager coherence re-proven: conc==solo 6/6 at 16 toks |
| 5b | MTP engine work (workdir, UNCOMMITTED) | DONE landed 2026-09-23 — root causes: (1) stale host readback (non-blocking compute stream, blocking memcpy, no sync → nondeterministic first bonus), fixed with device_.synchronize() before each host read; (2) verify window off-by-one (drafts verified one slot early), fixed with bonus-chain inputs [b@F,d0@F+1,d1@F+2,d2@F+3] + commit-list (len 2-5); (3) MTP qkv split EXONERATED. Workdir proof: accept 3/commit 5, ~96-104% accept, Paris/Rome rewind True, cross-path prefix exact, pre-EOS ids == CLI 9/9 | Landed into C:/src (this tree) and committed 7c3fedb: `src/runtime/engine/engine.cpp` (serve-MTP state machine), `src/runtime/engine/exl3_program.cpp` (S7 allow-flag + weights.mtp bind), `src/models/qwen3_5/execution/parameters.cpp` (exl3_fused MTP qkv), `src/models/qwen3_5/frontend/frontend.{h,cpp}` (decode_tokens). All bars re-proven on the landed binary — see §5 |
| 6 | Baseline ms/token (spec off, ctx ~2k) | DONE 2026-09-23 on landed binary | ctx 2,265-token prompt (server-counted), spec-off, T=0: wall 1.5 s / 24 tok, TTFT 914 ms → decode ~24 ms/tok; prefill ~2.48k tok/s; Paris answer correct. Matches ~30 ms/tok contract. 2026-09-23 re-measure on parked binary (graphs default-off): 2,305-tok prompt, 64-tok out ×3 runs — wall 2.36–2.41 s, server TTFT 751–758 ms steady (~3.05k tok/s prefill) → decode ≈ (2400−755)/64 ≈ 25.7 ms/tok. This is the number a future graph probe has to beat. Step-2a re-measure on ship binary 8cc7b792 (=cc65999, EOS-stop live, spec-off T=0): p64 (97-tok prompt, 96 out, ran to budget) TTFT 370 ms wall 2,989 ms → decode (2989−370)/96 ≈ 27.3 ms/tok; p2k (2,029-tok prompt, 96 out) TTFT 699 ms wall 3,217 ms → decode (3217−699)/96 ≈ 26.2 ms/tok, prefill ≈ 2.9k tok/s; ctx2k-factual (2,265 toks) TTFT ~806–951 ms (prefill ~2.6k tok/s) but EOS fires at 7 toks (prompt-dependent, not a rate probe). 2-client aggregate (2×64-tok streams, conc-8 server): makespan 2,122 ms / 128 toks = 60.3 tok/s (≈2× single-stream; ~33 ms/tok per stream under contention vs ~26–27 solo). Launches/step (static, single-seq decode): attn 64 (1/layer; per-row loop causal_softmax_attention.cpp:594 `for s < num_seqs` — the fusion target); EXL3 ≈ 370/step static-corrected (see lever ranking below; nsys confirm deferred). CORRECTED profile 2026-09-23 (checkpoint index, no GPU): 64 layers = 48 GDN + 16 full-attn (every 4th), ALL with SwiGLU mlp (intermediate 17408) — mlp was missing from the old 256–320 count. Launches/step ≈ 48×6 + 16×5 + lm_head ≈ 370 GEMMs + 16 attn. Weight-traffic ranking per decode step (3.5bpw, m=1, bytes ≈ time): mlp gate+up+down 3×(5120×17408)×64 ≈ 7.5GB ≈ 61% ← the lever; GDN in_proj pair + out_proj ×48 ≈ 2.4GB ≈ 20%; full-attn qkv+o ×16 ≈ 1GB ≈ 9%; lm_head (5120×248320) ≈ 0.56GB ≈ 5% in ONE launch (best occupancy, lowest priority); norms/conv negligible. SUPERSEDED 2026-09-24 by nsys Phase 1 + code read: gate/up is ALREADY one fused linear_swiglu dispatch (no unfused pair left to fuse — do not propose it again); EXL3 ≈87% of GPU time, flat across gemv_plain b3/b4 (decode m=1, ~36%) + gemm_m b3/b4 (prefill/batch, ~47%) + lm_head K6 (~5%); see row 7 gap note. Attn fusion correctly parked (16 launches/step are noise against 370). Known wart (pre-existing, out of scope): HTTP maps FinishReason::None → "stop", so budget-exhaustion also reads "stop" at the API; server log distinguishes ("stop token" vs "none"), and the outcome enum now records StopToken correctly |
| 7 | Graph ticket 2 (serve-path capture wiring) | BLOCKED — seam landed inert, parked default-off | 2026-09-23: `v3_capturing(cudaStream_t)` fixed (exl3_gemm.cu:709,730,875) + engine capture/replay seam (`ServeDecodeGraph`/`step_decode_layers` engine.cpp; `embed_serve_input`/`forward_serve_decode_layers` text.h/cpp; uploads/embed/sample outside capture; gate M==1 + decode-2+ + lane-forget on admission; NINFER_SERVE_GRAPH=1\|verbose\|dry to re-enable). Finding: capture records correctly (replays bit-exact, frozen 6511/314/9338/369 chain) and dry-run (same code, no capture) is correct — but the capture-step eager execution leaves hidden bit-identical to prev step (dup token), so checkpoint 5 fails. Next probe (separate go): ThreadLocal mode / Nsight. 8/8 spec-off matrix re-proven green on the parked binary. GAP NOTE 2026-09-24 (nsys Phase 1 + roofline — the gap is runtime, not the checkpoint): roofline 15GB/1.8TB/s ≈ 8ms/tok ≈ 120 tok/s; ExLlamaV3 lands ~100–120 tok/s plain, ~150–180 with MTP/DFlash/graphs on this class. We run ~30 tok/s (~26 ms/tok) because decode is m=1 gemv ×4 linears ×64 layers + had in/out, hundreds of launches/token, no graph — nsys: gemv_plain b3/b4 ~36% + gemm_m b3/b4 ~47% + head K6 ~5% = ~87% EXL3. NOT codebook/bpw (same file does ~4× in Tabby/ExLlama), NOT gate/up fusion (already fused), NOT lm_head (5%). Close order: (1) capture writes hidden + graph the decode layer loop (the ExLlama-shaped win); (2) one-layer ns vs LinearEXL3.forward (occupancy/cfg); (3) spec faster than eager only after (1). Blocker: spec-on + default graphs FATALs at startup (GraphExecUpdateFailure result 5) — eager spec-on needs --no-cuda-graph; graph-on-target fixes that first. FIX 2026-09-24 (checkpoint 5 CLEARED): root cause is a PLATFORM quirk, not our kernels — capmini-proven on this WSL box (Global AND ThreadLocal): capture records but never executes eagerly, replay is the only working executor. Fix (engine.cpp capture branch, host-only): capture step replays immediately for its outputs (+1 launch per lane admission). Proven: capture in=760→out=6511 fresh logits, full Paris chain no dups, 8/8 conc==solo green with graphs on (NINFER_SERVE_GRAPH=verbose). ThreadLocal prong closed (fails identically). Still parked default-off. SPEC-ON FATAL TICKET CLOSED 2026-09-24: MTP + default graphs no longer FATALs at startup — program_impl forces use_cuda_graph off for SpeculativeBackend::Mtp (stderr notice, eager path identical to proven --no-cuda-graph). Bar: spec-on starts clean + Paris correct (979/81 MTP weights, TTFT 884ms); spec-off GRAPH=1 13.6 ms/tok holds; 8/8 green. --no-cuda-graph flag still accepted (no-op for MTP). VER/M=4 GRAPH TICKET CLOSED 2026-09-24 AS INVALID (no GPU spent): one frozen ver/M=4 exec cannot span F — verify passes venv={1,F+4} and F bakes into full-attn three ways: route (Prompt/SmallT/ChunkedSmallT flips with max_visible_keys), grid (small_t splits/target_ctas/page_limit derive from it), window (logical_capacity/implementation_window + CTA window cull attend wrong KV = wrong tokens). Pinning the envelope is out (load-bearing: tight envelopes keep drafts off unwritten columns). Per-F cache rejected (an exec per token of context is a leak, not a graph strategy). MTP stays eager per bar-4. Later sliver only: dec/M=1 under MTP (bonus/single target token; no F+4 bake) — open only if a profile shows the bonus stack matters. |
| 8 | Fused single-M attention | BLOCKED behind 7 | Gate replay == eager unmet; per Known-skips §6 stays behind flag until bit-exact. maxAbs gate + no regress. Step-2b verdict (static, no GPU spent): the cheap version does not exist — ragged entry is a per-row loop (causal_softmax_attention.cpp:594 `for s < num_seqs`, one single-row kernel per row; small_t grids bind one table row/launch) and the fused on-device-resolve kernel from the comment was never written (`DeviceResolveRaggedToken` has zero callers). One-launch/layer needs a NEW kernel (deferred by the ticket itself), so: default stays per-row, no flag, no kernel. Revisit only with a new-kernel work order behind its own GPU go. 2b-FOLLOWUP (binary 9cd510b5, uncommitted): built the fused pure-decode path after all — gather span-1 rows → compact [D,H,1,B] → proven singular MultiBatch entry once → scatter (4 launches/layer for any B>1; B=1/mixed stay per-row), flag NINFER_FUSED_ATTN=1 default-off. Quality held: 8/8 green flag-off AND flag-on (conc==solo, token-identical both flags), mixed prefill+decode coherent, spec-on frozen F=19 accept 3 + EOS-through-commits + accept-2 rewind on the 2b binary. Speed did NOT move: 2-client 60.3 → 56.3 tok/s (B=2 saves nothing: 4 vs 4 launches/layer), 4-client 96.4 → 102.2 (+6% single-sample, noise — decode is EXL3-math-bound, not launch-bound). Per ticket: default stays per-row, flag parked opt-in; code rides in tree default-off, uncommitted |
| 9 | KVarN / DFlash2 | BLOCKED — needs standalone work orders | 2026-09-23 assessment: HyperQwen audit stands (zero `.cu`/`.cuh`/`.cpp`/`.rs` repo-wide, vLLM-0.28 Python/Triton only — nothing to port as C++/CUDA); KVarN is a backend-level KV-format change, DFlash2 needs a second draft model + degenerate-distribution guard. Neither is a serve tweak — no code started, none exists to land |
| 10 | Follow-up commit (lanes 1-3 + graph revert) | DONE | - | 66ae8c7 already holds 1-3; revert is workdir-clean |
| 11 | No-action audit (criterion 4) | DONE 2026-09-23 | Closing sweep: my changes scoped to 7 files (content diff 797+/71- CR-insensitive; tree-wide CRLF-vs-HEAD churn is pre-existing, untouched); no servers running, GPU free (1.2/32.6 GB); queue + Major-issues mirror consistent; frozen Paris/Rome coherence re-proven on every binary built this run. No further action outside the BLOCKED rows |
| 12 | Formatter/stop: serve ignores EOS + final-step overshoot | DONE 8cc7b792 | serial, never mixed into graph work | Step-1 EOS stop landed (serve+scheduler only; engine/text/GEMM untouched): stop ids ride batch::Request (model defaults via Engine::default_stop_token_ids + caller stops), scheduler finishes on a hit, pump drops the stop token + same-step tail and reports StopToken. Proven: Paris24 '**Paris**.' 8-10 toks stop (was 16-24 + drift), Rome '**Rome** (Italian: *Roma*).' 15 toks stop; 8/8 conc==solo; decode ~25 ms/tok unchanged (wall down = fewer steps); spec-on frozen F=19 accept 3, EOS drafted+accepted (F=24/32 commits incl 248046) then emit stops, partial-accept rewind exercised (accept 2). |
| 13 | Async readback for spec-on (8 → 5 host stalls) | DONE 2026-09-23 | 4 verify argmaxes batched behind one sync via dedicated `mtp_vtok_` outbox (`engine.cpp` layout + verify loop); bonus + 3 draft-chain syncs stay (true AR deps — no event scheme removes them). Bars held: accept 3/commit 5 identical, rewind True, cross-path True. Wall 2.52s/33tok (~46ms/tok) vs 2.54–2.89s pre-change — marginal: step is kernel-bound (12–13 row-forwards per ~4 tokens ≈ 3× eager work), not sync-bound. Recorded, not solved; remaining lever is structural (fewer rows), not readback |
| 14 | Ship gate: think-split in same tree as MTP binary | DONE ca8a26ed (uncommitted) | serial, before any ship claim | One binary, all re-proven 2026-09-24: spec-off 8/8 green + SSE clean; spec-on frozen Paris `**Paris**.` / Rome `*Roma*).` stop, F=19 accept 3, EOS-through-commits, accept-2 rewind; think-split LIVE (bat-and-ball, thinking default): reasoning_content=777 chars, content clean. Protocol lesson: frozen gates require enable_thinking:False — thinking-default wanders (F=59, different text, still coherent). |
| 15 | Lane B: batched MTP verify (width-4) | DONE 2026-09-23, lead-verified | 4 serial width-1 verify rows → one `target_verify_batch` (bonus+3 drafts, single sync); draft AR + accept + commit untouched; recurrent kernel stays width-1 via spare↔shadow ping-pong (no new math). Lead re-ran bars on landed binary: `drafts=[6511 314 9338] verify=[6511 314 9338 369] accepted=3 commit=5` identical, rewind True, spec-off 8/8 green. Wall: lane-measured spec-on 3.17→2.62s single-run noisy (~17%); my TTFTs 723–884ms vs ~1s pre-change, directionally consistent. Spec-on still ~3× spec-off, kernel-bound — recorded, not solved |

Rules: swarm only items marked swarm with disjoint files; everything
serial runs one lane at a time on the GPU. No item starts outside the
Known-skips list. Commit only on explicit go-ahead.

Side ledger (not queue work, uncommitted): `docs/cpp-optimization-and-bug-audit.md`
— read-only audit, 8 optimizations + 5 suspected bugs with tests, nothing executed.
`docs/optimize.md` — read-only pass 2, 8 consolidation targets + 3 new speed notes, nothing executed.

## Known skips (intentional)

| # | What | Why skipped | When | Accept bar |
|---|---|---|---|---|
| 1 | `CUDA_EXL3_AUTOTUNE=0` required at serve | Tuner-in-capture aborts; heuristic tier is the contract | Revisit only with a capture stream `v3_capturing()` actually sees | Serve starts clean without the env pin |
| 2 | Default context / KV = 8192 | Paris/Rome profile | 262k is a planner + compressed-KV project, not a flag | - |
| 3 | `max_concurrency` / `kMaximumConcurrency` = 8 | Max seqs/rows, not max tokens; chunk (1024) sets prefill M | - | - |
| 4 | MTP-in-pump | BLOCKED: spec backend startup-fatal + mtp weights unbound + no serve MTP state (see queue §5) | Engine ticket first (bind mtp, serve state), then pump branch | T=0 Paris/Rome unchanged, tok/s up, head launches/token down |
| 5 | CUDA graphs, single-seq decode | Stale-replay reverted | Gather/upload/sample-D2H outside graph; `v3_capturing()` on capture stream | Replay ids == eager ids; multi-seq is a separate ticket |
| 6 | Fused single-M attention | 64xN launches correct today | One launch/layer, CTA per row; per-row path behind flag until bit-exact | maxAbs gate + no Paris/Rome regress |
| 7 | KVarN (or fp8 KV) | New cache + attn; needed for 262k on one 5090 | Separate work order, not a serve tweak | - |
| 8 | DFlash2 | Second draft model + verify | Only after MTP-in-pump is real | - |
| 9 | Prefix page-share in admit | Digest + refcount + CoW | After two-curl is boringly stable | - |
| 10 | int8-QK prefill, sort-free sampler | - | Only if a profile says prefill/sample is the wall | - |
| 11 | HostKVExtentStore removal / CLI-legacy purge | Hygiene | After CLI Paris stays green | - |
| 12 | Hybrid Hadamard in vendored `exl3_had.cuh` | Option D | Revert to pristine, rerun P16/MG/OP; fork epilogue only if a gate fails | - |
| 13 | malaiwah K5K6-context / Gilded Gnosis 262k recipe | Another checkpoint + vLLM runtime | Loader increment at most; do not import their Python | - |
| 14 | Vision / 8MP | Out of scope for text serve | - | - |
| 15 | Interleave q/gate regression test | Missing in-tree | Add when `attn_input_proj` is touched again | One unit on q/gate scramble |
| 16 | GDN snapshot/restore for graphed mixers (ORT pattern) | Not implemented; nothing graphed touches recurrent state today | Only with an attention-backend project that graphs mixers (route by width/batch, grid by capacity, mask by position) | GDN slot addresses stable across replay; rewind == eager |

Done, not skips: EXL3 load, pump, one scheduler, real `run_batch_step`,
per-seq tables/GDN, think/content split, TTFT, CLI Paris. Nothing outside
this list gets started "because HyperQwen had it."

## Provenance ledger — everything imported, ported, or added

Per-source files (borrowed + not-borrowed + why) live in
[`docs/provenance/`](docs/provenance/): `cuda-exl3.md`, `exllamav3.md`,
`buun-llama-cpp.md`, `hyperqwen.md`, `base-and-checkpoint.md`.
Summary:

| # | Item | Source / version | Author / license | Why used | Expectation |
|---|---|---|---|---|---|
| 1 | NInfer base engine | github.com/Neroued/ninfer | Neroued | The native C++/CUDA engine we fork: CLI, serve skeleton, MTP/draft paths, graph machinery | Upstream concepts only; execution path is ours |
| 2 | Cinference fork changes | this repo history (`b74044f` MTP-10 publish) | satellitedown | HISTORICAL fork state: MTP-10 window, capture-based graph reuse, enlarged CPU/GPU round buffers, Huihui NVFP4 setup | Our serve narrowed to decode-only window 3, graphs off; fork flags are not our config |
| 3 | cuda-exl3 v3 kernels (vehicle) | github.com/Zeuss5/cuda-exl3 @ `6a1ffc3` | cuda-exl3 contributors, MIT | Single M-tiled fused EXL3 GEMM + per-shape autotuner + epilogue Hadamard; 6 files vendored under `src/ops/linear/exl3/torchexl3/`, sha-pinned (`VENDORED_SHA256.txt`), ATen host layer replaced with raw-pointer seam, device code byte-identical except the documented hybrid delta in `exl3_had.cuh` | Best available EXL3 GEMM for sm_120; 8-26x over SIMT prefill; tuner intact (verified, not assumed) |
| 4 | exllamav3 1.5.1 (reference/oracle) | pip install [exllamav3](https://github.com/turboderp/exllamav3) 1.5.1 (reference tree, not linked) | turboderp (ExLlamaV3) | Trellis/codebook/dequant ground truth; `LinearEXL3.forward`, `reconstruct_slice`, `had_r_128` grid convention, QTIP gemv lineage; every numeric gate oracles against it | Reference only — never linked, never shipped; gates stay green against it |
| 5 | buun-llama-cpp format docs | [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp.git) @ `a2fd78181` (docs only) | buun contributors | `llama-hdf5.h`, `ggml-cuda/exl3*.cuh`, format docs for the trellis layout cross-check | Docs only; layout [k/16,n/16,16K] confirmed independently on all 409 groups |
| 6 | HyperQwen methodology | github.com/rktchip/HyperQwen @ `fe09615` | rktchip | vLLM-0.28 fork (Python/Triton, zero portable C++/CUDA — verified hard negative): contributed the *designs* — continuous batching mechanism, DFlash2 draft algorithm, KVarN Hadamard-KV concept, draft-vocab coverage method, GPTQ-lm_head data point, MTP-finetune NEGATIVE (do not burn 6h GPU; ship checkpoint MTP as-is) | Concepts, not code; kvarn/DFlash2 ports are algorithm work scoped post-two-curl |
| 7 | Checkpoint `Qwen3.8-27B-EXL3-3.5bpw` | EXL3 checkpoint dir (safetensors + `quantization_config.json` + `config.json`/`tokenizer.json`; `library_name: exllamav3`, base `Qwen/Qwen3.8-27B`) | Base model Alibaba Qwen; local EXL3 quant via exllamav3 | The serving target: 409 trellis groups (270 K4 / 137 K3 / 1 K5 o_proj L63 / 1 K6 lm_head), all mul1-only, fused qkv (n=10240), ships its own MTP draft head | Every group serves (K5/K6 via v3, Gate K5K6); file is clean (0 non-finite scales, corrected recount) |
| 8 | Own: batch pager + serve pump | `src/batch/`, `src/serve/generation_service.cpp` | this tree | `PagedKvPool` (16-token logical pages) + device gather + triplet-mutex pump + decode-rows sampler: the multi-user substrate | Two-curl gate; pager wins over the legacy host arena (prefix-sharing to be ported into admit) |
| 9 | Own: hybrid Hadamard guard | 10 sites in vendored `exl3_had.cuh` (pin-recorded) | this tree (criterion-7 swap) | In-range fp16 product verbatim (bit-exact), exact fp32 fallback only on overflow-with-finite-scale; insurance against huge-activation overflow in fp16 Hadamard | No-op proven (P16 still bit-exact); post-two-curl slot decides revert-vs-fork (see §Hybrid note) |
| 10 | Deliberately NOT taken | cuda-exl3 `bindings.cpp`/`mla_decode.cu`/Python pkg/bench numbers; HyperQwen Python; kvarn glue | — | No libtorch in server; no MLA model; their numbers are not our numbers | Revisit MLA only for a DeepSeek-shaped model |

### Hybrid note (post-two-curl slot)

The hybrid is not a wrapper: its 10 sites span 5 warp functions feeding both the had_in kernels *and* the GEMM epilogue (`had128_warp_acc/out` from `exl3_gemm.cu:465-510`). Making the vendored file byte-identical while keeping the behavior therefore means forking the epilogue kernels too — or reverting to pristine (`dfa6f331…`, recoverable in one `cp`) and re-running P16/MG/OP: if green, the hybrid was dead insurance on this checkpoint and deletion is pure win. Pin file stays the drift-detection contract either way.

## What changed (upstream base — HISTORICAL fork state, not our serve config)

> The fork shipped MTP-10 + capture graphs + big round buffers for the
> NVFP4 path. Our EXL3 serve uses decode-only window 3 (blocked, engine
> ticket running), graphs off (reverted), chunk 1024 / cap 8. Do not read
> fork flags as current.

- **MTP-10 decoding:** draft window raised from 5 to 10 tokens.
- **Capture-based CUDA Graph reuse:** graph matching by captured node types and kernel functions; matching profiles + batch sizes share one executable.
- **Expanded CPU/GPU round handling:** enlarged draft/position buffers, updated validation, so MTP-10 reaches the decode path.
- **Ready-to-run Huihui setup:** published NVFP4 v3 model + one-menu installer.

## Enhancements over the base (this port)

> **Archive note:** the `batch.cu` / 4-descriptor / 64-wide / DFlash2
> material below is the staged HyperQwen-shaped design essay. The live
> serve path is: `ServeHookLoop` pump + one `RequestScheduler` +
> Engine-owned `run_batch_step` (embed, ragged attn, EXL3 linears,
> decode-rows sampler), per-seq tables/GDN, cap **8 seqs**. Serve
> `Engine::submit` throws; empty decode never finishes a request.

### Multi-user continuous batching (`src/batch/`, staged)

Why: upstream decode path is single-request. Serving many concurrent users needs
iteration-level mixed scheduling inside the CUDA-graph engine.

- `batch.cu` — interleaved device step: up to 4 batch descriptors per step
  (decode / prefill chunk / prefill recurrent / fused A), zero-copy chained kernels.
  This is the HyperQwen mechanism that yields 64-wide throughput.
  (Archived design claim — never measured on this engine; our serve
  runs cap 8, ~30 ms/token solo.)
- `paged_kv.cu` — 1-block (64-token) granular paged KV allocation. Less fragmentation
  at dense concurrency, +6.25% worst-case slack vs 4-block granularity.
- `request.cc` — DFlash2 draft state: recompute the draft path after accept, treat as
  a 2-per-step cost (no idle draft head); cf HyperQwen measurement.
- `cinference_hooks.cc` — exact 4-break state handoff to upstream executors: block-id
  clamps, verify window, GDN fold/release states, layer flags.
- `scheduler.h/.cc` — `decode_signature()` graph-replay gate (FNV-1a over decode seq ids)
  and `need_host_tables` opt-in for the host block-table matrix.

Linux-gated rest: real scheduler consumption loop, 4:1 KV quant wiring (`publish`),
64-wide soak, in-tree dev configure. All of the above compiled + `ctest` green
(`steps=8 long_slices=6`) on Windows before handoff.

Settings: `src/batch/README.recommendations.md`.

### EXL3 weight support (trellis decode + serve) — SERVING (frozen 2026-09-23)

> Supersedes the stale lines inside this section: **409 groups**
> (weight_map is the universe), **0 non-finite** scales file-wide,
> **v3 + legacy-gemv routing** (m=1 gemv, m>=2 v3, K5/K6 via v3),
> construct + two-curl coherent. The gate diary below is run-record;
> where it says "in progress / gemv-only / 401 / NaN checkpoint", the
> freeze above wins.

Goal: load and serve EXL3 checkpoints (Qwen3.8-27B-EXL3-3.5bpw) natively — DONE (frozen serve). MTP draft serving is BLOCKED (weights.mtp unbound, spec startup-fatal, pump never calls MTP); DFlash2 is a separate queued work order, not a live path. Port sources: the
[exllamav3](https://github.com/turboderp/exllamav3) 1.5.1 tree and
[buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp.git)
(`ggml/src/llama/llama-hdf5.h`, `ggml-cuda/exl3*.cuh`, format docs).

Contract (verified against the weights + 1.5.1 loader):
- safetensors tensors per linear: `trellis` int16 [n/16, k/16, 16K], optional
  `mul1`/`mcg`, plus fp16 `suh`[k] / `svh`[n] (pre-baked input/output scales).
- Tile order: k-words fastest; word count per tile = 16K, i.e. TWORDS = 8K u32.
- Bitrate K is per-tensor float; odd .5 rates use the mul1 codebook (cb2).
- Hadamard 128 on both sides, fp16 round: suh preyscale, then H128(1/sqrt128);
  svh postscale after the out Hadamard. r_scale = 0.088388347648.
- Decode (m<=8): 1.5.1 QTIP gemv (m16n8k16 MMA, fp16 accum), plain launches —
  Hadamard stage as its own kernel, no cooperative barrier (graph-safe).
- Prefill: reconstruct tiles to fp16 weights, then cinference's own bf16 GEMM.

Status (2026-09-21): pack layer + decode gemv + launch pipeline vendored
verbatim from 1.5.1 in
`src/ops/linear/exl3/` (MIT, header-only, no libtorch): `exl3_ptx_shim.cuh`,
`exl3_codebook.cuh`, `exl3_dq.cuh`, `exl3_hadamard.cuh`, `exl3_gemv_ns.cuh`
(mma + register decode extractors), full reference `exl3_gemv_kernel.cuh`
(cooperative) and the graph-safe plain carve `exl3_gemv_plain.cuh`
(hadamard in/out stages externalized as their own plain launches, no
cooperative barrier), and the plain launch pipeline `exl3_launcher.{h,cu}`
(plan/cfg heuristic from 1.5.1 + had_in -> gemv -> had_out -> bf16 cast,
12 instances bits{3,4} x cb{0,1} x mmode{0,1} x cfg{0,1}). All gated at real
launch instantiation (single/multi rows, both launch geometries, smem-stage
on/off) — 0 errors on sm_120a (nvcc 13.3, MSVC 19.44).

Checkpoint census (validated from `quantization_config.json`, which holds
707 logical tensors under `tensor_storage`), the exl3 groups: K=3 x137,
K=4 x262, K=5 x1 (o_proj layer 63), K=6 x1 (`lm_head`) — no half-integer
tile. The 1.5.1 gemv cb assert caps the template at bits 3/4, so
the 2 K>=5 tensors MUST be routed via reconstruct/dequant and never via
gemv. All exl3 linears use the cb=2 (mul1) codebook — all 401 have
`mul1` (checked mcg absent as well). Note: the earlier "K=4 all calibrate to cb1" line was wrong —
in 1.5.1 the cb rule is `mcg → cb1, mul1 → cb2`, not presence of mul1;
this checkpoint is mul1-only, cb0/cb1 unused.

CORRECTION (2026-09-22, wave-2 S3c): the census above is 401-based and
stale twice over -- (a) the file holds **409** trellis groups, not 401:
`quantization_config.tensor_storage` (707 entries) silently omits the 8
`mtp.*` groups while the safetensors `weight_map` holds all 409, so
**weight_map is the group universe, tensor_storage is not** (the loader
harvests mtp members from weight_map + shard headers); (b) K5/K6 serve
via the v3 row, not reconstruct/dequant (Gate K5K6). Retained as
run-record.

Remaining (staged next): `ops/linear` dispatch for a new QType (core Weight
seam + decode route into exl3_linear_decode); prefill reconstruct + bf16
GEMM for m>8 and the 2 K>=5 tensors (reconstruct tiles to fp16 then the
tree's bf16 hgemm); DFlash2 draft tensors through the same materializer.
Numeric verification (forward pass vs 1.5.1 oracle) is a Linux-side step.

EXL3 safetensors materializer (staged, `exl3_materialize.{h,cpp}`): parses
`quantization_config.json` (tensor_storage -> trellis/suh/svh/mul1), reads
flat + wrapped .safetensors headers + index.json, outputs per-group geometry
(suh[k], svh[n], trellis[k/16,n/16,16K], cb, byte offsets) with hard
invariant checks (16*K/dim/byte consistency + multi-file offset existence).
Live-verified with C++ probe P10: all 401 groups parsed, 5 structural
filters passed, 2 byte-range spot-loads match independent Python re-parse
byte-for-byte, k/n/u16 geometry triple confirmed.

EXL3 dispatch layer (staged, `exl3_dispatch.{h,cu}`): host-side linear op
`exl3_linear_op(A, out, scratch, Exl3Weight, stream)` + `exl3_plan_for` /
`exl3_build_plan` + `exl3_linear_workspace_bytes` + `exl3_gemv_eligible`,
with the `Exl3Weight` side-car (trellis/suh/svh + k/n/bits/mcg/mul1 +
Exl3GemvPlan). Carries pack pointers WITHOUT touching core Weight (surgical
rule). The launcher's `exl3_gemv_run`/`exl3_linear_decode` are `extern`
declared on exl3_launcher.h so the dispatch TU links them (the same seam
the future `QType::EXL3` case in `ops/linear/linear.cpp` uses); the binder
then fills Exl3Weight from the materializer's Exl3GroupLayout at
materialization time. Live-verified with C++ probe P11: cross-TU link
(dispatch + launcher), 12 production configs (bits {3,4} x cb2 x m {1,2,8}
x shapes {(128,128),(5120,8192)}) all 12 run clean on real GPU (had_in,
gemv, had_out, cast launches + sync), host plan logic asserted per run
(cb==2, mmode, cfg, block/cols/grid). P11 isolated small config (P11b) is
compute-sanitizer memcheck clean (0 errors); P11 full-suite sanitizer
noise is a sanitizer-tracking artifact (its in-program cudaMalloc/CHECK
frames fail under memcheck, not the kernel; the direct production run is
the authoritative gate).

- **Qwen3.8-27B-EXL3-3.5bpw ships non-finite scale vectors (2026-09-22):**
  SUPERSEDED -- see the P15 CORRECTION below (recount: 0/409; the census
  below read fp16 bits as int16 values). Retained as run-record:
  367/401 EXL3 groups have NaN (363 also Inf) entries inside their `suh`
  fp16 input-vector and/or `svh` fp16 output-vector. Both 1.5.1's own
  `LinearEXL3.forward` and the fused `reconstruct_had` kernel produce NaN
  for those groups — it is a quantizer/data defect, not an engine
  difference (verified by running 1.5.1 production on the same bytes). The
  port therefore carries `exl3_sanize_range` (in-place NaN/Inf->0 on the
  in-memory suh/svh copy at load) + `exl3_checkpoint_summary` (census of
  the defect, 401 groups, 367 suh-bad, 363 svh-bad on this checkpoint) so
  the binder can sanitize + surface the count. Zero is the principled
  replacement: each suh/svh entry scales one 128-wide Hadamard plane; a
  dead scale zeroes the whole plane's contribution, which is what a
  non-finite scale already does by destroying the plane.
- **Hadamard-plane grid convention trap (2026-09-22):** the vendored
  `had_hf_r_128_inner` reads its 128-entry scale window from
  `((half4*)scale)[blockIdx.y*32+t]`. If you launch a plane kernel with
  grid `(planes, rows)`, every row needs scale window `row*128 .. row*128+128`
  = planes 0..? — only the FIRST row gets its own window correctly; rows>0
  read plane 0's window (= the suh/svh of a DIFFERENT plane index) and all
  their outputs are slightly wrong (bit count: only ~6% exact). 1.5.1
  always launches `had_r_128` with grid `(rows, cols/128)` (row-major) so
  `blockIdx.y` = plane index. Any had-plane kernel in this tree MUST keep
  `gridDim.y` = plane index (see exl3_prefill_had_in/out and the launcher's
  exl3_had_in/out_plane). The m=1 decode path happens to have only one
  window anyway, which is why P11's green runs masked this until the m=16
  P12b bench exposed it.

CMake wiring: launcher.cu + dispatch.cu +
materialize.cpp into `src/ops/linear/exl3/sources.cmake` (incremental).

Gate P12/P12b/P12c numeric round (2026-09-22): the compile-green P11 runs were not numerically gated, and a full numeric bench against 1.5.1
production (P12 fused dequant, P12b prefill m=16, P12c decode m=1/2/8 vs
`LinearEXL3.forward`) caught two real faults: (1) exl3_had_in/out_plane
indexing uses gridIdx.x as the 128-col plane, so their scale window reads the
pre-scale at gridIdx.y*32+t, giving a WRONG window for every row (prefill
100% off; decode m<=8 only partially clean); fixed by switching to 1.5.1's
`had_r_128` geometry, grid (rows, k/128) with the plane as gridDim.y (rows,
k/128); (2) I was dequant+filling with the fused `reconstruct_had` into a
"K-wise OBA" while the prefill GEMM had the had on both A and C — double had
composition; the reference non-fused path for rows < 1024 is dequant_naked +
had_in (suh) -> gemm -> had_out (svh), which is now what exl3_prefill_op
implements (1.5.1's own `reconstruct_hgemm` at rows < 1024, via
`reconstruct_slice` per exl3.py, confirmed in source).

Gate P12 (dequant oracle, 2026-09-22): C++ `exl3_reconstruct_had` (fused, my
new vendored `exl3_reconstruct.cuh` from 1.5.1 `quant/reconstruct.cu`, K
1..8 x cb 0..2, bit-for-bit vendored) run on the real smallest group (k=256,
n=4608, K=3) vs 1.5.1 `exllamav3_ext.reconstruct_had_slice` -> byte-exact
1.5MB dump. Then the *checkpoint* defect became visible: 367/401 groups of
Qwen3.8-27B-EXL3-3.5bpw have NaN/Inf in suh (367) and/or svh (363) — the
quantizer evidently emitted non-finite scale rows; 1.5.1's own
`LinearEXL3.forward` also produces NaNs on those groups, so this is a data
defect, not an engine difference. P13 gates it: materializer now exposes
`exl3_checkpoint_summary` (counts of nonfinite suh/svh entries across the
whole index, 401 OK) and `exl3_sanize_range` (replaces NaN/Inf with 0 in an
in-memory copy — the recommended load-time fix; 0 bytes at checkpoint
regions, which is the well-defined cue in the quant; the port should apply
it at load and flag the count in the summary).

Gate P12b (prefill, 2026-09-22): full m=16 prefill of the smallest group
(k=256, n=4608) vs the 1.5.1 chain (dequant_naked reconstruct_slice +
had_r_128 x + hgemm_recon + had_r_128 y + bf16 cast, on the same bytes) —
maxAbs 0.0078, maxRel ~1.3%, sign-flips 2 (both at near-zero), NaN/Inf 0.
That is pure f16 vs hgemm accumulation-rounding difference (hgemm_recon
accumulates in f32 then rounds to f16, then cast bf16; my SIMT fp16 GEMM
accumulates f32, rounds f16, casts bf16) — inside the documented f16 GEMM
tolerance, not a fault. exl3_prefill.{h,cu} staged.

Gate P12c (decode, 2026-09-22): deterministically-tiled exl3_linear_op
(bits=4, cb2, m=1/2/8) vs 1.5.1 LinearEXL3.forward (same bytes via
reconstruct_hgemm -> gemv path) — maxAbs 0.5, maxRel 0.38%, 0 out-of-band.
Clean numeric confirmation of the now-fixed decode path (post-grid-convention
swap) — P11 confirmed the launch; P12c confirms the math.

Gate P14 (K>=5 reconstruct, 2026-09-22): the 2 reconstruct-only tensors
(o_proj L63 K=5, lm_head K=6) dequant via `exl3_reconstruct_rot` (naked fp16,
no H-fold) matches 1.5.1 `reconstruct_slice` **byte-for-byte** (K5 full 61 MB;
K6 row0 + row1280 full + 400/400 20x20 point-matrix). This proves the only
K-specific step (dequant) is exact; the SIMT GEMM and Had planes are K-agnostic.
Load-time `exl3_sanize_range` (zero NaN/Inf in suh/svh) fix-counts match an
independent Python census (o_proj suh fix=198, svh fix=155; lm_head suh fix=166,
svh fix=7278 — the loader-path counts are authoritative: this is the byte path
P14 dequant matched 1.5.1 byte-exact on; a naive quantization_config-offset
read of o_proj.suh counts 191 and is off by 7 entries — the per-file
safetensors header, not the config, is the source of truth), and every
untouched entry stays byte-identical.

Gate P15 (K>=5 serve chain, 2026-09-22): on o_proj L63 (K=5) my
`exl3_prefill_op` and the 1.5.1 reference **both** all-NaN, even after applying
the same sanitizer to the same bytes — **parity, not a port defect**. 1.5.1's
own `LinearEXL3.forward` (rt_151, raw group, no sanitize): o_proj K5
`nan=81920/81920`, lm_head K6 `nan=3973120/3973120`, and a defective K4 group
(layers.11 k_proj) `nan=16384/16384` too — the NaN chain is the **defect
signature itself**, not a K>=5-specific condition: all 367 defect groups
cannot be served from fp16 Hadamard by any implementation, upstream included
(a 6000x-scale controlled run through 1.5.1's had kernel stays finite, so the
kernel path is sound; the group's suh Had amplitudes reach ~6e7). Conclusion:
this checkpoint is **not fp16-servable as shipped** (367/401 groups) until
re-quantized; the port reproduces the reference exactly in every passable case
(P12/P12b/P12c byte/exact) and in the failing ones too (parity). The K5/K6
dequant step remains byte-exact (P14) — the loss is upstream of the engine.

CORRECTION (2026-09-22, supersedes the defect narrative above): a byte-exact
recount with the verified fp16-view method (cross-checked against torch
safetensors) finds **0 non-finite suh/svh entries in all 409 groups** -- the
file is clean. The 367/401 census read fp16 BITS as int16 VALUES (trellis
codes have high bits set and look NaN-as-fp16), and the P15 "parity-NaN"
runs used 6000x synthetic scales, not checkpoint amplitudes (real maxscale:
3.3 on K5 svh, 1.5 on K6). What survives: amplitude-overflow physics (huge
ACTIVATIONS can still overflow fp16 had -- the hybrid stays as insurance)
and the K5/K6 serving question, now answered below (Gate K5K6: they serve).
The sanitizer stays as a no-op safeguard for future dirty files.

Gate TU (compile, 2026-09-22): all 4 exl3 TUs in `sources.cmake`
(`exl3_launcher.cu`, `exl3_dispatch.cu`, `exl3_prefill.cu`,
`exl3_materialize.cpp`) compile clean under nvcc 13.3 / sm_120a /
MSVC 19.44 (`-std=c++20 --expt-relaxed-constexpr
-Xcompiler "/Zc:preprocessor"`; only the known benign 20012-D
`__device__`-on-defaulted-ctor warnings from the shim). This covers the
hybrid-scale `exl3_hadamard.cuh` edit (included by launcher + prefill):
compile-clean. Full-repo cmake configure + link remain Linux-gated
(WSL2 is not installed on this box; VS2022-bundled cmake exists at
`Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe` for
the day a Windows configure is attempted -- but the tree targets Linux,
so WSL2 install is the blocking user action).

Gate P16 (v3 row, 2026-09-22, open): the ATen-stripped cuda-exl3 TUs
(`torchexl3/exl3_gemm.cu` + `exl3_hadamard.cu`, 4 pristine headers
untouched) COMPILE + LINK clean via Phase-0 wrappers
(`exl3_v3_alias.h` global `half`/`bfloat16` + `<cuda_runtime.h>` first;
no vendored kernel body touched). Three first-build fixes, all in the
stripped seam (kernels byte-identical): `cudaStreamIsCapturing` arity ->
`v3_capturing()` helper + forward decl; `exl3_pick_split_row` de-inlined
(cross-TU symbol); probe `%`-on-float + missing `<cuda_runtime.h>`.
Smoke `exl3_p16.exe` (deterministic tiles, k=n=128): m=1/cb2 and m=2/cb1
runs OK, ahad + bf16 out fully finite at suh_scale=1. Bit-exact oracle
(`p16_oracle.py`: codebook graft + tile dequant + had + fp32 GEMM) and
the suh_scale~6000 K5-micro check are still open.

Gate P16 verdict (2026-09-22, CLOSED with one quirk): the v3 fused row
(`had_in` + M-tiled MMA GEMM + epilogue had + svh + bf16) reproduces the
1.5.1 chain (`had_r_128` + `reconstruct_slice` + `hgemm_recon` +
`had_r_128`, cross-checked against an independent numpy fp32 chain that
matches 1.5.1 bit-exact) as follows -- m=2: 0/256 bits differ; m=16:
0/2048, maxAbs 0.0; bits 3/4 x cb 1/2 all green at m>=2. Quirk: at m=1
only, a few outputs deviate (13/128 bits, maxAbs 0.25, up to 37 ulp on
small values; deterministic run-to-run; independent of split 1/4, BM
16/128, and autotune on/off). Serving consequence: route m=1
(single-token decode) through the verified legacy gemv path (P12c-green,
bit-exact at m=1/4/8) and use the v3 row for m>=2 prefill/concurrency --
the actual criterion-1 target. Root cause of the m=1 deviation is still
open (predicated-fetch path is the only m-dependent code, but row 0's
MMA is per-row independent -- not yet explained).

Probe bug found by the numbers (lesson): the P16 probe passed the RAW
input `A` to `exl3_gemm_row` instead of the Hadamard output `ahad`
(one-word fix). Symptom was `same structure, ~9334x scale` on a zero
trellis -- sum(x)=146.6 vs sum(xh)=0.0157 gave it away. The v3 kernel was
innocent; always verify which buffer the probe actually passes.

v3 mapping audit (all cleared): codebook hex constants IDENTICAL to
1.5.1; `dq_dispatch` is the same function with the same `lane*8`
convention (1.5.1's `reconstruct_kernel` calls it identically, then
shuffles to row-major); v3 feeds the fragments straight to
`mma.m16n8k16` and P16 proves that mapping correct. Note: upstream
1.5.1 `exl3_gemm` is a host dispatcher over successive GEMV launches,
not a fused kernel -- v3's MMA GEMM is from-scratch work, which is why
P16 (not upstream) is its reference.

Hybrid port into v3 (criterion 7, CLOSED): `torchexl3/exl3_had.cuh` now
has `had_hybrid_mul1/mul2` (in-range fp16 product verbatim, exact fp32
fallback only on overflow-with-finite-scale, non-finite scales pass
through for the sanitizer), applied at all 10 `__hmul2` scale sites
(input pre-scale x2, generic pre/post, split + non-split epilogues).
Unit gate 6/6 on-device (in-range, +/-60000 recovery, NaN/inf scale
passthrough, true-overflow stays inf). No-op proof: P16 m=2/m=16 still
bit-exact post-hybrid.

K5-micro outcome (honest bound): at suh_scale=6000, v3 and 1.5.1 agree
exactly (ahad 1 nonfinite each at index 0, out 128/128 NaN both) -- v3's
fp16 pre-scale overflows just like 1.5.1's, parity confirmed. The hybrid
fixes the pre-scale stage (unit-proven), but the DC Hadamard coefficient
(sum*rscale ~= 97000 > fp16 max at 6000x) still overflows the fp16 ahad
STORE -- no fp16-ahad pipeline can pass 6000x, and no ramp-input scale
isolates pre-scale overflow (any scale big enough to overflow products
also blows the DC store). Remaining K5 work is now precisely scoped:
rescale/DC handling for the real checkpoint amplitudes (or fp32 ahad),
not pre-scale. Real-K5-suh validation of the hybrid is still open
(requires checkpoint I/O through the materializer).

Gate MG -- multi-group row (2026-09-22, CLOSED bit-exact): the
previously declared-only `exl3_aln_rows_bf16` is now defined. Design is
v3's own: one `had_in` launch (shared x, stacked suh [groups,k]) then
ONE fused GEMM launch over all shards -- trellis/svh/C are addressed by
absolute column, only the activation slice differs per shard
(`ShardMap{nblk_end[], n_groups}` built from group_n widths). New raw
entry `exl3_gemm_rows` beside the single-shard rows; geometry enforced
in the seam (widths sum to n, each a 128-multiple, groups 1..8).
Gate (k=128, n=256, 2x{128} shards, per-shard 1.5.1 oracle): 4/2/2
0/512, 3/2/1 0/512, 4/16/2 0/4096 -- all bit-exact, maxAbs 0.0.
Fused m=2 row runs ~9.4us event-timed (shape-specific, bench pending).
Seam compile fixes (all in untracked `exl3_aln.*`, kernels untouched):
`exl3_v3_alias.h` is now included by `exl3_aln.h` itself (the header IS
v3-row interface, so the alias belongs there); v3 declarations moved
from an anonymous namespace to `namespace cuda_exl3` (they were
"referenced but not defined"); call sites qualified `cuda_exl3::`
(no ADL for these arg types). Repo TU `exl3_aln.cu` compiles clean
(`MG_COMPILE_OK` covers v3 TUs + mg probe + link + aln TU).

Gate OP -- QType::EXL3 routing (2026-09-22, CLOSED): `QType::EXL3 = 9`
added to `core/weight.h`; `Exl3Weight` carries binder-populated
`groups/group_n[8]`; new `exl3_op.{h,cu}` implements
`detail::exl3_dispatch` (binder contract: `w.payload` -> side-car) with
the verified routing -- groups>1: v3 multi; m==1: legacy gemv (exact
bf16->fp16 cast kernel in-scratch); m>=2: v3 single; K>=5/non-128/!ok
throw instead of serving NaN. `linear.cpp` dispatch + capacity cases and
`sources.cmake` (aln + op + 2 v3 wrapper TUs) wired. Routing gate
(exact-x tiles, per-route 1.5.1 oracle): R1 m=1 legacy 20/128 bits,
maxAbs 0.125 -- inside the P12c 0.5 envelope (gemv-vs-hgemm order, not a
routing fault); R2 m=2 v3 0/256; R3 multi 0/512. Full routing stack
(v3 + aln + op + legacy + arena) compiles+links (`OP_COMPILE_OK`).

Gate BENCH -- per-m latency on the 5090 (2026-09-22, serving shapes from
the checkpoint header; 2 runs, +/-15% clock variance, shape stable):
dense k=5120 n=6144 (in_proj_z class), fused k=5120 n=10240 g=3 (qkv
class), dispatch = true serving path incl. workspace + memset + cast.
m=1 legacy 38us; v3 24-61us flat across m=2..128; fused 27-94us for
1..128 tokens in ONE launch; old SIMT prefill 242/925/1601us at
m=16/64/128. The criterion-7 swap earns 8x/25x/26x at m=16/64/128.
Concurrency moral (kernel microbench, NOT e2e): 128 decode tokens batched = one 61us launch/layer,
~3ms full-model step -- batching is ~free up to m=128. Measured e2e
serve is ~30 ms/token solo spec-off; the K56 projection below is
scratch arithmetic from these per-m timings, not a measured number.

Gate BIND -- side-car builder (2026-09-22, CLOSED): new
`exl3_bind.{h,cu}` (`exl3_build_sidecar`: validate -> host-sanitize in
place -> upload -> `exl3_plan_for(m=1)`; `exl3_free_sidecar`; in
`sources.cmake`). Census decision: all 409 checkpoint groups are
uniform-K with a single codebook (270 K4 / 137 K3 / 1 K5 / 1 K6, every
group mul1-only), so EVERY group serves dense, groups=1 -- gate/up are
already separate groups (n=17408 each), qkv fused (n=10240) needs only
an activation-side output split, never a weight split. The MG row stays
staged+proven for future mixed checkpoints. Real-tile gate (torch-loaded
z + qkv + o_proj-K5, builder -> dispatch -> chain oracle): ok=1,
fix-counts agree with the independent oracle census, m=1/m=2/fused all
finite with maxAbs 0.01562 (sparse 1-ulp, medRel 0.0 -- the honest
MMA-vs-hgemm regime on real magnitudes; P16 synthetic bit-exactness was
luck of small values); K5/K6 refusal LIFTED (see Gate K5K6 -- builder marks
ok=1, dispatch serves K3..K6, legacy gemv stays K3/K4-only).
`formats.cpp` registers the "exl3" spelling both directions
(`FORMATS_COMPILE_OK` under cl/c++20).

Gate K5K6 -- K5/K6 serve at production amplitudes (2026-09-22, CLOSED):
real o_proj-K5 (k=6144 n=5120) + lm_head-K6 slice (n=128), ramp AND
randn s=2 inputs, 1.5.1 chain vs v3-hybrid row: oracle 0-NaN in all 8
cases (the old parity-NaN was 6000x synthetic scale, never checkpoint
amplitudes); v3-hybrid 0-NaN in all 8, maxAbs <=0.04 (m=1 quirk <=0.031
on K5 -- smaller than the K4 m=1 quirk bound). Through DISPATCH
(builder -> dispatch -> oracle): 4/4 finite, maxAbs <=0.0625 (carries
bf16-vs-fp16 input rounding). OP regression identical (20/128@0.125,
0/256, 0/512). Every group of this file now serves: K3/K4 all routes,
K5/K6 via v3.

Gate K56-BENCH -- per-m K5/K6 on the 5090 (2026-09-22, 2 runs,
<0.1% variance): o_proj-K5 (6144x5120) 23-59us flat m=1..128 (same
class as K4 dense -- o_proj costs no more than any other layer);
lm_head-K6 FULL width (5120x248320, 953MB trellis transient)
579/596/627/758/1844us at m=1/2/8/32/128. Full-width K6 m=2 is
BIT-EXACT vs the torch chain (0/496640). Serving story (SCRATCH
projection from the per-m microbench timings above, NOT measured e2e —
superseded by the measured ~30 ms/token solo spec-off in the Current
contract): single-user decode ~= 47x35us + 579us ~= 2.2ms/token (~=450 tok/s); a
128-wide batch ~= 47x61us + 1844us ~= 4.7ms for 128 tokens (~=27k tok/s).
lm_head is ~25% of every decode step -- the head, not the body, is
the first thing to quantize further or speculate past.

Criterion-5 status (2026-09-22): every new/edited TU compiles on
Windows (v3 + aln + op + bind + legacy + prefill under nvcc/sm_120a;
`linear.cpp` + `formats.cpp` under cl/c++20 -- `LINEAR_COMPILE_OK`,
`FORMATS_COMPILE_OK`). Full-tree configure is Linux-gated: it needs
pkg-config + FFmpeg dev + libcurl pkg-config, none present on this box
(WSL2 install pending).

## Interesting findings / gotchas

- **Hadamard rounding moments.** The reference applies suh/svh and r_scale inside
  fp16 in the exact steps shown above. Reorder or change rounding and you diverge
  from golden EXL3 outputs.
- **Non-toplevel trellis keys are full tensor names** (dots in the key); per-tensor
  K comes from `trellis.shape[-1]/16` in `quantization_config.json`.
- **cb is not a per-tensor byte** in this checkpoint family: K + mul1 presence fix it.
  K=3.5 -> cb2 + trellis; K=4 -> cb0/cb1 by calibration (unused split here: cb1).
- **Why our concurrency wins over upstream NInfer (HISTORICAL note — not
  this engine's numbers):** upstream hard-caps at
  `kMaximumConcurrency=8`, single staged prefill lane, MTP<=3, DFlash2<=15, fp16 KV.
  The 64-wide plateau / 1,039 tok/s at 64 on HyperQwen's 24 GB cohort is
  HyperQwen's vLLM measurement on different hardware and stack — it does
  not describe this C++/CUDA serve path. Our measured number is ~30
  ms/token solo spec-off (Current contract).
- **Any TU including exl3_dispatch.h must be .cu, not .cpp** (device
  intrinsics in the header chain; nvcc compiles .cpp as host-only).
  exl3_bind started life as .cpp and died at `__dp4a unknown`.
- **safetensors data_offsets are (start, end), not (start, length).**
- **Reading F16 safetensors as int16 gives BITS** -- `.astype(float32)`
  on the raw read yields garbage-thousands (an all-NaN oracle that sent
  us hunting before the `.view('<f2')` fix). The C++ side never trips
  this (bytes are bytes until the kernel interprets them).
- **Census correction: 409 groups, not 401.** The P10/P13 "401" set was
  layers + lm_head; the file holds 8 more under `mtp.` (fc + mlp
  down/gate/up + attn q/k/v/o). All 8 mtp groups are CLEAN (0 defects) --
  in fact the full-file recount finds 0 non-finite suh/svh in ALL 409
  groups (the 367 number read fp16 bits as int16 values; see the P15
  CORRECTION). The model is a
  hybrid: interleaved `linear_attn` (gated-delta, fused qkv n=10240) and
  `self_attn` (GQA with separate q n=12288 / k,v n=1024 / o) layers, plus
  an MTP draft block. Drafting relevance (CORRECTED 2026-09-23 — the old
  "servable today" line was false): the checkpoint SHIPS its draft head
  (all-EXL3-K4, pristine) and cinference carries mtp.cpp, but
  `weights.mtp` is unbound, spec is startup-fatal, and the pump never
  calls MTP — MTP-in-pump is BLOCKED on the engine ticket (Work queue
  §5), not servable.
- **A fused GEMM epilogue can beat the two-kernel reference by 1 ulp:**
  v3 folds the out-Had into the MMA epilogue (no intermediate fp16
  rounding), so sparse 1-ulp diffs vs hgemm+had on real magnitudes are a
  sign of FEWER roundings, not a bug. Synthetic-tile bit-exactness was
  luck of small values accumulating exactly.
- **CMake:** a bare `cmake <dir>` configures in-source and can clobber staging;
  always use explicit `-S` / `-B`.
- **Header-only compile probes are a weak gate** for templated kernels:
  `--expt-relaxed-constexpr` defers body semantics until real instantiation, so a
  probe with no launch can pass while an actual call site breaks; explicit
  `template __global__ void k<...>();` lines are also rejected by nvcc. Gate with
  a plain `__host__` function that launches (`k<...><<<1, blocks, 0, nullptr>>>(args)`).
  Also: launching from a kernel needs -rdc; from a host function it doesn't.
- **Cooperative groups pull CCCL:** `#include <cooperative_groups.h>` (via cuda
  headers) hard-fails with the default traditional MSVC preprocessor; the probe
  line needs `-Xcompiler "/Zc:preprocessor"`. We sidestep cooperative launch
  entirely in the plain carve, but the reference file still keeps the include.
- **Cached build paths don't follow renames.** The local PD/L2 plain-cache
  build carried a pinned pre-rename YES path
  (`exl3::exl3_gemv_pool_kernel`) via an old binary archive; after `pool ->
  plain` the local PD/L2 ports were rebuilt and the bin-path probe pinned to
  the NEW path. The pre-rename gate had run against the production-named
  binary (identical kernel, name-identical source) and its A/B match still
  holds: bin-path passes 1.0005352 / 2.0015795 vs Linux-gated production
  1.00053 / 2.00158.
- **Vendor-verbatim hygiene:** mechanical `namespace`-wrapped block copies can
  nest a helper namespace one level shallower than the reference (call sites
  then break with `no member of namespace`). Diff the wrapped file against the
  reference (line-level diff, expected differences only) after every reparent.
- **Int128 portability:** `__int128` in `runtime/contract/resources.h`
  (unsigned) has no Windows guard; in-tree dev configure is Linux-only until
  a standard C++/C99 type or builtins check is added.
- **Hadamard planes are warp-granular.** `had_*_r_128_inner` indexes
  `threadIdx.x & 31` and loads `half4[t]` — ONE 32-thread warp covers exactly
  128 elements. Launching a 128-thread block over one plane is a data race
  (four warps over the same 128 slots). Reference gemv runs these stages with
  `t = threadIdx.x / 32 + (t % 32) * gridDim.x`-style warp math inside the
  512/256-thread blocks.
- **Local model tool-calls:** tool calling works; the observed failure was
  format compliance (prose without the tool block), not environment.

- **compute-sanitizer on dense-exl3 gemv:** under memcheck, when chaining multiple high-block launches with allocs (like P11's 12-config matrix), the tool's own tracking surfaces false-out-of-bounds reads at the program's own `cudaMalloc`/CHECK frames (not the kernel) — see P11 `P11 runs=12 fail=73` + its in-config `fail=0` and the clean isolated P11b. The authoritative kernel gate is the direct production run; use sanitizer only on a single-config micro-probe (P11b).
- **This checkpoint's .safetensors are flat.** No `tensors` wrapper, no
  `version` key: the header JSON root maps tensor-name -> {shape,dtype,
  data_offsets} directly. Ported/existing readers may expect the standard
  (wrapped) layout. exl3_materialize accepts both; if you add a generic
  reader, do not assume `tensors`.
- **EXL3 trellis layout: [k/16, n/16, 16K]**: dim0 = k-side (input), dim1 =
  n-side (output). Confirmed on all 401 groups (401/401) + at lm_head
  (320*16 = 5120 = hidden = k; 15520*16 = 248320 = vocab = n). suh is
  always the k-side [k]; svh always the n-side [n]. c == 16K works across
  the board (0 violations). Note: an earlier draft of the load plan
  misread dimension order — check the k/n mapping independently.

### HyperQwen audit (criteria 5-6, 2026-09-21)

**Criterion 5 - EXL3/DFlash2 port from HyperQwen: verified hard negative.**
[HyperQwen](https://github.com/rktchip/HyperQwen.git) contains ZERO `.cu`/`.cuh`/`.cpp`/`.rs` files, and a
repo-wide grep for `exllama|exl2code|trellis` matches only a `.mp4`. It is a
vLLM 0.28 fork (43 Python/Triton patches + KVarN python files) serving the
**W4A16 AutoRound** model - its DFlash2 is Python speculator code, its KV
compression is Triton KVarN. No C++/CUDA to port. `buun-llama-cpp` is the
same stack class (178 py / 6 cu); criterion 2 stays satisfied by the
exllamav3 1.5.1 vendoring above - the only first-class EXL3 C++/CUDA
source on disk.

**Criterion 6 - HyperQwen speed/memory features cinference could adopt**,
ranked (engine wiring already exists in-tree: MTP + DFlash2 draft paths in
`mtp.cpp`/`draft.cpp`, CUDA decode graph in `decode_graph.cpp`, K8V4/int8
quant KV append, continuous batching — so these are algorithm ports, not
vLLM patches):

| item | measured win (HyperQwen docs) | adoption for cinference |
|---|---|---|
| DFlash2 n-gram lookup drafting + quantized candidate chains | 121.8 vs MTP 111.1 tok/s at C1 (MTP leads at C8: 407 vs 390) | port the draft proposal algorithm into `draft.cpp`; 4th-draft KV reuse |
| KVarN KV compression (Hadamard + 4-bit K / 2-bit V per 128-token tile) | 302k-420k-token pool vs ~205k (1.6-4x), needle correct 4k-240k | hand-rolled CUDA backend over the paged KV arena; buys 262k-ctx headroom; Huawei CSL, Apache-2.0 |
| split-KV verify attention (FLASH_ATTN, query-row tiling) + int4/int8/fp8 variants | DFlash2 + CUDA-graph correctness at long ctx | hand-rolled CUDA pass over KV pages; gated on DFlash2 k>1 port |
| int8-QK Triton prefill attention (head_dim 256) | prefill parity with flash | port logic, reuse int8-QK GEMM primitives |
| sort-free small-k top-k/top-p + multi-block row softmax | drafting sampling speedup | small self-contained kernel |
| Gumbel-salt draft temperature (vllm #54282) | acceptance at shifted temps | sampling logic, no CUDA |
| prewarm all spec rungs at boot | TTFT | one-shot boot pass |

- **KVarN is the top long-context play** (262k ctx is the 2026-09 primary
  target), but it is a backend-level change (KV format + attention kernels) —
  a standalone work order, not a drop-in.
- **DFlash2 acceptance degrades ~1.97 tok/step at long context** with
  ultra-peaked degenerate distributions (kvarn-v2-runner note) — any DFlash2
  port needs the degenerate-distribution guard, or acceptance silently
  collapses.
- **Freshness re-check 2026-09-22 (2nd pass, per-area evidence):**
  HEAD still `fe09615` (docs/docker only). `drafter/README.md` documents
  the portable methodology wins: draft-vocab coverage 92.1->97.5%
  (+10 tok/s -- measure coverage FIRST when wiring cinference's MTP
  speculative path), GPTQ int4 lm_head (KL 0.0068->0.0029, -1.8ms/step),
  and the MTP-finetune NEGATIVE (7M tokens, no needle movement -- do NOT
  burn 6h GPU; ship the checkpoint's MTP as-is, which our dispatch
  already serves). `batch/README.md`: 1,094 tok/s steady-state e2e at
  64-concurrent on a 3090 (vLLM) -- the honest e2e reference next to our
  linear-only scratch projection of 27k tok/s @128 on the 5090 (derived
  from per-m microbench timings, not measured e2e). kvarn glue is 2x~250-line
  patches over `files/vllm` Python -- concept (Hadamard KV rotation)
  noted, no C++ to port. Still 39 `patches/` + 4 kvarn (43 total),
  zero `.cu`/`.cuh`/`.cpp`/`.rs` repo-wide. `offload-wsl2-devptr.patch`
  (vLLM-Python device-VA fix, different stack) stays closed.
  Nothing new to port as code.

## System requirements

- **GPU:** single NVIDIA RTX 5090 (Blackwell sm_120a, 32 GB). The 27B EXL3 checkpoint (~15 GB) plus workspace must fit one card; the engine assumes one process owns the whole GPU.
- **OS:** Linux (Ubuntu 24.04+; WSL2 Ubuntu with NVIDIA CUDA-on-WSL driver works). Full-tree configure and serve-link are Linux-only: they need `pkg-config`, `libcurl` dev, and FFmpeg dev (`libavformat`/`libavcodec`/`libavutil`/`libswscale`). Windows builds cover translation-unit probes only (MSVC + CUDA toolkit), never the link.
- **Toolchain:** CMake 3.28+, Ninja, GCC 13+, CUDA toolkit 13.x (`nvcc`, sm_120a). Python 3.10+ for oracle/probe scripts only — never on the serve step.
- **Model input:** an EXL3 checkpoint directory (`model.safetensors*` + `model.safetensors.index.json` with a 409-entry `weight_map` + `quantization_config.json` + `config.json`/`tokenizer.json`).

## Known issues (serve correctness)

- **Serve requires `CUDA_EXL3_AUTOTUNE=0` at startup:** the v3 autotuner's
  capture guard (`v3_capturing()`) only queries stream 0, but finalize
  captures the GDN path on a worker stream — the first-ever EXL3 launches
  (empty tune cache) time candidates inside the capture, fail, and abort
  startup (`exl3_gemm: launch failed` + capture-invalidated abort). With
  the heuristic tier forced, startup completes cleanly. Long-term fix
  (pre-warm tuner or capture-aware guard) belongs to the graph-capture
  work order, not the serve path.
- **GDN/NVFP4 include foot-gun:** `src/ops/wrapper/gdn_input_proj.cpp` sits next to the EXL3 seam and its NVFP4 includes resolve under `ops/gdn_input_proj/...`, not `ops/linear/...`. EXL3 edits that touch this file have corrupted the path before (build breaks at dep-scan with a missing-header error). GDN/NVFP4 is ninfer's other quant path — EXL3 serve must not retarget it by accident. Keep the two include families separate; a dep-scan failure naming `nvfp4_gdn_input_plan.h` means the path prefix, not the file.
- **`exl3_program` is program listing only:** the serve-startup binder (`src/runtime/engine/exl3_program.{h,cpp}`) builds the model/program from the checkpoint dir. It must never become a second dispatch or generate loop next to `step_forward` / `exl3_dispatch`. One pump, one dispatch — if a second loop appears here, delete it.
- **Opaque EXL3 store pattern:** the full `Exl3EngineStore` types in `exl3_bind.h` are `__CUDACC__`-guarded, so host TUs (serve, program binder) must forward-declare under `namespace ninfer::exl3` with exact signatures (see `generation_service.cpp`). A private-namespace redeclaration links nowhere — undefined reference at link, always.
- **Stale-binary rule:** two-curl only counts against the binary that contains all of: pump mutex, workspace reserve, default concurrency 8, no decode break. Check the link timestamp before serving.
- **m=1 v3 quirk:** the v3 fused row deviates slightly at m=1 only (13/128 bits, maxAbs 0.25); single-token decode routes through the legacy gemv path (bit-exact), v3 serves m>=2. See Gate P16.
- **lm_head dominates decode:** full-width K6 head is ~25% of every decode step (~579us at m=1). Speculation or further head quantization is the first perf lever, not body kernels.
- **Serve invariants (dual-path):** serve-mode `Engine::submit` throws — admission is via the hook-loop inbox only; exactly one `RequestScheduler` (hook-loop owned) serves both prepare and pump; GDN/conv state is per-seq (zeroed at admit, freed at finish), never a shared buffer; the NVFP4/`.ninfer` CLI path (incl. `--spec mtp`) must keep working — EXL3 is the other door, not a replacement.
- **Graph ticket 2 shape:** capture layer launches only; block-table/upload, embed, and table gather stay eager; sample D2H after replay; `v3_capturing()` must see the capture stream (it currently queries stream 0).

## Major open issues (ordered; statuses mirror the Work queue above)

1. **Linux two-curl gate** — DONE (frozen: solo Paris + concurrent Paris/Rome, 24/24, T=0; binary sha 2746a95d).
2. **Graph capture on stable decode-only membership** — BLOCKED (named capture-stream blocker fixed 2026-09-23: `v3_capturing(cudaStream_t)`; replay == eager still needs EXL3 serve-path capture wiring; capture step doesn't write hidden; spec-on + graphs FATALs at startup — GraphExecUpdateFailure result 5, needs --no-cuda-graph. Gap merged into row 7: ~30 tok/s is the honest eager number, gap is runtime — close order (1) capture+graph loop, (2) one-layer ns vs oracle, (3) spec>eager after (1). UPDATE 2026-09-24: checkpoint-5 root-caused to platform (capture records, never executes eagerly here — capmini-proven) and cleared via capture+replay (8/8 green graphs-on); still parked default-off until perf probe + spec-on GraphExecUpdateFailure fixed).
3. **MTP-in-pump** — DONE 2026-09-23 (landed + lead-verified on C:/src binary: accept 3/commit 5, rewinds + cross-path exact, spec-off hold).
4. **Fused single-M attention** — BLOCKED behind graphs (stays flagged until bit-exact).
5. **KVarN / DFlash2 as separate work orders** — BLOCKED (no portable code exists: HyperQwen is vLLM Python/Triton only; both need standalone work orders, not serve tweaks).

## Run

> **Sibling product vs this engine.** The Huihui installer below is the
> NVFP4 sibling path, not EXL3 serve. To run this engine on EXL3:

```bash
CUDA_EXL3_AUTOTUNE=0 CUDA_EXL3_SPLIT_TARGET=0 \
  ./apps/ninfer-serve /path/to/Qwen3.8-27B-EXL3-3.5bpw \
  --host 127.0.0.1 --port 18080 --greedy
```

Requires Linux + NVIDIA drivers + the EXL3 checkpoint dir. Frozen
binary: 216731704 B, sha256 `2746a95d…94aa17f46b`.

For [Huihui Qwen3.8-27B Abliterated NVFP4](https://huggingface.co/satellitedown/Huihui-Qwen3.8-27B-abliterated-NVFP4-NInfer-v3), use the [one-menu installer](https://github.com/satellitedown/fast-long-context-cinference):

```bash
git clone https://github.com/satellitedown/fast-long-context-cinference.git
cd fast-long-context-cinference
bash setup.sh
```

Requires Linux and working NVIDIA drivers. Choose **1** to install, then **3** to start.

## Performance

> **Not this engine.** The table below is the Huihui NVFP4 + MTP-10
> sibling artifact on its own flags — it does not describe EXL3 serve.
> This engine's measured number: **~30 ms/token solo spec-off**
> (~33 tok/s; see Current contract). Kept for sibling reference only.

| Prompt tokens | Tokens/s |
|---:|---:|
| 8,192 | 450.78 |
| 32,768 | 432.60 |
| 131,072 | 364.84 |
| 260,000 | 299.64 |

Huihui Qwen3.8-27B Abliterated NVFP4, MTP-10, K8V4. Generation speed on synthetic recall, thinking off. [Measurements](results/rtx5090-archive-recall.json).

[Build from source](docs/maintainer/build-system.md) · [Technical docs](docs/README.md) · [Attribution](NOTICE)