# MTP Breakdown-Run Spec — per-step T_draft / T_ver(M=4) / T_commit + accept histogram (5090 hold lift only)

> Pre-code artifact per committee demand. No code changes. Single GPU, serial session.
> Goal: price every option with a measured per-step breakdown + accept histogram, then compute the
> decode ceiling `(T_draft + T_ver) / E[tokens/step]`. Plus a conc-1 vs conc-2 step-time check
> (14 ms vs 34 ms suspicion: graphed B=1 vs eager B≥2).

## 0. Frozen inputs

| Item | Value |
|---|---|
| Repo | `C:/src/cinference` (verify `git rev-parse --short HEAD` at run start, record in header) |
| Binary | `/root/cinference-graph-build/apps/ninfer-serve` (record `sha256sum` + `stat -c '%s %y'`) |
| Model | `/mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw` |
| Env (mandatory) | `CUDA_EXL3_AUTOTUNE=0 CUDA_EXL3_SPLIT_TARGET=0` |
| Spec | `--spec mtp --draft-tokens 3 --lm-head-draft` (MTP-3; verify path asserts `kDrafts == 3`) |
| Graphs | run BOTH `NINFER_SERVE_GRAPH=verbose` (graphed arm) and `NINFER_SERVE_GRAPH=0` (eager arm) |
| Debug/trace | `NINFER_MTP_DEBUG=1` (else **no `[serve-mtp]` lines — see §4**), `NINFER_SERVE_STEP_TRACE=1` |
| Concurrency | serial: one request at a time, `--max-batch 1`-equivalent; conc-2 probe in §6 only |
| GPU | single 5090, no other CUDA process (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) |

## 1. What is being measured (definitions)

Per **MTP decode step** (one `step_mtp_decode`, `F` = anchor pos, `lane` = KV row):

- `T_draft` = 3× serial AR draft rows (`mtp_forward_ar_step` + 1 `synchronize` + D2H each; `engine.cpp` ll. 954–984). 3 host round-trips by construction.
- `T_ver` = one width-4 target verify (`target_verify_batch` over `[b@F, d0@F+1, d1@F+2, d2@F+3]`, M=4, 1 sync; ll. 1001–1107). **M is always 4 here.**
- `T_commit` = `commit_len` × `run_single_row` (ordinary ladder, `allow_graph=true`) + `mtp_forward_batch` refill each row (ll. 1141–1155). `commit_len = 5` if `accepted==3`, else `accepted+2`.
- Sync gaps = 5 `device_.synchronize()` per step by construction: 1 bonus-argmax (l. 933) + 3 draft (l. 967) + 1 verify (l. 1104). Commit rows have no extra sync.
- `accepted ∈ {0,1,2,3}` from `[serve-mtp] … accepted=N commit=M` (emitted **only** under `NINFER_MTP_DEBUG`, l. 1173–1179).
- `E[tokens/step] = mean(commit_len)` over decode steps (prefill/fill steps excluded).
- Ceiling: `(T_draft + T_ver) / E[tokens/step]` in ms/tok — the per-token cost **before** commit; compare against measured step ms/tok to price commit + gaps.

Graph log actions (`NINFER_SERVE_GRAPH=verbose` only; `graph_verbose_`, l. 452–458):

- `[vgraph] lane=… F=… action=replay|capture|eager-first|eager-disabled|capture-fail` (ll. 1045–1049)
- `[cgraph] lane=… pos=… action=replay|capture|eager-first|eager-disabled|capture-fail` (ll. 813–817)
- First verify/commit per process = `eager-first`, 2nd = `capture`, 3rd+ = `replay` (steady state). `capture-fail`/`eager-disabled` = fall back to eager arm semantics.

## 2. Workload (3 prompts, serial, fixed order)

| ID | Prompt (exact) | Why | Gen tokens |
|---|---|---|---|
| FOX-64 | `Repeat the word "fox" exactly 64 times, separated by single spaces, with no other text.` | high-accept (repetitive; cf. recall archive 93–98% draft acceptance) | 64 |
| PARIS | `Write a 150-word travel paragraph about Paris, mentioning the Seine, the Louvre, and café culture.` | mid-accept prose baseline | 150 |
| CODE-LOW | `Emit the first 40 digits of pi as a comma-separated list inside a python list literal, then the squares of 17, 23, 41 each on its own line prefixed by "# ". No other text.` | low-accept (code/numbers, entropic) | ~80 |

All at temperature 0 / greedy (`--temperature 0`), same `--max-tokens` per prompt, one request at a time. Run order FOX → PARIS → CODE-LOW per arm; 1 warmup FOX (discarded, covers `eager-first`+`capture`) + 1 measured pass each.

## 3. Exact commands

### 3.1 Serve launch (graphed arm; repeat with `NINFER_SERVE_GRAPH=0` for eager arm)

```bash
export CUDA_EXL3_AUTOTUNE=0 CUDA_EXL3_SPLIT_TARGET=0
export NINFER_MTP_DEBUG=1 NINFER_SERVE_STEP_TRACE=1 NINFER_SERVE_GRAPH=verbose
LOG=mtp-graph.log
/root/cinference-graph-build/apps/ninfer-serve /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --port 8901 2> "$LOG" 1> "${LOG%.log}.out" &
SRV=$!; sleep 20; curl -sf http://127.0.0.1:8901/health || curl -sf http://127.0.0.1:8901/v1/models; echo "srv=$SRV log=$LOG"
```

Eager arm: same but `NINFER_SERVE_GRAPH=0`, `LOG=mtp-eager.log`, `--port 8902`.

### 3.2 Serial client (record wall ms/token per prompt)

```bash
# FOX-64
curl -s http://127.0.0.1:8901/v1/completions \
 -H 'Content-Type: application/json' \
 -d '{"prompt":"Repeat the word \"fox\" exactly 64 times, separated by single spaces, with no other text.","max_tokens":64,"temperature":0}' \
 -o fox.json -w 'http=%{http_code} time_total=%{time_total}s\n'; python3 -c "import json;d=json.load(open('fox.json'));print('fox_chars=',len(d.get('choices',[{}])[0].get('text','')))"
# PARIS (same shape, max_tokens 150) -> paris.json ; CODE-LOW (max_tokens 96) -> codelow.json
```

### 3.3 nsys profile (cuda-only, one prompt per capture; repeat per prompt × arm = 6 captures min)

```bash
export CUDA_EXL3_AUTOTUNE=0 CUDA_EXL3_SPLIT_TARGET=0
export NINFER_MTP_DEBUG=1 NINFER_SERVE_GRAPH=verbose
nsys profile --trace=cuda --cuda-memory-usage=false \
  --output=mtp-fox-graph --force-overwrite=true \
  /root/cinference-graph-build/apps/ninfer-serve /mnt/c/models/Qwen3.8-27B-EXL3-3.5bpw \
  --spec mtp --draft-tokens 3 --lm-head-draft --port 8903 &
# then: drive ONE prompt (FOX) serially, stop server at prompt end; report maps to mtp-fox-graph.nsys-rep + .sqlite
```

Keep captures small: profile **decode only** — start traffic after warmup, stop server right after EOS. nsys row range for decode: filter sqlite rows to timestamps between first and last `[serve-mtp]` line of the prompt (get epoch bounds via `date +%s%N` echo markers around the curl, or match log line numbers → wall time).

### 3.4 Concurrency probe (§6)

```bash
# conc-1: single FOX-64 as above. conc-2: two FOX-64 curls in parallel (background both, wait), server otherwise identical.
# Record per-request time_total and server [serve-step] m= / span lines during overlap.
```

## 4. Log greps (exact strings, from `src/runtime/engine/engine.cpp`)

```bash
LOG=mtp-graph.log   # or mtp-eager.log
# 4a. per-step accept/commit (NINFER_MTP_DEBUG=1 REQUIRED; format l.1174):
grep -c '\[serve-mtp\]' "$LOG"                                   # N decode steps total
grep -o 'accepted=[0-3] commit=[0-9]' "$LOG" | sort | uniq -c     # histogram raw
grep '\[serve-mtp\]' "$LOG" | head -3                             # eyeball format
# expected line shape:
# [serve-mtp] seq=7 lane=0 F=41 stash=1 anchor=1234 drafts=[55 66 77] verify=[55 66 12 34] accepted=2 commit=4
# 4b. graph actions (NINFER_SERVE_GRAPH=verbose REQUIRED; ll.1047,1115? actually 1047/815):
grep -o '\[vgraph\] .* action=[a-z-]*' "$LOG" | awk '{print $NF}' | sort | uniq -c   # expect steady-state replay
grep -o '\[cgraph\] .* action=[a-z-]*' "$LOG" | awk '{print $NF}' | sort | uniq -c
# 4c. batch shape during steps (NINFER_SERVE_STEP_TRACE=1; ll.719,725,752):
grep '\[serve-step\]' "$LOG" | head -5
# 4d. failure sentinels (must be ZERO in a good run):
grep -c 'capture-fail' "$LOG"; grep -c 'eager-disabled' "$LOG"   # eager arm: vgraph/cgraph absent entirely
```

Accept-histogram aggregator (run per prompt window; use per-prompt log slices, see §5):

```bash
grep '\[serve-mtp\]' "$LOG" | grep -o 'accepted=[0-3]' | sort | uniq -c
# success shape (example, counts vary):
#   41 accepted=0
#   22 accepted=1
#   11 accepted=2
#    6 accepted=3
python3 -c "
import re,collections,sys
acc=collections.Counter(int(m) for m in re.findall(r'accepted=([0-3])',open(sys.argv[1]).read()))
com=[int(m) for m in re.findall(r'commit=(\d+)',open(sys.argv[1]).read())]
n=sum(acc.values()); e=sum(com)/len(com)
print({k:acc[k] for k in [0,1,2,3]}, 'n=',n,'E[tok/step]=%.3f'%e)" "$LOG"
# success shape: {0: 41, 1: 22, 2: 11, 3: 6} n=80 E[tok/step]=2.775
```

## 5. nsys sqlite queries (cuda-only row range)

Open with `sqlite3 <capture>.sqlite` (or `.nsys-rep` → export sqlite). Table names vary by nsys version; try in order `CUPTI_ACTIVITY_KIND_KERNEL`, `KERNEL`, `CUPTI_ACTIVITY_KIND_MEMCPY`, `MEMCPY`.

```sql
-- 5a. list tables present
.tables
-- 5b. decode-window kernel totals grouped by name (restrict :t0/:t1 to first..last [serve-mtp] ns of the prompt)
SELECT name AS kernel, COUNT(*) AS n, SUM(duration)/1e6 AS total_ms, AVG(duration)/1e6 AS mean_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start BETWEEN :t0 AND :t1
GROUP BY name ORDER BY total_ms DESC LIMIT 25;
-- 5c. sync/gap proxy: memcpy H2D/D2H counts + durations in window (draft=3×H2D+D2H, verify=bulk H2D + D2H)
SELECT name, COUNT(*) AS n, SUM(duration)/1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE start BETWEEN :t0 AND :t1
GROUP BY name;
-- 5d. per-step cadence: order kernel bursts by start; verify bursts are the widest M=4 launches
SELECT start, name, duration/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start BETWEEN :t0 AND :t1
ORDER BY start LIMIT 40;
```

Mapping rows → phases (no code change; classify by kernel/launcher adjacency):

- `T_draft`: kernels between a draft H2D (`mtp_ids_`/`mtp_pos_`) and its D2H (`mtp_tok_`), ×3 per step; expect 3 roughly-equal blocks.
- `T_ver` (M=4): the single widest `target_verify_batch` burst per step (4-column); compare mean vs draft-block sum.
- `T_commit`: `commit_len` repetitions of ordinary-decode + `mtp_forward_batch` refill bursts after the verify D2H.
- Sync gaps: `cudaDeviceSynchronize` host-side gaps show as idle gaps between bursts (5/step: 1 bonus + 3 draft + 1 verify) — report mean gap ms from 5d start-deltas minus burst durations.

Success shapes:

- 5b returns ≥3 kernel families with `total_ms` descending; top rows stable across FOX/PARIS/CODE captures (±15%).
- 5c memcpy `n` scales ≈ steps × (draft H2D/D2H pairs + verify bulk); CODE-LOW shows more steps per token (lower E[tok/step]) than FOX-64.
- 5d shows repeating 3-narrow + 1-wide + commit-tail pattern per step.

## 6. Conc-1 vs conc-2 step-time check (14 ms vs 34 ms suspicion)

Hypothesis: graphed B=1 ≈ 14 ms/step; eager B≥2 ≈ 34 ms/step (B≥2 never graphs: `isDecodableGraphCandidate`-style gate `n_dec==1 && !mtp` / MTP width gate fails → eager).

```bash
# conc-1 (serial): §3.2 FOX, record curl time_total; step ms = wall/steps where steps = grep -c serve-mtp in window
# conc-2 (parallel): launch two identical FOX curls simultaneously against the SAME server; record both time_totals
grep '\[serve-step\]' mtp-conc2.log | grep -E 'm=[12] |span=' | tail -10   # confirm m=2 overlap rows
```

Report table (success shape):

| arm | conc | steps | wall s | ms/step | graph action mix |
|---|---|---|---|---|---|
| graph | 1 | ~30 | ~0.42 | ~14 | replay ~100% |
| graph/eager | 2 | ~60 | ~2.04 | ~34 | eager-first/disabled or m=2 rows |

Pass criterion: conc-2 ms/step ≥ 2× conc-1 with `[serve-step] m=2` overlap and no `replay` during overlap → suspicion confirmed (batching, not regression). If conc-2 ≈ conc-1, suspicion is wrong; escalate with logs.

## 7. Ceiling computation + decision table

```bash
# inputs: T_draft_ms, T_ver_ms (means from §5b/5d), E from §4 aggregator
python3 -c "Td=9.0;Tv=6.0;E=2.775;print('ceiling_ms_per_tok=%.3f'%((Td+Tv)/E))"
```

Report per prompt × arm:

| prompt | arm | steps | E[tok/step] | T_draft | T_ver(M=4) | T_commit | gaps | ceiling ms/tok | measured ms/tok |
|---|---|---|---|---|---|---|---|---|---|
| FOX-64 | graph | … | ~3–4 | … | … | … | … | … | … |
| PARIS | graph | … | ~2–3 | … | … | … | … | … | … |
| CODE-LOW | graph/eager | … | ~1–2 | … | … | … | … | … | … |

Pricing rule: if `measured ≈ ceiling + T_commit/E + gaps` → commit+gaps dominate ⇒ options that cut commit rows (higher accept) or fuse refill win; if `measured ≈ T_ver`-bound with low E ⇒ verify/M=4 dominates ⇒ cut M or draft depth. No option proceeds without filling its row.

## 8. Acceptance checklist (done = all verified)

- [ ] 6 nsys captures (3 prompts × 2 arms) + 2 conc probes exist as `.nsys-rep`/`.sqlite`, decode-windowed by `[serve-mtp]` bounds.
- [ ] Per prompt × arm: histogram over `{0,1,2,3}` with `n` = `[serve-mtp]` count; `E[tok/step]` computed from `commit=` values.
- [ ] Per prompt × arm: `T_draft`, `T_ver(M=4)`, `T_commit`, mean sync gap — each traced to a numbered sqlite query output.
- [ ] Ceiling `(T_draft+T_ver)/E` in ms/tok vs measured wall ms/tok, per row of §7.
- [ ] Conc-1 vs conc-2 table filled; 14-vs-34 suspicion confirmed or refuted with `[serve-step] m=` evidence.
- [ ] Zero unexplained `capture-fail`; any `eager-disabled` only in the `NINFER_SERVE_GRAPH=0` arm.
- [ ] Header records: git SHA, binary sha256+size, `nvidia-smi` clean, full env dump.
