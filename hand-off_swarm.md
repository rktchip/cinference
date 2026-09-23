# Hand-off Addendum: Swarm Plan + Honest State + Direct-Audit Ask

Companion to `hand-off.md` (same dir). ASCII on purpose (the write path keeps
corrupting long non-ASCII/code cells). Read this BEFORE dispatching any
sub-agent. Confidence tags: [SOLID] verified on-disk + re-run, [MODERATE]
verified once, not re-run under latest edits, [FRAGILE] not compiled / not
re-run, audit the bytes.

============================================================
PART A. WHERE I HAVE BEEN STRUGGLING (honest, in priority order)
============================================================

A1. THE WRITE PATH CORRUPTS LONG HAND-TYPED CELLS.  Top problem all session.
    Long execute_code cells, heredocs, sed/printf, even big patch new_string
    get mangled (dropped parens, backslash drops, "ninfa"-for-"ninfer", stray
    "str"— tokens, non-ASCII artifacts). Workaround that WORKED: author long
    code only via generator scripts that BYTE-COPY known-good lines from disk,
    sha256 after every write, never retype a long line. Cost: a lot of time,
    and it is the reason the v3 files below are ATEN-STRIPPED but NOT COMPILED.
    CONSEQUENCE FOR AUDIT: any file tagged MODIFIED is untrusted until someone
    diffs it against the pristine source AND compiles it. See Part C.

A2. P12b WAS NOT CLEANLY RE-VERIFIED AFTER THE HYBRID HAD EDIT.
    I changed exl3_hadamard.cuh (hybrid pre/post scale, criterion 7). I
    re-ran P12c (clean) but P12b only showed same-order output, and the
    sign-flip count read 4 where the original pass was 2. I did NOT chase
    this to a definitive re-compare.  Tag [FRAGILE]. This is item C2 below.

A3. P15 UNDER THE HYBRID IS NOT YET PROVEN FINITE.
    The hybrid is supposed to make K5 (o_proj) finite where 1.5.1 is all-NaN.
    I had the P16 suh_scale micro-harness ready to prove it but was interrupted
    before the ~6000x run. Tag [FRAGILE] until that run lands.

A4. EVERYTHING GPU is SERIAL on ONE 5090.
    PID 13908 holds ~25.5 GB (:8290, tp8). Only ~6.5 GB is free. Any
    sub-agent swarm that launches device work in parallel will OOM or collide.
    This is a hard constraint that shapes Part B (serial GPU lane).

A5. Windows scope is gates-only. Full cmake configure, forward pass, KV
    residency, engine loop, bit-exact full-model oracle are ALL Linux-gated.
    I have only single-layer numeric gates + launch matrices. Do not claim
    "serves Qwen 3.8" -- that is Linux work not yet run.

============================================================
PART B. SUGGESTED SUB-AGENT SWARM STRUCTURE
============================================================

Principle: the dependency chain is mostly SEQUENTIAL and the GPU is a shared
scarce resource. So this is NOT "many agents, all racing." It is a small
pipeline: 1 critical-path seeder -> 2 host-only parallel workers -> 1 serial
GPU finisher -> 1 integrator/auditor. Keep agents FEW and SPECIALIZED (the
write path corrupts long cells, so fewer long-lived agents with clear file
ownership outperforms many overlapping ones).

GLOBAL RULES every sub-agent must inherit (put in each task prompt verbatim):
 - Repo C:/src/cinference, pristine scratch C:/src. Do NOT touch
   PID 13908 / :8290. Keep device tests under ~6 GB.
 - Never create a git identity, never commit, keep scratch OUT of commits.
 - Author long code only via generator scripts that copy known-good lines
   from disk + sha256-verify afterwards. Never retype a long line.
 - Report the FIRST fatal error only, then STOP at your stop-line. No extra
   steps. Deliver the exact artifact + the command to verify it.
 - Windows: nvcc sm_120a, MSVC 19.44 BuildTools, /Zc:preprocessor.
   venv python for 1.5.1 oracle: /c/src/exllama/.venv/Scripts/python.exe.

-- PHASE 0 (critical path, ONE agent, unblocks everything) -----------------
Goal: make the ATEN-STRIPPED v3 TUs COMPILE, and leave a byte-verified,
sha'd "v3 compiles" checkpoint.
Inputs: torchexl3/exl3_common.cuh, exl3_codebook.cuh, exl3_dq.cuh,
  exl3_had.cuh (pristine), torchexl3/exl3_gemm.cu + exl3_hadamard.cu
  (ATen-stripped), exl3_v3_alias.h.
Steps:
 1. Build p16_bat.bat (vcvarsall x64 line byte-copied from exl3_p12c.bat,
    sha-verified) that nvcc-compiles exl3_gemm.cu + exl3_hadamard.cu
    (sm_120a, -std=c++20, --expt-relaxed-constexpr, /Zc:preprocessor,
    -I torchexl3 so the bare internal includes resolve, and the v3 TUs see
    the global half/bfloat16 aliases from exl3_v3_alias.h -- include that
    header at the top of each TU or a transparent prelude TU).
 2. Fix first-compile errors ONLY (expected: the unqualified half/bfloat16
    alias, a stray ATen include, one or two names). Re-sha after each fix.
    Do not "improve" kernel bodies -- they are the vendored MIT original.
 3. Deliver: a .bat that returns 0, the resulting .o files, and the sha256
    of the two modified TUs. STOP when the TUs compile clean.
Stop-line: "both v3 TUs compile, zero errors" -- or FIRST compile error,
  reported with file:line, and stop.
Why first: no P16 numeric run, no row wiring, no bench is possible until
this compiles. It is the single unblocker.

-- PHASE 1 (after Phase 0 compiles, TWO agents in PARALLEL) ----------------
Both are HOST-ONLY (no device launches), so they safely run together.

 Agent 1A -- P16 oracle (numpy, no GPU):
   Goal: produce p16_oracle.py that emits the reference for the P16 tiles.
   Inputs: p16_row.cu (tile formulas), torchexl3/exl3_codebook.cuh (the exact
     decode tables -- graft the same ones already byte-verified in p12_oracle.
     py: cb0 3inst table, cb2 mul1 dp4a(x,0x01010101,0x6400) + k_inv 0x1eee,
     k_bias 0xc931), torchexl3/exl3_dq.cuh (bit-extract formula: 16-bit chunk
     at offset t*bits + bits - 16 within each 16-unit, read from int16).
   Deliver: p16_oracle.py writing p16_ahad_ref.bin (u16) and
     p16_out_ref.bin (bf16) for bits 3/4, cb2, m in {1,4,8}, and a one-line
     checksum. Must NOT allocate >= 1k rows (guardrail).
   Stop-line: oracle emits both reference bins for all combos, or first error.

 Agent 1B -- P12b re-audit (the correctness gap, A2):
   Goal: definitively re-verify P12b AFTER the hybrid had edit, and explain
     the sign-flip delta (was it 2 pre-edit, 4 post-edit? why?).
   Inputs: exl3_hadamard.cuh (current, hybrid), p12b_oracle.py /
     cmp_p12b.py, final_p12b.py from scratch.
   Deliver: a number (pre-edit vs post-edit sign-flips + maxAbs), a one-line
     verdict (bit-exact / within 1-2 ulp / regression), and the exact
     command that reproduces it. If it is a regression, PROPOSE the minimal
     restore (revert hybrid to pure 1.5.1 fp16 scale, OR fix the pick) but do
     NOT apply it -- leave the decision to the integrator. Do not launch
     anything that allocates more than the already-budgeted gate size.
   Stop-line: verdict + reproducible command, or first error.

-- PHASE 2 (after Phase 0 + 1A, ONE agent, SERIAL GPU LANE) ---------------
Only one agent may touch the GPU at a time (A1 rule).
Goal: run P16 and record the two gates.
  (a) suh_scale=1, bits 3/4, cb2, m in {1,4,8}: compare p16_ahad vs
      p16_ahad_ref (bit-exact) and p16_out vs p16_out_ref.  PASS = ahad
      bit-exact AND out within 1-2 ulp AND non-finite count = 0.
  (b) suh_scale ~ 6000 (K5 o_proj is the natural K5 micro-probe; the K5 dump bin
      already shown all-NaN at scale 1.0 through 1.5.1): expect FINITE out
      from the hybrid row (where 1.5.1-fp16-semantics would NaN). This is the
      P15 micro-pass right (A3).
 Deliver: p16 run log (m, bits, split, ms5, ahad maxErr, out maxErr,
   non-finite count, finite-at-6000x yes/no) + the .bin artifacts.
 Stop-line: both runs land, or OOM/first error (report VRAM state, stop, do
   NOT kill PID 13908).

-- PHASE 3 (integrator + final audit, ONE agent, after 1A/1B/2) -----------
Goal: fold in decisions, wire, bench, re-audit.
 - Decide hybrid-in-v3-had_in (2-line port into torchexl3/exl3_had.cuh
   had128_warp_in) vs keep K5 in the legacy reconstruct path. Use the 1B
   + 2(b) results.
 - Write the multi-group row body (exl3_aln_rows_bf16) -- currently declared
   only (A: see hand-off 6 for the ShardMap approach).
 - Wire rows into the QType::EXL3 dispatch (binder seam): m<=128 via the v3
   row; K5/K6 + fallback via the existing staged gemv reconstitute path.
 - Bench: v3 row vs current 11-gemv, per m in {1,4,8,32,64,128}, single 5090 --
   this is the criterion-1 number.
 - Re-gate the full matrix (P12c, P12b strictly), update README (P16 +
   criterion-7 replace section + gotchas), run the FINAL audit (criterion 4).
   Only a clean audit closes -- if a change was made, re-audit once more.
 Deliver: green matrix + README diff + audit report. Stop at the audit gate.

Sizing/ownership: Phase 0 = 1 agent (owns the v3 compile). Phase 1 = 2
(side-by-side, host-only, no file overlap). Phase 2 = 1 agent (sole GPU).
Phase 3 = 1 agent. Total 5, or collapse Phase 1B+2 to one if preferred.
Do NOT run Phase 2 in parallel with anything else. File ownership is
disjoint across the 1A/1B pair, so no merge conflict.

============================================================
PART C. WHAT I BELIEVE IS DONE CORRECTLY  (direct-audit checklist)
============================================================
Audit each with the given command. Tags = how much I trust it pre-audit.

 [SOLID] 1. Original 1.5.1-ported baseline gates. On the staged tree:
   P12 (dequant byte-exact), P12c (decode m=1/4/8), P13 (sanitizer census),
   P14 (C5/C6 byte-exact + sanitize counts 198/155, 166/7278). Re-run the
   P12c/P14 bats from scratch and diff the .bin outputs against the oracles.
   Command form: bash exl3_p12c.bat && bash cmp_p12c.py (scratch dir).

 [SOLID] 2. P15 IS A CHECKPOINT DATA DEFECT (parity), NOT a port bug.
   Evidence: 1.5.1's OWN LinearEXL3.forward is ALL-NaN on K5 o_proj
   (81920/81920) and lm_head (3973120/3973120); a clean K4 group is finite;
   a controlled 6000x through 1.5.1's had kernel stays finite (kernels sound,
   checkpoint amplitudes overflow fp16). Audit by re-running rt_151.py
   (scratch) and confirming the all-NaN counts reproduce.

 [SOLID] 3. The pristine v3 headers (common/codebook/dq/had) are byte-verified
   (sha sidecars present in torchexl3/). Audit: sha256sum torchexl3/* against
   the sidecar .sha files.

 [MODERATE] 4. The ATen-stripped v3 TUs (exl3_gemm.cu, exl3_hadamard.cu).
   I STRIPPED the ATen host layer but have NOT compiled them. Audit: compile
   (Phase 0 bat) AND diff the kernels against the pristine C:/src/cuda-exl3
   source -- the kernel bodies must be byte-identical, only the host layer
   should differ. If any kernel body changed, that is a defect.

 [FRAGILE] 5. The hybrid pre/post scale in exl3_hadamard.cuh + the P12b gate.
   Re-audit P12b after the edit (Phase 1B). If sign-flips regressed, restore.

 [FRAGILE] 6. P15 finite-under-hybrid. Prove with the 6000x run (Phase 2b)
   before claiming K5 can be served.

 [FRAGILE] 7. exl3_aln.h / exl3_aln.cu (the raw-row seam). Not compiled. The
   multi-group row is declared but NOT defined. Audit by compiling with the
   v3 TUs and confirming the row bodies actually call the v3 kernel.

 NOT-DONE (do not pre-judge): QType dispatch wiring, DFlash2 draft tensors
 (0 draft tensors in this checkpoint -- binder seam exists, no work needed
 until a draft checkpoint arrives), multi-user engine loop, KV residency,
 full-model oracle. All Linux-gated.

NET: items 1-3 are the load-bearing, high-confidence ground. 4-7 need the
audit runs before anyone should call the replacement "done." The baseline
(1-2) is the safe fallback that always works while 4-7 are proven out.

============================================================
(End of addendum -- pair this with hand-off.md)
============================================================