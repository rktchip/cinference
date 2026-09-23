#!/bin/sh
# scripts/selfcheck.sh
# Phase 1 slot E selfcheck (Linux twin of selfcheck.ps1).
# Source-grep checks only: no build, no test binaries, no GPU, no tree edits.
# Prints PHASE <id> PASS|FAIL lines plus INFO lines, then a final STATE= line.
# STATE=LINUX_ONLY means P1-P6 plus P8-P11 all pass (P7 report-only pre-Phase-4);
# STATE=CODE means a real tree problem.
# ASCII only. POSIX sh.
set -u
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
FAILED=""

phase() {
  # $1=id $2=0/1 $3=detail
  if [ "$2" -eq 0 ]; then
    echo "PHASE $1 PASS $3"
  else
    echo "PHASE $1 FAIL $3"
    FAILED="$FAILED $1"
  fi
}

count_in_file() {
  # $1=file $2=pattern -> count (or -1 if missing)
  if [ ! -f "$1" ]; then echo -1; return; fi
  n=$(grep -c "$2" "$1" 2>/dev/null || true)
  echo "${n:-0}"
}

count_in_dirs() {
  # $1=pattern $2...=dirs -> total match lines in *.cpp/*.h/*.cc
  pat="$1"; shift
  total=0
  for d in "$@"; do
    if [ -d "$ROOT/$d" ]; then
      n=$(grep -rn --include='*.cpp' --include='*.h' --include='*.cc' "$pat" "$ROOT/$d" 2>/dev/null | wc -l)
      total=$((total + n))
    fi
  done
  echo "$total"
}

first_ln() {
  # $1=file $2=pattern -> first matching line number or -1
  if [ ! -f "$1" ]; then echo -1; return; fi
  n=$(grep -n "$2" "$1" 2>/dev/null | head -1 | cut -d: -f1)
  if [ -z "$n" ]; then echo -1; else echo "$n"; fi
}

echo "INFO root=$ROOT"

GS="$ROOT/src/serve/generation_service.cpp"

# P1_PUMP
RUN_LN=$(first_ln "$GS" 'GenerationService::run')
WARM_LN=$(grep -n '^void GenerationService::warmup' "$GS" 2>/dev/null | head -1 | cut -d: -f1)
BODY=""
if [ "$RUN_LN" -ge 0 ] 2>/dev/null; then
  if [ -n "$WARM_LN" ]; then
    BODY=$(sed -n "${RUN_LN},${WARM_LN}p" "$GS")
  else
    BODY=$(sed -n "${RUN_LN},\$p" "$GS")
  fi
fi
P1=0
echo "$BODY" | grep -q 'schedule_step' || P1=1
echo "$BODY" | grep -q 'dispatch_step' || P1=1
echo "$BODY" | grep -q 'on_step_done' || P1=1
if echo "$BODY" | grep -q '\.wait[[:space:]]*('; then P1=1; fi
SUBMITS=$(count_in_dirs 'submit[[:space:]]*(' src/serve apps/serve)
if [ "$SUBMITS" -ne 1 ]; then P1=1; fi
if [ "$RUN_LN" -lt 0 ] 2>/dev/null; then P1=1; fi
NOWAIT="true"; echo "$BODY" | grep -q '\.wait[[:space:]]*(' && NOWAIT="false"
phase "P1_PUMP" "$P1" "run_ln=$RUN_LN submits=$SUBMITS nowait=$NOWAIT"

# P2_LANE
CALLPAT='[.>]set_prefill_lane[[:space:]]*('
BATCH_CALLS=$(count_in_dirs "$CALLPAT" src/serve src/batch apps/serve)
SRC_CALLS=$(count_in_dirs "$CALLPAT" src)
CORE_CALLS=$(count_in_file "$ROOT/src/runtime/engine/engine_core.h" "$CALLPAT")
STATE="unguarded"
if [ "$CORE_CALLS" -eq 0 ] && [ "$SRC_CALLS" -eq 0 ]; then STATE="no-call-sites"; fi
if grep -q 'legacy_single_lane_' "$ROOT/src/runtime/engine/engine_core.h" 2>/dev/null; then
  LEGACY="true"
  if [ "$STATE" = "unguarded" ]; then STATE="legacy-gated"; fi
else
  LEGACY="false"
fi
P2=0
[ "$BATCH_CALLS" -ne 0 ] && P2=1
[ "$SRC_CALLS" -ne "$CORE_CALLS" ] && P2=1
if [ "$CORE_CALLS" -ne 0 ] && [ "$LEGACY" != "true" ]; then P2=1; fi
phase "P2_LANE" "$P2" "batchpath_calls=$BATCH_CALLS core_calls=$CORE_CALLS state=$STATE"

# P3_M
A=$(count_in_file "$ROOT/src/runtime/engine/step_forward.h" 'tokens\.size')
B=$(count_in_file "$ROOT/src/batch/cinference_hooks.cc" 'tokens\.size')
P3=0; [ "$A" -le 0 ] && P3=1; [ "$B" -le 0 ] && P3=1
phase "P3_M" "$P3" "step_forward_h=$A hooks_cc=$B"

# P4_ATTN (scoped: uniform else-if must follow the ragged if; an
# earlier uniform-only MTP helper is expected and ignored).
TX="$ROOT/src/models/qwen3_5/execution/text.cpp"
RAG_LN=$(first_ln "$TX" 'if (active_ragged_batch_ != nullptr)')
UNI_LN=-1
if [ "$RAG_LN" -ge 0 ] 2>/dev/null; then
  UNI_LN=$(grep -n 'active_sequence_batch_ != 0' "$TX" 2>/dev/null | cut -d: -f1 | awk -v r="$RAG_LN" '$1>r{print $1; exit}')
  [ -z "$UNI_LN" ] && UNI_LN=-1
fi
KERN="false"; grep -q 'causal_softmax_attention_ragged' "$TX" 2>/dev/null && KERN="true"
P4=0
[ "$RAG_LN" -lt 0 ] && P4=1
[ "$UNI_LN" -le "$RAG_LN" ] && P4=1
[ "$KERN" != "true" ] && P4=1
phase "P4_ATTN" "$P4" "ragged_ln=$RAG_LN uniform_ln=$UNI_LN kernel=$KERN"

# P5_GUARD
P5=0
grep -q 'RequireDeviceBuffersForServe' "$GS" 2>/dev/null || P5=1
grep -q 'GenerationService::~GenerationService' "$GS" 2>/dev/null || P5=1
phase "P5_GUARD" "$P5" "ctor_guard=$(grep -c 'RequireDeviceBuffersForServe' "$GS" 2>/dev/null || echo 0)"

# P6_TESTS
P6=0
for f in tests/test_run_pumps_hook_loop.cc tests/test_ragged_attention_map.cc tests/ops/test_exl3_launch_m47.cc; do
  [ -f "$ROOT/$f" ] || P6=1
done
grep -q 'test_run_pumps_hook_loop' "$ROOT/tests/cmake/RuntimeTests.cmake" 2>/dev/null || P6=1
grep -q 'test_ragged_attention_map' "$ROOT/tests/cmake/RuntimeTests.cmake" 2>/dev/null || P6=1
grep -q 'test_exl3_launch_m47' "$ROOT/tests/ops/tests.cmake" 2>/dev/null || P6=1
phase "P6_TESTS" "$P6" "files+cmake checked"
SP="$ROOT/tests/test_serve_path.cc"
if [ ! -f "$SP" ]; then
  echo "INFO test_serve_path.cc gone"
elif grep -q 'hook_loop-only' "$SP" 2>/dev/null; then
  echo "INFO test_serve_path.cc present-hookloop-rewrite"
elif grep -q '\.wait' "$SP" 2>/dev/null; then
  echo "INFO test_serve_path.cc present-legacy"
else
  echo "INFO test_serve_path.cc present-other"
fi

# P7_MTP (report only, non-blocking)
if grep -q '!plan\.has_prefill()' "$ROOT/src/batch/scheduler.h" 2>/dev/null; then
  echo "PHASE P7_MTP PASS spec-off-on-prefill"
else
  echo "PHASE P7_MTP FAIL no-prefill-gate"
fi

# P8_DECODE: run() must not bail on decode rows (slot A owns the runner seam).
# Brace-extract run(), strip //- and /* */-comments, then FAIL if
# decode_seq_ids.empty() is followed within ~6 lines by break, or if the
# 'until that runner lands' marker is present. PASS otherwise.
P8=0; P8BREAK="false"; P8MARKER="false"
if [ "$RUN_LN" -ge 0 ] 2>/dev/null; then
  RUNRAW=$(awk -v s="$RUN_LN" 'NR<s{next}{t=$0;o=gsub(/{/,"{",t);t=$0;c=gsub(/}/,"}",t);depth+=o-c;print;if(depth>0)st=1;if(st&&depth<=0)exit}' "$GS")
  if [ -z "$RUNRAW" ]; then
    P8=1
  else
    echo "$RUNRAW" | grep -q 'until that runner lands' && P8MARKER="true"
    STRIPPED=$(printf '%s\n' "$RUNRAW" | sed -e 's|//.*$||' -e 's|/\*[^*]*\*/||g')
    if printf '%s\n' "$STRIPPED" | awk '/decode_seq_ids\.empty\(\)/{w=6;next} w>0{if($0 ~ /(^|[^_A-Za-z])break[[:space:]]*;/)f=1;w--} END{exit (f?0:1)}'; then
      P8BREAK="true"
    fi
    [ "$P8BREAK" = "true" ] && P8=1
    [ "$P8MARKER" = "true" ] && P8=1
  fi
else
  P8=1
fi
phase "P8_DECODE" "$P8" "decode_break=$P8BREAK marker=$P8MARKER"

# P9_PUMP_LOCK: run() must hold a lock_guard (or unique_lock) over a mutex
# whose scope covers schedule_step AND dispatch_step AND on_step_done.
# Brace-extracted run() (see P8), comments stripped; accept pump_mutex_ or
# equivalent *mutex* name, report the name. FAIL until slot A lands.
P9=1; P9MUTEX="none"; P9LOCK=-1; P9SCHED=-1; P9DISP=-1; P9DONE=-1
if [ "${RUN_LN:-"-1"}" -ge 0 ] 2>/dev/null && [ -n "${STRIPPED:-}" ]; then
  P9LOCK=$(printf '%s\n' "$STRIPPED" | grep -n 'lock_guard\|unique_lock' | head -1 | cut -d: -f1)
  [ -z "$P9LOCK" ] && P9LOCK=-1
  if [ "$P9LOCK" -ge 0 ] 2>/dev/null; then
    LOCKLINE=$(printf '%s\n' "$STRIPPED" | sed -n "${P9LOCK}p")
    CANDS=$(printf '%s\n' "$LOCKLINE" | grep -o '[A-Za-z_][A-Za-z0-9_]*mutex[A-Za-z0-9_]*' || true)
    if [ -n "$CANDS" ]; then
      PICK=$(printf '%s\n' "$CANDS" | grep '_' | tail -1 || true)
      [ -z "$PICK" ] && PICK=$(printf '%s\n' "$CANDS" | tail -1)
      P9MUTEX="$PICK"
    else
      P9MUTEX="unknown"
    fi
  fi
  P9SCHED=$(printf '%s\n' "$STRIPPED" | grep -n 'schedule_step' | head -1 | cut -d: -f1)
  P9DISP=$(printf '%s\n' "$STRIPPED" | grep -n 'dispatch_step' | head -1 | cut -d: -f1)
  P9DONE=$(printf '%s\n' "$STRIPPED" | grep -n 'on_step_done' | head -1 | cut -d: -f1)
  [ -z "$P9SCHED" ] && P9SCHED=-1
  [ -z "$P9DISP" ] && P9DISP=-1
  [ -z "$P9DONE" ] && P9DONE=-1
  if [ "$P9LOCK" -ge 0 ] 2>/dev/null && [ "$P9SCHED" -gt "$P9LOCK" ] 2>/dev/null \
    && [ "$P9DISP" -gt "$P9LOCK" ] 2>/dev/null && [ "$P9DONE" -gt "$P9LOCK" ] 2>/dev/null; then
    P9=0
  fi
fi
phase "P9_PUMP_LOCK" "$P9" "mutex=$P9MUTEX lock_ln=$P9LOCK sched=$P9SCHED disp=$P9DISP done=$P9DONE"

# P10_WORKSPACE: ctor must call exl3_engine_reserve_workspace (slot A).
P10=1
P10N=$(count_in_file "$GS" 'exl3_engine_reserve_workspace')
[ "$P10N" -gt 0 ] 2>/dev/null && P10=0
phase "P10_WORKSPACE" "$P10" "calls=$P10N"

# P11_CONCURRENCY: serve_options.h max_concurrency default must read 8.
P11=1; P11VAL="none"
SOPTS="$ROOT/src/serve/serve_options.h"
if [ -f "$SOPTS" ]; then
  P11VAL=$(grep -o 'max_concurrency[[:space:]]*=[[:space:]]*[0-9][0-9]*' "$SOPTS" 2>/dev/null | head -1 | grep -o '[0-9][0-9]*' | tail -1 || true)
  [ -z "$P11VAL" ] && P11VAL="none"
  [ "$P11VAL" = "8" ] && P11=0
fi
phase "P11_CONCURRENCY" "$P11" "default=$P11VAL"

if [ -z "$FAILED" ]; then
  echo "STATE=LINUX_ONLY"
else
  echo "STATE=CODE"
fi
exit 0
