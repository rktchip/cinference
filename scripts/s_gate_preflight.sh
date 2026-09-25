#!/bin/sh
# scripts/s_gate_preflight.sh
# Mechanical single-server rule for all S-number timings (S1..S4).
# Usage:
#   s_gate_preflight.sh pre   [--max-idle-util PCT] [--out REF]  # before launch: expect 0 compute procs
#   s_gate_preflight.sh stamp [--allow N]                        # one-shot stamp during/after run
#   s_gate_preflight.sh post --ref REF [--allow N] [--max-clock-drift PCT]
#     # end-of-window stamp: VOID if nproc/PIDs differ from REF or SM clock
#     # drifted more than PCT% (default 3). Compares against the START stamp.
# An S-line without a matching START+END stamp pair is void by default.
# The rule is retroactive: unstamped history (14.3, 22.6, conc gaps, TTFT
# slopes, bucket tables, the eager-verify price) re-baselines in the first
# clean gate. Interleave A/B (slots-vs-legacy, on-vs-off) A-B-A-B back to
# back in one session at the same thermal state; stamps record drift that
# interleaving does not cancel (m16 is math-bound and moves with SM clock).
# POSIX sh. Runs under WSL (nvidia-smi present). No GPU work itself.
set -u

MODE="${1:-stamp}"
ALLOW=1
MAXIDLE=5
MAXDRIFT=3
OUT=""
REF=""
if [ "$MODE" = "pre" ]; then ALLOW=0; fi
shift 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --allow) ALLOW="$2"; shift 2;;
    --max-idle-util) MAXIDLE="$2"; shift 2;;
    --max-clock-drift) MAXDRIFT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    *) shift;;
  esac
done

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
HASH="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

GPUQ="$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,clocks.current.sm,clocks.current.memory,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")"
# util,memutil,temp,smclk,memclk,memused,memtotal
UTIL="$(echo "$GPUQ" | cut -d, -f1 | tr -d ' ')"
MEMU="$(echo "$GPUQ" | cut -d, -f2 | tr -d ' ')"
TEMP="$(echo "$GPUQ" | cut -d, -f3 | tr -d ' ')"
SMCLK="$(echo "$GPUQ" | cut -d, -f4 | tr -d ' ')"
MEMCLK="$(echo "$GPUQ" | cut -d, -f5 | tr -d ' ')"
MEMUSED="$(echo "$GPUQ" | cut -d, -f6 | tr -d ' ')"
MEMTOTAL="$(echo "$GPUQ" | cut -d, -f7 | tr -d ' ')"

CAPPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "")"
if [ -z "$(echo "$CAPPS" | tr -d ' \n\r')" ]; then
  NPROC=0
  PIDS="-"
else
  NPROC="$(echo "$CAPPS" | grep -c .)"
  PIDS="$(echo "$CAPPS" | tr '\n' ';')"
fi
NSERV="$(echo "$CAPPS" | grep -c 'ninfer-serve' || true)"

save_ref() {
  # $1=file
  {
    echo "hash=$HASH"
    echo "nproc=$NPROC"
    echo "pids=$PIDS"
    echo "smclk=$SMCLK"
    echo "temp=$TEMP"
  } > "$1"
}

pct_diff() {
  # $1=old $2=new -> integer percent |new-old|/old, or -1 if non-numeric
  case "$1$2" in ''|*[!0-9]*) echo -1; return;; esac
  if [ "$1" -eq 0 ]; then echo -1; return; fi
  d=$(( $2 > $1 ? $2 - $1 : $1 - $2 ))
  echo $(( d * 100 / $1 ))
}

# 3x1s idle samples catch display/compositor activity on the 5090
if [ "$MODE" = "pre" ]; then
  S1="$UTIL"
  sleep 1
  S2="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  sleep 1
  S3="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  echo "STAMP hash=$HASH mode=start nproc=$NPROC ninfer=$NSERV pids=[$PIDS] idle_util=[$S1,$S2,$S3]% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
  [ -n "$OUT" ] && save_ref "$OUT"
  FAIL=""
  [ "$NPROC" -gt "$ALLOW" ] && FAIL="compute_procs=${NPROC}>${ALLOW}"
  for s in "$S1" "$S2" "$S3"; do
    case "$s" in ''|*[!0-9]*) continue;; esac
    [ "$s" -gt "$MAXIDLE" ] && FAIL="${FAIL:+$FAIL }idle_util=${s}>${MAXIDLE}"
  done
  if [ -n "$FAIL" ]; then echo "REFUSE $FAIL"; exit 1; fi
  echo "GO"
  exit 0
fi

if [ "$MODE" = "post" ]; then
  echo "STAMP hash=$HASH mode=end nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
  if [ -z "$REF" ] || [ ! -f "$REF" ]; then echo "VOID no-start-stamp"; exit 1; fi
  RNPROC=""; RPIDS=""; RSM=""; RHASH=""
  while IFS='=' read -r k v; do
    case "$k" in
      nproc) RNPROC="$v";; pids) RPIDS="$v";; smclk) RSM="$v";; hash) RHASH="$v";;
    esac
  done < "$REF"
  VOID=""
  [ "$NPROC" != "$RNPROC" ] && VOID="nproc ${RNPROC}->${NPROC}"
  [ "$PIDS" != "$RPIDS" ] && VOID="${VOID:+$VOID }pids [${RPIDS}]->[${PIDS}]"
  [ "$HASH" != "$RHASH" ] && VOID="${VOID:+$VOID }hash ${RHASH}->${HASH} (rebuilt mid-run)"
  DRIFT="$(pct_diff "$RSM" "$SMCLK")"
  # Idle P-state wobble (±15MHz at ~200MHz) is meaningless; void only on a
  # real excursion: >3% AND >100MHz absolute. Under load (~2.5GHz) the 3%
  # binds; at idle the floor absorbs P-state bounce.
  SMVOID=""
  if [ "$DRIFT" -ge 0 ] && [ "$DRIFT" -gt "$MAXDRIFT" ]; then
    ABS="$(( RSM > SMCLK ? RSM - SMCLK : SMCLK - RSM ))"
    case "$RSM$SMCLK" in ''|*[!0-9]*) SMVOID="non-numeric-clock";; *) [ "$ABS" -gt 100 ] && SMVOID="sm_clock ${RSM}->${SMCLK}MHz (${DRIFT}%>${MAXDRIFT}%, ${ABS}MHz>100MHz)";; esac
  fi
  [ -n "$SMVOID" ] && VOID="${VOID:+$VOID }$SMVOID"
  if [ "$NPROC" -gt "$ALLOW" ]; then VOID="${VOID:+$VOID }compute_procs=${NPROC}>${ALLOW}"; fi
  if [ -n "$VOID" ]; then echo "VOID $VOID"; exit 1; fi
  echo "SEALED drift=${DRIFT}%"
  exit 0
fi

echo "STAMP hash=$HASH mode=run nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
if [ "$NPROC" -gt "$ALLOW" ]; then echo "REFUSE compute_procs=${NPROC}>${ALLOW}"; exit 1; fi
exit 0
