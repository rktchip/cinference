#!/bin/sh
# scripts/s_gate_preflight.sh
# Mechanical single-server rule for all S-number timings (S1..S4).
# Usage:
#   s_gate_preflight.sh pre   [--max-idle-util PCT]   # before launching server: expect 0 compute procs
#   s_gate_preflight.sh stamp [--allow N]             # during/after run: print STAMP line, refuse if > N compute procs
# An S-line without a STAMP line is void by default.
# POSIX sh. Runs under WSL (nvidia-smi present). No GPU work itself.
set -u

MODE="${1:-stamp}"
ALLOW=1
MAXIDLE=5
if [ "$MODE" = "pre" ]; then ALLOW=0; fi
shift 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --allow) ALLOW="$2"; shift 2;;
    --max-idle-util) MAXIDLE="$2"; shift 2;;
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

# 3x1s idle samples catch display/compositor activity on the 5090
if [ "$MODE" = "pre" ]; then
  S1="$UTIL"
  sleep 1
  S2="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  sleep 1
  S3="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  echo "STAMP hash=$HASH mode=pre nproc=$NPROC ninfer=$NSERV pids=[$PIDS] idle_util=[$S1,$S2,$S3]% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
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

echo "STAMP hash=$HASH mode=run nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
if [ "$NPROC" -gt "$ALLOW" ]; then echo "REFUSE compute_procs=${NPROC}>${ALLOW}"; exit 1; fi
exit 0
