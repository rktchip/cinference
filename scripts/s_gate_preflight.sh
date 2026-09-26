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
# Run identity = pinned exe sha256 (doc commits don't change the binary).
# HEAD travels as information only.
PIN_EXE="/root/ninfer-serve-e28cacb"
if [ -f "$PIN_EXE" ]; then
  HASH="$(sha256sum "$PIN_EXE" 2>/dev/null | cut -c1-12)"
  [ -z "$HASH" ] && HASH="unknown"
else
  HASH="no-pinned-exe"
fi
HEAD="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

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
# The 5090 drives the desktop: Explorer/Search/Widgets register as compute
# apps and come and go. Exclude known display noise (fail-closed: anything
# unknown still counts).
CAPPS_F="$(echo "$CAPPS" | grep -vi "explorer.exe\|SearchHost\|StartMenuExperienceHost\|WidgetBoard\|CrossDeviceResume\|ShellHost\|ApplicationFrameHost" || true)"
if [ -z "$(echo "$CAPPS_F" | tr -d ' \n\r')" ]; then
  NPROC=0
  PIDS="-"
else
  NPROC="$(echo "$CAPPS_F" | grep -c .)"
  PIDS="$(echo "$CAPPS_F" | tr '\n' ';')"
fi
NSERV="$(echo "$CAPPS" | grep -c 'ninfer-serve' || true)"

# Shared (sysmem-spill) check: adapter-total SharedUsage in MiB. Any paging
# voids the window (>100MB). Powershell takes seconds; pre/post only.
# NOTE: ref files must live under /root/ (proven persistent); /tmp does NOT
# persist across wsl.exe invocations (s1A/s1B refs lost, 2026-09-25).
SHARED_RAW="$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Measure-Object SharedUsage -Sum).Sum" 2>/dev/null | tr -d ' \r\n' || echo "")"
case "$SHARED_RAW" in ''|*[!0-9]*) SHARED=-1;; *) SHARED=$(( SHARED_RAW / 1048576 ));; esac
# Ghost-row correction (2026-09-26): dead processes leave stale SharedUsage
# rows with empty ProcessId (e.g. 1.48GB ghost at arm-C end stamp while the
# server held 80MB and 14GB dedicated sat free). Gate on live shared only.
GHOST_RAW="$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GpuProcessMemory | Where-Object { -not \$_.ProcessId } | Measure-Object SharedUsage -Sum | Select-Object -ExpandProperty Sum" 2>/dev/null | tr -d ' \r\n' || echo "")"
case "$GHOST_RAW" in ''|*[!0-9]*) GHOST=0;; *) GHOST=$(( GHOST_RAW / 1048576 ));; esac
if [ "$SHARED" -ge 0 ]; then
  SHARED=$(( SHARED - GHOST ))
  [ "$SHARED" -lt 0 ] && SHARED=0
fi

save_ref() {
  # $1=file
  {
    echo "hash=$HASH"
    echo "nproc=$NPROC"
    echo "pids=$PIDS"
    echo "smclk=$SMCLK"
    echo "temp=$TEMP"
    echo "shared=$SHARED"
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
  echo "STAMP hash=$HASH head=$HEAD mode=start nproc=$NPROC ninfer=$NSERV pids=[$PIDS] idle_util=[$S1,$S2,$S3]% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB shared=${SHARED}MiB"
  [ -n "$OUT" ] && save_ref "$OUT"
  FAIL=""
  [ "$NPROC" -gt "$ALLOW" ] && FAIL="compute_procs=${NPROC}>${ALLOW}"
  { [ "$SHARED" -ge 0 ] && [ "$SHARED" -gt 512 ]; } 2>/dev/null && FAIL="${FAIL:+$FAIL }shared=${SHARED}MiB>512MiB(spill)"
  for s in "$S1" "$S2" "$S3"; do
    case "$s" in ''|*[!0-9]*) continue;; esac
    [ "$s" -gt "$MAXIDLE" ] && FAIL="${FAIL:+$FAIL }idle_util=${s}>${MAXIDLE}"
  done
  if [ -n "$FAIL" ]; then echo "REFUSE $FAIL"; exit 1; fi
  echo "GO"
  exit 0
fi

if [ "$MODE" = "post" ]; then
  echo "STAMP hash=$HASH head=$HEAD mode=end nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB shared=${SHARED}MiB"
  if [ -z "$REF" ] || [ ! -f "$REF" ]; then echo "VOID no-start-stamp"; exit 1; fi
  RNPROC=""; RPIDS=""; RSM=""; RHASH=""; RSHARED=""
  while IFS='=' read -r k v; do
    case "$k" in
      nproc) RNPROC="$v";; pids) RPIDS="$v";; smclk) RSM="$v";; hash) RHASH="$v";; shared) RSHARED="$v";;
    esac
  done < "$REF"
  VOID=""
  # nproc/pids are INFO-ONLY (2026-09-26): WSL nvidia-smi cannot map PIDs
  # across the VM boundary (everything reads [Not Found]), so this leg can
  # never distinguish a co-tenant from display noise. Co-tenant guard is
  # idle_util (pre) + LOADCLK under load (client) instead.
  NOTE=""
  [ "$NPROC" != "$RNPROC" ] && NOTE="nproc ${RNPROC}->${NPROC}"
  [ "$PIDS" != "$RPIDS" ] && NOTE="${NOTE:+$NOTE }pids [${RPIDS}]->[${PIDS}]"
  [ -n "$NOTE" ] && echo "NOTE $NOTE"
  [ "$HASH" != "$RHASH" ] && VOID="${VOID:+$VOID }exe ${RHASH}->${HASH} (binary swapped mid-run)"
  case "$RSHARED" in ''|-1) ;; *) [ "$RSHARED" -gt 512 ] && VOID="${VOID:+$VOID }start-shared=${RSHARED}MiB>512MiB(spill)";; esac
  case "$SHARED" in -1) ;; *) [ "$SHARED" -gt 512 ] && VOID="${VOID:+$VOID }end-shared=${SHARED}MiB>512MiB(spill)";; esac
  DRIFT="$(pct_diff "$RSM" "$SMCLK")"
  # Idle clocks are INFO-ONLY (2026-09-26): P-state bounce of hundreds of MHz
  # at idle means nothing. Arm comparison uses LOADCLK under load (client).
  SMVOID=""
  if [ "$DRIFT" -ge 0 ] && [ "$DRIFT" -gt "$MAXDRIFT" ]; then
    ABS="$(( RSM > SMCLK ? RSM - SMCLK : SMCLK - RSM ))"
    case "$RSM$SMCLK" in ''|*[!0-9]*) SMVOID="non-numeric-clock";; *) [ "$ABS" -gt 100 ] && SMVOID="sm_clock ${RSM}->${SMCLK}MHz (${DRIFT}%>${MAXDRIFT}%, ${ABS}MHz>100MHz, idle-info-only)";; esac
  fi
  [ -n "$SMVOID" ] && echo "NOTE $SMVOID"
  if [ "$NPROC" -gt "$ALLOW" ]; then echo "NOTE compute_procs=${NPROC}>${ALLOW} (info-only, blind leg)"; fi
  if [ -n "$VOID" ]; then echo "VOID $VOID"; exit 1; fi
  echo "SEALED drift=${DRIFT}%"
  exit 0
fi

echo "STAMP hash=$HASH head=$HEAD mode=run nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB"
if [ "$NPROC" -gt "$ALLOW" ]; then echo "REFUSE compute_procs=${NPROC}>${ALLOW}"; exit 1; fi
exit 0
