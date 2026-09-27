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
    --port) PORT_ARG="$2"; shift 2;;
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
# Port rule (2026-09-26): a differently named binary can survive a
# name-based kill and share :8902, silently cross-serving requests.
# Attribute by socket, not by name: pre refuses on any holder.
PORT="${PORT:-8902}"
[ -n "${PORT_ARG:-}" ] && PORT="$PORT_ARG"
PORT_HOLDERS="$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -o 'users:(("[^"]*",pid=[0-9]*[^)]*))' || true)"
if [ -z "$(echo "$PORT_HOLDERS" | tr -d ' \n\r')" ]; then
  NPORT=0
else
  NPORT="$(echo "$PORT_HOLDERS" | grep -c .)"
fi

# Shared (sysmem-spill) check: adapter-total SharedUsage in MiB.
# Powershell takes seconds; pre/post only.
# NOTE: ref files must live under /root/ (proven persistent); /tmp does NOT
# persist across wsl.exe invocations (s1A/s1B refs lost, 2026-09-25).
# Ghost rule (2026-09-26): subtract ONLY known ghosts (empty-PID rows already
# present at preflight, saved to ref). Any NEW row counts (fail-closed: a live
# WSL co-tenant can also lack a mappable PID). Single PS call returns
# "total_shared ghost_shared max_dedicated_row".
MEMQ="$(powershell.exe -NoProfile -Command "\$p=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GpuProcessMemory; \$t=(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Measure-Object SharedUsage -Sum).Sum; \$g=(\$p | Where-Object { -not \$_.ProcessId } | Measure-Object SharedUsage -Sum).Sum; \$m=(\$p | Measure-Object DedicatedUsage -Maximum).Maximum; \"\$t \$g \$m\"" 2>/dev/null | tr -d '\r\n' || echo "")"
SHARED_RAW="$(echo "$MEMQ" | cut -d' ' -f1)"
GHOST_RAW="$(echo "$MEMQ" | cut -d' ' -f2)"
SRVROW_RAW="$(echo "$MEMQ" | cut -d' ' -f3)"
case "$SHARED_RAW" in ''|*[!0-9]*) SHARED=-1;; *) SHARED=$(( SHARED_RAW / 1048576 ));; esac
case "$GHOST_RAW" in ''|*[!0-9]*) GHOST_KNOWN=0;; *) GHOST_KNOWN=$(( GHOST_RAW / 1048576 ));; esac
case "$SRVROW_RAW" in ''|*[!0-9]*) SRVROW=-1;; *) SRVROW=$(( SRVROW_RAW / 1048576 ));; esac

save_ref() {
  # $1=file
  {
    echo "hash=$HASH"
    echo "nproc=$NPROC"
    echo "pids=$PIDS"
    echo "smclk=$SMCLK"
    echo "temp=$TEMP"
    echo "ghost=$GHOST_KNOWN"
    echo "ded0=$MEMUSED"
    echo "shared_raw=$LIVE"
  } > "$1"
}

pct_diff() {
  # $1=old $2=new -> integer percent |new-old|/old, or -1 if non-numeric
  case "$1$2" in ''|*[!0-9]*) echo -1; return;; esac
  if [ "$1" -eq 0 ]; then echo -1; return; fi
  d=$(( $2 > $1 ? $2 - $1 : $1 - $2 ))
  echo $(( d * 100 / $1 ))
}

# Live shared: at pre, all current ghosts are pre-existing by definition.
LIVE=$(( SHARED - GHOST_KNOWN ))
[ "$LIVE" -lt 0 ] && LIVE=0
if [ "$MODE" = "pre" ]; then
  S1="$UTIL"
  sleep 1
  S2="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  sleep 1
  S3="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  echo "STAMP hash=$HASH head=$HEAD mode=start nproc=$NPROC ninfer=$NSERV pids=[$PIDS] idle_util=[$S1,$S2,$S3]% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB shared_live=${LIVE}MiB(ghosts=${GHOST_KNOWN}) srvrow=${SRVROW}MiB"
  [ -n "$OUT" ] && save_ref "$OUT"
  FAIL=""
  [ "$NPORT" -gt 0 ] && FAIL="port_${PORT}_held=[${PORT_HOLDERS}]"
  [ "$NPROC" -gt "$ALLOW" ] && FAIL="${FAIL:+$FAIL }compute_procs=${NPROC}>${ALLOW}"
  { [ "$LIVE" -gt 512 ]; } 2>/dev/null && FAIL="${FAIL:+$FAIL }shared_live=${LIVE}MiB>512MiB(spill)"
  for s in "$S1" "$S2" "$S3"; do
    case "$s" in ''|*[!0-9]*) continue;; esac
    [ "$s" -gt "$MAXIDLE" ] && FAIL="${FAIL:+$FAIL }idle_util=${s}>${MAXIDLE}"
  done
  if [ -n "$FAIL" ]; then echo "REFUSE $FAIL"; exit 1; fi
  echo "GO"
  exit 0
fi

#   s_gate_preflight.sh teardown [--port P]  # kill by PORT, not by name:
#     SIGINT every :8902 holder, wait, verify free, SIGKILL stragglers.
#     A name-based pkill can miss a differently named binary (2026-09-26:
#     verify-8a1c90c survived a rung2-pattern kill and co-served :8902).
if [ "$MODE" = "teardown" ]; then
  for round in 1 2; do
    PIDS="$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
    [ -z "$(echo "$PIDS" | tr -d ' \n\r')" ] && { echo "PORT-FREE :$PORT"; exit 0; }
    SIG="INT"; [ "$round" -eq 2 ] && SIG="KILL"
    for p in $PIDS; do echo "TEARDOWN -$SIG $p"; kill "-$SIG" "$p" 2>/dev/null || true; done
    sleep 8
  done
  LEFT="$(ss -ltnp 2>/dev/null | grep ":$PORT" || true)"
  if [ -n "$(echo "$LEFT" | tr -d ' \n\r')" ]; then
    echo "TEARDOWN-FAILED :$PORT still held: $LEFT"; exit 1
  fi
  echo "PORT-FREE :$PORT"
  exit 0
fi

if [ "$MODE" = "post" ]; then
  echo "STAMP hash=$HASH head=$HEAD mode=end nproc=$NPROC ninfer=$NSERV pids=[$PIDS] util=${UTIL}% memutil=${MEMU}% temp=${TEMP}C sm=${SMCLK}MHz memclk=${MEMCLK}MHz vram=${MEMUSED}/${MEMTOTAL}MiB shared=${SHARED}MiB"
  if [ -z "$REF" ] || [ ! -f "$REF" ]; then echo "VOID no-start-stamp"; exit 1; fi
  RNPROC=""; RPIDS=""; RSM=""; RHASH=""; RGHOST=""; RDED0=""; RSHARED_RAW=""
  while IFS='=' read -r k v; do
    case "$k" in
      nproc) RNPROC="$v";; pids) RPIDS="$v";; smclk) RSM="$v";; hash) RHASH="$v";; ghost) RGHOST="$v";; ded0) RDED0="$v";; shared_raw) RSHARED_RAW="$v";;
    esac
  done < "$REF"
  # Post live-shared: subtract ONLY ref-known ghosts; any new row counts.
  case "$RGHOST" in ''|*[!0-9]*) RGHOST=0;; esac
  LIVEE=$(( SHARED - RGHOST ))
  [ "$LIVEE" -lt 0 ] && LIVEE=0
  echo "POST shared_live=${LIVEE}MiB(known_ghosts=${RGHOST}) srvrow=${SRVROW}MiB ded0=${RDED0}"
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
  case "$RSHARED_RAW" in ''|-1) ;; *) [ "$RSHARED_RAW" -gt 512 ] && VOID="${VOID:+$VOID }start-shared-live=${RSHARED_RAW}MiB>512MiB(spill, ghosts already removed at pre)";; esac
  case "$SHARED" in -1) ;; *) [ "$LIVEE" -gt 512 ] && VOID="${VOID:+$VOID }end-shared-live=${LIVEE}MiB>512MiB(spill)";; esac
  # Memory co-tenant check (no PIDs needed): adapter dedicated, minus the
  # server row (max dedicated row), minus the pre display baseline, must not
  # grow >1GB during the run. Catches a mid-run co-tenant that idle-util and
  # load clocks can miss.
  case "$RDED0/$SRVROW/$MEMUSED" in *[!0-9/]*|*/) COCHECK="non-numeric-mem";; *)
    COGROW=$(( MEMUSED - SRVROW - RDED0 ))
    [ "$COGROW" -gt 1024 ] && VOID="${VOID:+$VOID }dedicated-grew=${COGROW}MiB>1024MiB(co-tenant? ded ${RDED0}->${MEMUSED} minus srvrow ${SRVROW})"
  ;; esac
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
