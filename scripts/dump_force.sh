#!/bin/sh
# scripts/dump_force.sh — aggregate forced-token decode scoring from a server log.
# Usage: dump_force.sh <server-log>
# Fail-closed scoring guard (2026-09-28): refuses (exit 1) unless the log
# contains the FORCE boot line "FORCE: N tokens loaded from <path>", printed
# at startup when NINFER_FORCE_TOKENS loads. A log without it means the
# server silently fell back to normal generation — scoring it would bless
# bad data (one full 7-leg loop was wasted this way). Never weaken this to
# a warning.
# Pre-launch companion: scripts/s_gate_preflight.sh wslpaths (aborts on
# C:/-style paths in NINFER_FORCE_TOKENS et al. before the server starts).
set -u

LOG="${1:-}"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "REFUSE dump_force: no such server log: ${LOG:-<empty>}" >&2
  echo "REFUSE dump_force: no such server log: ${LOG:-<empty>}"
  exit 1
fi

BOOT="$(grep -h '^FORCE: ' "$LOG" 2>/dev/null | head -n 1 || true)"
if [ -z "$(echo "$BOOT" | tr -d ' \n\r')" ]; then
  echo "REFUSE dump_force: $LOG lacks the FORCE boot line (server ran unforced; do not score)" >&2
  echo "REFUSE dump_force: $LOG lacks the FORCE boot line (server ran unforced; do not score)"
  exit 1
fi
echo "BOOT $BOOT"

FORCED="$(grep -c 'forced=' "$LOG" 2>/dev/null || true)"
TIE="$(grep -c 'TIE-CLASS' "$LOG" 2>/dev/null || true)"
NOTTIE="$(grep -c 'NOT-TIE' "$LOG" 2>/dev/null || true)"
EXH="$(grep -c 'EXHAUSTED' "$LOG" 2>/dev/null || true)"
FATAL="$(grep -c 'FATAL' "$LOG" 2>/dev/null || true)"
echo "SUMMARY log=$LOG forced_steps=$FORCED tie_class=$TIE not_tie=$NOTTIE exhausted=$EXH fatal=$FATAL"
if [ "$EXH" != "0" ] && [ -n "$EXH" ]; then
  echo "NOTE force ids exhausted mid-run ($EXH rows kept argmax)"
fi
exit 0
