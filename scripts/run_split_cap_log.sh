# Self-gate (2026-09-28): never launch without a preflight GO (port/VRAM/idle/paths).
_GATE="$(bash "$(dirname "$0")/s_gate_preflight.sh" pre 2>&1 | tail -n 1)"
[ "$_GATE" = "GO" ] || { echo "LAUNCH-REFUSED (no preflight GO): $_GATE" >&2; exit 1; }
bash /mnt/c/src/cinference/scripts/run_split_oracle.sh \
  /root/cinference-split-build/apps/ninfer-serve "$1" "$2"
