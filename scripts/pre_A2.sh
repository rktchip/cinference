pkill -INT -f ninfer-serve-e28cacb
sleep 8
pgrep -f ninfer-serve-e28cacb || echo SERVER-DOWN
bash /mnt/c/src/cinference/scripts/s_gate_preflight.sh pre --out /root/s1A2.ref
