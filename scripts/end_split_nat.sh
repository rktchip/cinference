pkill -INT -f "ninfer-serve.*8903"
sleep 8
pgrep -f "ninfer-serve.*8903" || echo SERVER-DOWN
bash /mnt/c/src/cinference/scripts/s_gate_preflight.sh post --ref /root/split0.ref --out /root/split0.end
echo END_EXIT=$?
