pkill -INT -f ninfer-serve-e28cacb
sleep 12
pgrep -f ninfer-serve-e28cacb || echo SERVER-DOWN
ls -la /root/speccoff_node* 2>&1
bash /mnt/c/src/cinference/scripts/s_gate_preflight.sh post --ref /root/s1Cap.ref --out /root/s1Cap.end
