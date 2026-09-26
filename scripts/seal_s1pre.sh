python3 /mnt/c/src/cinference/scripts/trajmeans.py /root/s1A2.log --hist fox fox fox paris paris paris city city city 2>&1 | tail -n 12
pkill -INT -f ninfer-serve-e28cacb
sleep 8
pgrep -f ninfer-serve-e28cacb || echo SERVER-DOWN
