grep -a "s1Bleg-B" /root/s1Bleg.log | grep -av "PROMPTS\|DONE" > /tmp/armB_client.txt
wc -l /tmp/armB_client.txt
python3 /mnt/c/src/cinference/scripts/trajmeans.py /root/s1Bleg.log --hist fox fox fox paris paris paris city city city 2>&1 | tail -n 18
