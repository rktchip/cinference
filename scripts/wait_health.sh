for i in $(seq 1 24); do
  sleep 5
  if curl -sf http://127.0.0.1:8902/health; then
    echo WAIT-HEALTH-LISTENING
    break
  fi
done
tail -c 400 /root/s1Bleg.log
echo
