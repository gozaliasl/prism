#!/bin/bash
# Wait for PID $1 (300px), then start 1arcmin
WAIT_PID=${1:?}
echo "[chain] waiting for 300px PID $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[chain] 300px finished; starting 1arcmin"
nohup /Volumes/exthd-prism/prism-lensing/scripts/run_hydris_50_1arcmin.sh &
echo "[chain] 1arcmin PID $!"
