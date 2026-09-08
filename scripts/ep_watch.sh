#!/usr/bin/env bash
# ep_watch.sh NODE -- log wall-clock time at which each "[MP Episode k/10]" line appears in collect10/cmd.log
R=/data/safelab_fr3_eval_20260819/a100_bench; N=$1; F=$R/$N/collect10/cmd.log; O=$R/$N/collect10_ep_times.txt; last=-1
for i in $(seq 1 5400); do
  c=$(grep -ac "^\[MP Episode" "$F" 2>/dev/null || echo 0)
  [ "$c" != "$last" ] && { echo "$(date +%s) episodes_done=$c" >> "$O"; last=$c; }
  [ -f "$R/$N/collect10/timing.txt" ] && grep -q "^T1=" "$R/$N/collect10/timing.txt" 2>/dev/null && break
  sleep 2
done
