#!/usr/bin/env bash
# bench_loop.sh GPU HOURS : opportunistic micro-benchmark sampler.
#  - every 2 min: sample util of GPU for 30 s; if mean util <= 15% and <=2 other procs -> run a "clean" bench immediately
#  - otherwise run a "contended" bench at most every 40 min (so we still get a spread)
#  - stops after HOURS or after 3 clean runs. Log: $R/logs/bench_loop_gpu<G>.log
set -u
GPU=$1; HOURS=${2:-6}; R=/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821
L=$R/logs/bench_loop_gpu$GPU.log; END=$(( $(date +%s) + HOURS*3600 )); last_contended=0; clean=0
export NO_MPS=${NO_MPS:-0}
echo "$(date -Is) LOOP_START gpu=$GPU hours=$HOURS no_mps=$NO_MPS" >> $L
while [ $(date +%s) -lt $END ] && [ $clean -lt 3 ]; do
  s=0; for i in 1 2 3 4 5 6; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU | tr -d ' '); s=$((s+u)); sleep 5; done; mu=$((s/6))
  np=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | wc -l)
  now=$(date +%s)
  if [ $mu -le 15 ] && [ $np -le 2 ]; then
    echo "$(date -Is) CLEAN_WINDOW util=$mu procs=$np -> bench" >> $L
    bash $R/scripts/run_bench.sh $GPU 200 640 480 >> $L 2>&1; clean=$((clean+1))
  elif [ $((now-last_contended)) -ge 2400 ]; then
    echo "$(date -Is) CONTENDED util=$mu procs=$np -> bench" >> $L
    bash $R/scripts/run_bench.sh $GPU 200 640 480 >> $L 2>&1; last_contended=$(date +%s)
  else
    sleep 90
  fi
done
echo "$(date -Is) LOOP_END clean_runs=$clean" >> $L
