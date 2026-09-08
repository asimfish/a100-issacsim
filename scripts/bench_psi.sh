#!/usr/bin/env bash
# bench_psi.sh GPU NODE -- wait for bench_all to finish, then run project RL modes
GPU=$1; NODE=$2; R=/data/safelab_fr3_eval_20260819/a100_bench
while ! grep -q ALL_DONE $R/$NODE.all.log 2>/dev/null; do sleep 30; done
while pgrep -f "bench_isaac.sh rl_cartpole_cam" >/dev/null; do sleep 30; done
for m in rl_psi_state rl_psi_rgb; do bash $R/bench_isaac.sh $m $GPU $NODE; done >> $R/$NODE.all.log 2>&1
echo PSI_DONE >> $R/$NODE.all.log
