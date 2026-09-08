#!/usr/bin/env bash
# bench_all.sh GPU NODE  -- run all modes sequentially on one GPU
GPU=$1; NODE=$2; R=/data/safelab_fr3_eval_20260819/a100_bench
for m in eval10 collect10 rl_cartpole rl_franka rl_cartpole_cam; do bash $R/bench_isaac.sh $m $GPU $NODE; done >> $R/$NODE.all.log 2>&1
echo ALL_DONE >> $R/$NODE.all.log
