#!/usr/bin/env bash
# run_bench.sh GPU [FRAMES] [CAM_W] [CAM_H] : run bench_render.py with the exact runtime env of the eval scripts.
# env NO_MPS=1 -> point the CUDA client at a non-existent MPS pipe dir so it bypasses a running MPS daemon (compute mode Default).
set -u
GPU=${1:-0}; FRAMES=${2:-300}; W=${3:-640}; H=${4:-480}
EVALROOT=/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821
BASE=/mnt/nas/data/safelab/retreat/lyf-back/chembench
PATCHED=$EVALROOT/patched
PY="$BASE/.conda_envs/chembench_isaacsim51/bin/python"
export CONDA_PREFIX="$BASE/.conda_envs/chembench_isaacsim51"
export PATH="$CONDA_PREFIX/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$EVALROOT/pydeps_modern:$PATCHED:$BASE:$BASE/psilab/source/psilab:$BASE/psilab/source/psilab_tasks:$BASE/psilab/source/isaaclab:$BASE/psilab/source/isaaclab_assets:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES CHEMBENCH_DISABLE_PARTICLE_FLUIDS=1 CHEMBENCH_PHYSX_PROFILE=balanced_2048 CHEMBENCH_RENDER_PROFILE=quality
export CHEMBENCH_GPU_PARTICLE_COLLISION_MODE=off   # same as the eval protocol (nosdf): no GPU-particle SDF colliders
export CUDA_VISIBLE_DEVICES=$GPU
if [[ "${NO_MPS:-0}" == "1" ]]; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-$$; fi
MPS=$(pgrep -c -x nvidia-cuda-mps-control)
TAG=$(hostname -s)_gpu${GPU}_${W}x${H}_$(date +%m%d_%H%M%S)$([[ "${NO_MPS:-0}" == "1" ]] && echo _nomps)
OUT=$EVALROOT/logs/bench_render_$TAG
export BENCH_META="procs_on_gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $GPU | wc -l) util_before=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i $GPU) mps_daemon=$MPS no_mps=${NO_MPS:-0} load1=$(cut -d' ' -f1 /proc/loadavg)"
echo "host=$(hostname) gpu=$GPU $BENCH_META" | tee $OUT.meta
timeout 1800 "$PY" $EVALROOT/scripts/bench_render.py --headless --enable_cameras --device cuda:0 --frames $FRAMES --cam_w $W --cam_h $H --out $OUT.json \
  --experience "$BASE/psilab/apps/isaaclab.python.headless.rendering.isaac51.kit" > $OUT.log 2>&1 &
PID=$!
# background-load sampler (5 s): util of this GPU, #processes sharing it, loadavg1 -> $OUT.gpu_samples.csv
( while kill -0 $PID 2>/dev/null; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' '),$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | wc -l),$(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $OUT.gpu_samples.csv 2>/dev/null &
SAMP=$!
# fail fast if Kit cannot create a Vulkan device (nothing useful can be measured after that)
while kill -0 $PID 2>/dev/null; do
  if grep -a -q -E "vkCreateDevice failed|Failed to create any GPU devices" $OUT.log 2>/dev/null; then
    echo "RENDER_DEVICE_FAIL $(grep -a -m1 -E 'CUDA being in bad state|vkCreateDevice failed|MPS server' $OUT.log)" | tee -a $OUT.meta
    pkill -9 -P $PID; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; echo "rc=killed"; exit 3
  fi
  sleep 5
done
wait $PID; echo "rc=$?" | tee -a $OUT.meta
kill $SAMP 2>/dev/null
# summarize background load over the *measurement* window (last 40% of samples ~ after Kit start-up)
python3 - "$OUT.gpu_samples.csv" <<'PY' | tee -a $OUT.meta
import sys,csv
rows=[r for r in csv.reader(open(sys.argv[1])) if len(r)==5]
tail=rows[int(len(rows)*0.6):] or rows
if tail:
    u=[float(r[1]) for r in tail]; m=[float(r[2]) for r in tail]; p=[float(r[3]) for r in tail]; l=[float(r[4]) for r in tail]
    print(f"bg_during_measure: util_mean={sum(u)/len(u):.0f}% util_min={min(u):.0f}% mem_used_mean={sum(m)/len(m)/1024:.1f}GB procs_mean={sum(p)/len(p):.1f} load1_mean={sum(l)/len(l):.0f} samples={len(tail)}")
PY
grep -a -c "switching to software" $OUT.log | sed 's/^/physx_sw_fallback_lines=/' | tee -a $OUT.meta
grep -a "\[bench\]" $OUT.log | tail -5; ls $OUT.json 2>/dev/null
