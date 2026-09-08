#!/usr/bin/env bash
# bench_isaac.sh MODE GPU NODE  -- identical workload on A100 (volc) and RTX 5090 (bjxy); one process per GPU.
# MODE: eval10 | collect10 | rl_cartpole | rl_franka | rl_cartpole_cam
# Writes: $R/<NODE>/<MODE>/{cmd.log, gpu_samples.csv, load.txt, timing.txt}
set -uo pipefail
MODE=$1; GPU=$2; NODE=$3
BASE=/mnt/nas/data/safelab/retreat/lyf-back/chembench
EVALROOT=/data/safelab_fr3_eval_20260819
PATCHED=$EVALROOT/psibot400demo_eval_20260821/patched/psilab_tasks
R=$EVALROOT/a100_bench; OUT=$R/$NODE/$MODE; rm -rf "$OUT"; mkdir -p "$OUT/cache/torch_ext" "$OUT/tmp" "$OUT/video" "$OUT/run" "$OUT/lerobot"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | head -1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU
export CONDA_PREFIX=$BASE/.conda_envs/chembench_isaacsim51
export PATH=$CONDA_PREFIX/bin:/home/liyufeng/miniforge3/envs/chembench/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH="$EVALROOT/pydeps_modern:$PATCHED:$BASE:$BASE/diffusion_policy:$BASE/psilab/source/psilab:$BASE/psilab/source/psilab_tasks:$BASE/psilab/source/isaaclab:$BASE/psilab/source/isaaclab_assets:$BASE/psilab/source/isaaclab_tasks:$BASE/psilab/source/isaaclab_rl:$BASE/psilab/source/isaaclab_mimic:$BASE/curobo/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export XDG_CACHE_HOME=$OUT/cache TORCH_EXTENSIONS_DIR=$OUT/cache/torch_ext TMPDIR=$OUT/tmp TORCH_CUDA_ARCH_LIST=$CC MAX_JOBS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CHEMBENCH_TIMM_PRETRAINED_OFF=1 CHEMBENCH_TIMM_PRETRAINED_FALLBACK=1 CHEMBENCH_DISABLE_PARTICLE_FLUIDS=1 CHEMBENCH_SAVE_FAILED_LEROBOT=0
export CHEMBENCH_EVAL_VIDEO_CRF=18 CHEMBENCH_EVAL_VIDEO_PRESET=fast CHEMBENCH_PHYSX_PROFILE=balanced_2048 CHEMBENCH_RENDER_PROFILE=quality
PY=$CONDA_PREFIX/bin/python; KIT=$BASE/psilab/apps/isaaclab.python.headless.rendering.isaac51.kit
HP=/home/liyufeng/hp2_ckpts/act-rlb-clear_reagent_bottle_large16k-s3951-ppu5-v1
OVR='{"finger_grasp_mode":"full","finger_close_scale":1.0,"lift_height_desired":0.30,"min_success_lift_ratio":0.75}'
case $MODE in
  eval10) CMD=("$PY" "$BASE/psilab/scripts_psi/workflows/imitation_learning/play.py" --task Psi-IL-Grasp-v2 --runner_type act --checkpoint "$HP/policy_best.ckpt" --num_envs 1 --seed 42 --scene room_cfg:PSI_DC_Grasp_CFG --object_name clear_reagent_bottle_large --obs_mode rgb --camera_names chest_camera head_camera third_camera --max_episode 10 --episode_length_s 6.0 --enable_random --env_overrides '{"random_position_offset_range":[0.01,0.01,0.0]}' --enable_eval --record_video --video_episodes 10 --video_length 200 --video_min_frames 30 --video_folder "$OUT/video" --video_camera third_camera --strict_complete_eval --log_path "$OUT/run" --disable_particle_fluids --headless --enable_cameras --device cuda:0 --experience "$KIT" --num_queries 8 --temporal_agg false);;
  collect10) CMD=("$PY" "$BASE/psilab/scripts_psi/workflows/motion_planning/play.py" --task Psi-MP-Grasp-v2 --num_envs 1 --seed 14101 --scene room_cfg:PSI_DC_Grasp_CFG --object_name glass_beaker_100ml --enable_random --enable_cameras --enable_eval --sample_step 4 --max_episode 10 --target_success_count 10 --enable_lerobot --lerobot_output_dir "$OUT/lerobot" --lerobot_task "grasp glass_beaker_100ml" --lerobot_fps 30 --camera_names chest_camera head_camera third_camera --env_overrides "$OVR" --disable_particle_fluids --headless --experience "$KIT");;
  rl_cartpole) CMD=("$PY" "$R/rl_train.py" --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --max_iterations 10 --seed 1 --headless --experience "$BASE/psilab/apps/isaaclab.python.headless.kit");;
  rl_franka) CMD=("$PY" "$R/rl_train.py" --task Isaac-Franka-Cabinet-Direct-v0 --num_envs 4096 --max_iterations 10 --seed 1 --headless --experience "$BASE/psilab/apps/isaaclab.python.headless.kit");;
  rl_cartpole_cam) CMD=("$PY" "$R/rl_train.py" --task Isaac-Cartpole-RGB-Camera-Direct-v0 --num_envs 128 --max_iterations 10 --seed 1 --headless --enable_cameras --experience "$KIT" "--kit_args=--/persistent/isaac/asset_root/cloud=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1");;
  rl_psi_state) export CHEMBENCH_DISABLE_SCENE_RANDOM=1; CMD=("$PY" "$BASE/psilab/scripts_psi/workflows/reinforcement_learning/rl_games/train.py" --task Psi-Direct-RL-Grasp-Beaker100-v1 --num_envs 64 --object_name glass_beaker_100ml --max_epoch 10 --batch_size 512 --learning_rate 3e-4 --mini_epochs 2 --disable_tensorboard --disable_mixed_precision --headless --disable_particle_fluids --log_path "$OUT/run" --experience "$KIT");;
  rl_psi_rgb) export CHEMBENCH_DISABLE_SCENE_RANDOM=1 CHEMBENCH_DIRECT_RL_OBS_MODE=rgb_state CHEMBENCH_TILED_CAMERA_WIDTH=224 CHEMBENCH_TILED_CAMERA_HEIGHT=224; CMD=("$PY" "$BASE/psilab/scripts_psi/workflows/reinforcement_learning/rl_games/train.py" --task Psi-Direct-RL-Grasp-Beaker100-v1 --num_envs 64 --object_name glass_beaker_100ml --max_epoch 10 --batch_size 512 --learning_rate 3e-4 --mini_epochs 2 --disable_tensorboard --disable_mixed_precision --headless --enable_cameras --disable_particle_fluids --log_path "$OUT/run" --experience "$KIT");;
  *) echo "bad mode"; exit 2;;
esac
cd "$OUT"
# background load samplers
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' '),$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"),$(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > "$OUT/gpu_samples.csv" 2>/dev/null &
SP=$!
{ echo "node=$NODE gpu=$GPU cc=$CC mode=$MODE"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU"; nproc; uptime; } > "$OUT/load.txt"
T0=$(date +%s.%N); echo "T0=$T0 $(date -Is)" > "$OUT/timing.txt"
printf '%q ' "${CMD[@]}" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"
timeout -k 60s 7200s "${CMD[@]}" > "$OUT/cmd.log" 2>&1; rc=$?
T1=$(date +%s.%N); echo "T1=$T1 $(date -Is) rc=$rc wall=$(python3 -c "print(round($T1-$T0,1))")" >> "$OUT/timing.txt"
kill $SP 2>/dev/null; uptime >> "$OUT/load.txt"
# per-episode markers from video / lerobot file mtimes
for f in "$OUT"/video/*.mp4 "$OUT"/lerobot/*/videos/*/third_camera/*.mp4 "$OUT"/lerobot/*/data/*/*.parquet; do [ -f "$f" ] && printf "%s %s\n" "$(stat -c %Y.%3N "$f" 2>/dev/null || stat -c %Y "$f")" "$(basename "$f")"; done | sort -n >> "$OUT/timing.txt" 2>/dev/null
grep -aE "fps step|epoch:|Iteration|success|SSR|Traceback|Error" "$OUT/cmd.log" | tail -60 > "$OUT/summary.log"
echo "DONE $MODE rc=$rc wall=$(python3 -c "print(round($T1-$T0,1))")s"
