#!/bin/bash
# Fetch episode videos for the same checkpoint from volc (A100) and bjxy (5090) into ./pairs
set -u
cd ~/Desktop/research/volc_render_check/pairs || exit 1
V=/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821/videos
while read -r tag vc bc; do
  [ -n "$tag" ] || continue
  for ep in 001 002 003; do
    [ -f ${tag}_ep${ep}_volc.mp4 ] || scp -q volc-a100:$V/$vc/episode_$ep.mp4 ./${tag}_ep${ep}_volc.mp4 2>/dev/null
    [ -f ${tag}_ep${ep}_bjxy.mp4 ] || scp -q bjxy_5090:$V/$bc/episode_$ep.mp4 ./${tag}_ep${ep}_bjxy.mp4 2>/dev/null
  done
  echo "$tag: $(ls ${tag}_* 2>/dev/null | wc -l) files"
done <<'EOF'
alcohol_lamp16k_s3952 i224-act-rlb-alcohol_lamp16k-s3952 bjxy-rlb16k-alcohol_lamp-s3952
clear_vol16k_s3951 i224-act-rlb-clear_volumetric_flask_250ml16k-s3951 bjxy-rlb16k-clear_volumetric_flask_250ml-s3951
crbL16k_s3951 i224-act-rlb-clear_reagent_bottle_large16k-s3951 bjxy-rlb16k-clear_reagent_bottle_large-s3951
brown_vol16k_s3951 i224-act-rlb-brown_volumetric_flask_250ml16k-s3951 bjxy-rlb16k-brown_volumetric_flask_250ml-s3951
erlenmeyer_ao_s3961 i224-act-rlb-erlenmeyer_flask__ao_8k-s3961 bjxy-rlb8k-erlenmeyer_flask__ao_-s3961
EOF
ls -la | awk '{print $5, $9}' | grep mp4 | head -40
