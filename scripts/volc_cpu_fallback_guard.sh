#!/usr/bin/env bash
# volc_cpu_fallback_guard.sh - kill evals whose Isaac Kit lost the GPU (Vulkan device init failed -> PhysX on CPU,
# or PhysX CUDA errors flooding). Such runs keep going silently and would write a wrong SSR. After killing, drop the
# empty RESULT row and age the log so the owning queue relaunches the candidate.
set -u
R=/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821; OUT=$R/VOLC_SCORES.tsv; G=$R/logs/cpu_fallback_guard.log
PAT="switching to software|Failed to create any GPU devices|vkCreateDevice failed"
while true; do
  for p in $(pgrep -f "^[^ ]*python[^ ]* [^ ]*play.py"); do
    lp=$(tr "\0" "\n" < /proc/$p/cmdline 2>/dev/null | grep -A1 "^--log_path$" | tail -1); [ -n "$lp" ] || continue
    C=$(basename "$lp"); L=$R/logs/$C.log; [ -f "$L" ] || continue
    n1=$(grep -acE "$PAT" "$L"); n2=$(grep -ac "PhysX CUDA error" "$L")
    if [ "$n1" -gt 0 ] || [ "$n2" -gt 1000 ]; then
      echo "$(date -Is) KILL $C pid=$p devfail=$n1 physx_cuda_err=$n2" >>"$G"
      kill "$p" 2>/dev/null; sleep 20; kill -9 "$p" 2>/dev/null; sleep 30
      mv -f "$L" "$L.cpufallback.$(date +%s)"
      awk -F"\t" -v c="$C" '!($1==c && $4 !~ /SSR:/)' "$OUT" > "$OUT.tmp.$$" && cat "$OUT.tmp.$$" > "$OUT"; rm -f "$OUT.tmp.$$"
      echo "$(date -Is) requeued $C (row dropped, log aged)" >>"$G"
    fi
  done
  sleep 60
done
