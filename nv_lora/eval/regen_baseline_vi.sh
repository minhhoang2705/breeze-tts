#!/usr/bin/env bash
# Regenerate the nv01-8000 baseline's vi_clone_300 audio with THIS repo + breeze_tts env, so the
# reading-quality comparison against the LoRA checkpoints has no code/env confound. The older
# audio under anhnct/tts-eval/gen/ (anhnct's repo/env) is kept only as a cross-check.
# Runs ON THE LOGIN NODE: tmux new -d -s nv_lora_regen "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/regen_baseline_vi.sh > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/regen.log 2>&1"
# JOB is required (no default — shared-cluster job ids get repurposed, see nv_lora/env.sh):
#   JOB=22286 bash regen_baseline_vi.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
E=$REPO/nv_lora/eval
BASE=/mnt/data/users/anhnct/breeze-tts/outputs/exported/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k__checkpoint-8000
mkdir -p "$E/logs" "$E/gen"
srun --overlap --jobid="$JOB" env TMPDIR=/tmp/minhth21 bash -c "mkdir -p /tmp/minhth21 && cd $REPO && \
  $PY_BREEZE \
  $GENERATOR $BASE \
  $TE/evalsets/vi_clone_300.jsonl $E/gen/baseline_8000_regen__vi_clone_300 \
  --tag-style vlsp" >> "$E/logs/gen_baseline_8000_regen.log" 2>&1
echo "done baseline_8000_regen vi_clone_300"
