#!/usr/bin/env bash
# Export LoRA checkpoints and generate eval audio for them plus the init-checkpoint baseline.
# Runs ON THE LOGIN NODE (start detached: tmux new -d -s nv_lora_eval "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/run_gen.sh > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/run_gen.log 2>&1");
# work goes into job $JOB via srun --overlap. Resumable: exports with export_info.json
# are reused and breeze_generate.py skips wavs already written. Exports go next to the
# team's other exports in anhnct/breeze-tts/outputs/exported (group ttsdata, g+rw).
#
# Generation mirrors the team pipeline (breeze-tts-remote/eval_breeze.sh): breeze_generate.py,
# zero-shot clone per row, seed 42, VLSP tag style, no --fast — the same settings that
# produced the baseline's vi_clone_300 audio under tts-eval/gen/, which is reused as-is.
#
# RUN/BASE/BASE_NAME/BASE_VI_WAVS/STEPS/TRAIN_DIR/JOB are overridable env vars (defaults
# below reproduce nv_lora01, exporting onto nv01 checkpoint-8000). Shared path config
# (REPO/PY_BREEZE/GENERATOR/EXPORTS/...) lives in nv_lora/env.sh; JOB is required, no
# default (shared-cluster job ids get repurposed — see env.sh). For nv_lora02 (from vi33k
# checkpoint-33000), set e.g.:
#   JOB=22286 RUN=nv_lora02_p1-r32_from-vi33k_remap-v3-46k \
#   BASE=/mnt/data/users/anhnct/breeze-tts/outputs/exported/vi_all_20k_full_p2__checkpoint-33000 \
#   BASE_NAME=baseline_vi33k STEPS="4400 4800 5200 5600" bash run_gen.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
E=$REPO/nv_lora/eval
PY=$PY_BREEZE
RUN="${RUN:-nv_lora01_p1-r32_from-nv01-8k_vieneu-tc4.4k}"
BASE="${BASE:-/mnt/data/users/anhnct/breeze-tts/outputs/exported/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k__checkpoint-8000}"
BASE_NAME="${BASE_NAME:-baseline_8000}"
TRAIN_DIR="${TRAIN_DIR:-$REPO/outputs/$RUN}"
BASE_VI_WAVS="${BASE_VI_WAVS:-/mnt/data/users/anhnct/tts-eval/gen/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k__checkpoint-8000__vi_clone_300}"
VI_CLONE=/mnt/data/users/anhnct/tts-eval/evalsets/vi_clone_300.jsonl
read -ra STEPS <<< "${STEPS:-200 400 600 800}"

mkdir -p "$E/logs" "$E/gen"
python3 "$E/make_evalsets.py" "$E/evalsets"

srun --overlap --jobid="$JOB" env TMPDIR=/tmp/minhth21 bash -s <<EOF
set -euo pipefail
mkdir -p /tmp/minhth21
cd $REPO
for step in ${STEPS[*]}; do
  out=$EXPORTS/${RUN}__checkpoint-\$step
  if [[ -f \$out/export_info.json ]]; then echo "reusing \$out"; continue; fi
  if [[ -e \$out ]]; then echo "incomplete export at \$out; inspect and remove it by hand" >&2; exit 1; fi
  CUDA_VISIBLE_DEVICES= $PY -m breeze_train.export $BASE $TRAIN_DIR/checkpoint-\$step \$out \
    --policy p1 --name ${RUN}-checkpoint-\$step > $E/logs/export_\$step.log 2>&1
  chgrp -R ttsdata \$out && chmod -R g+rwX \$out
  echo "exported \$out"
done

gen() {  # gen NAME MODEL_DIR EVALSET...
  local name=\$1 model=\$2; shift 2
  for evalset in "\$@"; do
    local file=$E/evalsets/\$evalset.jsonl
    [[ \$evalset == vi_clone_300 ]] && file=$VI_CLONE
    $PY $GENERATOR \$model \$file $E/gen/\${name}__\$evalset --tag-style vlsp \
      >> $E/logs/gen_\$name.log 2>&1
    echo "done \$name \$evalset (\$(ls $E/gen/\${name}__\$evalset | grep -c '\.wav\$') wavs)"
  done
}
pids=()
gen $BASE_NAME $BASE tc_val_99 tc_clone_100 & pids+=(\$!)
for step in ${STEPS[*]}; do
  gen lora_\$step $EXPORTS/${RUN}__checkpoint-\$step tc_val_99 tc_clone_100 vi_clone_300 & pids+=(\$!)
done
rc=0; for p in "\${pids[@]}"; do wait \$p || rc=1; done
exit \$rc
EOF
ln -sfn "$BASE_VI_WAVS" "$E/gen/${BASE_NAME}__vi_clone_300"
echo "all generation finished"
