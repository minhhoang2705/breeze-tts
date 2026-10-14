#!/usr/bin/env bash
# Export nv_lora03b (spkfix) checkpoints 160/320/480 onto the exported checkpoint-4400 base,
# then generate eval audio for tc_val_99/tc_clone_100/sn_clone_100/br_clone_100/vi_clone_300.
# Run inside the Slurm allocation (this script is handed to srun by the caller).
set -euo pipefail
PY=/mnt/data/users/minhth21/conda_env/miniconda3/envs/breeze_tts/bin/python
GENERATOR=/mnt/data/users/anhnct/breeze-tts-remote/remote/breeze_generate.py
EXPORTS=/mnt/data/users/anhnct/breeze-tts/outputs/exported
REPO=/mnt/data/users/minhth21/breeze-tts
E=$REPO/nv_lora/eval
RUN=nv_lora03b_p1-r32_from-nvlora02-4400_exp10-1.8k-ovs4x-spkfix
BASE=$EXPORTS/nv_lora02_p1-r32_from-vi33k_remap-v3-46k__checkpoint-4400
TRAIN_DIR=$REPO/outputs/$RUN
VI_CLONE=/mnt/data/users/anhnct/tts-eval/evalsets/vi_clone_300.jsonl
STEPS="160 320 480"
TMPDIR=/tmp/minhth21
mkdir -p "$TMPDIR" "$E/logs" "$E/gen"
cd "$REPO"

for step in $STEPS; do
  out=$EXPORTS/${RUN}__checkpoint-$step
  if [[ -f $out/export_info.json ]]; then echo "reusing $out"; continue; fi
  CUDA_VISIBLE_DEVICES= $PY -m breeze_train.export "$BASE" "$TRAIN_DIR/checkpoint-$step" "$out" \
    --policy p1 --name "${RUN}-checkpoint-$step" > "$E/logs/export_spkfix_$step.log" 2>&1
  chgrp -R ttsdata "$out" 2>/dev/null || true
  chmod -R g+rwX "$out" 2>/dev/null || true
  echo "exported $out"
done

gen() {  # gen NAME MODEL_DIR EVALSET...
  local name=$1 model=$2; shift 2
  for evalset in "$@"; do
    local file=$E/evalsets/$evalset.jsonl
    [[ $evalset == vi_clone_300 ]] && file=$VI_CLONE
    $PY "$GENERATOR" "$model" "$file" "$E/gen/${name}__$evalset" --tag-style vlsp \
      >> "$E/logs/gen_$name.log" 2>&1
    echo "done $name $evalset ($(ls "$E/gen/${name}__$evalset" | grep -c '\.wav$') wavs)"
  done
}
pids=()
for step in $STEPS; do
  gen "spkfix_$step" "$EXPORTS/${RUN}__checkpoint-$step" tc_val_99 tc_clone_100 sn_clone_100 br_clone_100 vi_clone_300 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
