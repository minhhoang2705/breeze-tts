#!/usr/bin/env bash
# nv_lora: LoRA (p1) fine-tune of a Breeze checkpoint on a non-verbal-tag manifest.
# Modeled on breeze-tts-remote/experiments/nv01_.../run.sh.
#
# Runs ON THE LOGIN NODE; GPU work goes into job $JOB via srun --overlap. Start long
# stages detached, e.g.  tmux new -d -s nv_lora "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/run.sh train > /mnt/data/users/minhth21/breeze-tts/nv_lora/logs/run.log 2>&1"
#   bash run.sh cache            encode missing codec-cache entries (idempotent)
#   bash run.sh smoke            10 training steps into outputs/<RUN>_smoke (deleted first)
#   bash run.sh train [ARGS..]   full run into outputs/<RUN>; extra ARGS go to train.py
#                                (e.g. --resume, --max-steps 1200)
# All of RUN/INIT/CONFIG/DATA_DIR/MAX_STEPS/JOB are overridable env vars (defaults below
# reproduce nv_lora01: VieNeu [throatclearing] LoRA from nv01 checkpoint-8000). Shared
# path config (REPO/PY_BREEZE/BASE_CKPT/...) lives in nv_lora/env.sh; JOB is required
# there's no default since shared-cluster job ids get repurposed (21775/worker-5 was this
# project's sandbox, is now the live breeze_serve inference job). Every run evaluates
# DATA_DIR/val.jsonl (unseen sentences) and, when present, DATA_DIR/val_voices.jsonl
# (unseen voices), and renders sample_prompts.jsonl at step 0 and every save. codec_cache
# is shared across runs (keyed by audio content hash).
#   JOB=22286 bash run.sh train
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
D=$REPO/nv_lora
PY=$PY_BREEZE
INIT="${INIT:-/mnt/data/users/anhnct/breeze-tts/outputs/exported/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k__checkpoint-8000}"
RUN="${RUN:-nv_lora01_p1-r32_from-nv01-8k_vieneu-tc4.4k}"
CONFIG="${CONFIG:-$D/train_config.json}"
DATA_DIR="${DATA_DIR:-$D/data}"
MAX_STEPS="${MAX_STEPS:-800}"   # nv_lora01 default: ~2.9 epochs of 4,434 clips at global batch 16

stage="${1:?usage: run.sh cache | smoke | train [train.py args]}"
shift
mkdir -p "$D/logs"

on_gpu() {
  timeout "${TIMEOUT:-0}" srun --overlap --jobid="$JOB" \
    env PYTORCH_ALLOC_CONF=expandable_segments:True TMPDIR=/tmp/minhth21 \
    bash -c 'mkdir -p "$TMPDIR" && cd '"$REPO"' && "$@"' _ "$@"
}

train() {
  local out=$1; shift
  local extra=(--eval-manifest "val_sentences=$DATA_DIR/val.jsonl" --sample-prompts "$D/sample_prompts.jsonl")
  if [[ -f $DATA_DIR/val_voices.jsonl ]]; then
    extra+=(--eval-manifest "val_voices=$DATA_DIR/val_voices.jsonl")
  fi
  on_gpu "$PY" train.py "$INIT" \
    --config "$CONFIG" \
    --manifest "$DATA_DIR/train.jsonl" --cache-dir "$D/codec_cache" \
    --output-dir "$out" "${extra[@]}" "$@"
}

case "$stage" in
  cache)
    for split in train val val_voices; do
      [[ -f $DATA_DIR/$split.jsonl ]] || continue
      on_gpu "$PY" -m breeze_train.codec_cache "$BASE_CKPT" "$DATA_DIR/$split.jsonl" "$D/codec_cache" \
        > "$D/logs/cache_$split.log" 2>&1
      tail -n 1 "$D/logs/cache_$split.log"
    done
    ;;
  smoke)
    rm -rf "$REPO/outputs/${RUN}_smoke"
    TIMEOUT=1800 train "$REPO/outputs/${RUN}_smoke" --max-steps 10 --save-steps 10 \
      > "$D/logs/smoke.log" 2>&1
    tail -n 15 "$D/logs/smoke.log"
    ;;
  train)
    if [[ -e $REPO/outputs/$RUN && " $* " != *" --resume"* ]]; then
      echo "outputs/$RUN exists; pass --resume or set RUN=<new name>" >&2
      exit 1
    fi
    {
      echo "run=$RUN start=$(date -u +%FT%TZ) job=$JOB init=$INIT config=$CONFIG data_dir=$DATA_DIR max_steps=$MAX_STEPS args=$*"
    } >> "$D/run_info.txt"
    train "$REPO/outputs/$RUN" --max-steps "$MAX_STEPS" "$@" > "$D/logs/train_$RUN.log" 2>&1
    echo "end=$(date -u +%FT%TZ)" >> "$D/run_info.txt"
    ;;
  *) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
