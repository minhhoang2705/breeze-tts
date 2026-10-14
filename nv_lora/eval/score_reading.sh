#!/usr/bin/env bash
# Score vi_clone_300 reading quality (1-WER, pMOS, spk_sim) for every model with tts-eval.
# Runs ON THE LOGIN NODE (detached: tmux new -d -s nv_lora_read "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/score_reading.sh > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/score_reading.log 2>&1");
# scoring runs inside job $JOB. Results: nv_lora/eval/runs/<model>/summary.json.
# Same invocation as eval_quality.sh: shared tts-eval checkout used read-only, models from
# local files only (HF_HUB_OFFLINE=1), no bytecode caches.
#   baseline_8000_regen  nv01-8000 regenerated with this repo/env (the comparison baseline)
#   baseline_8000        nv01-8000 audio generated earlier by anhnct's repo/env (cross-check)
# MODELS/JOB are overridable env vars (MODELS default reproduces nv_lora01). Shared path
# config (REPO/PY_TTS_EVAL/TE/...) lives in nv_lora/env.sh; JOB is required, no default
# (shared-cluster job ids get repurposed — see env.sh). For nv_lora02:
#   JOB=22286 MODELS="baseline_vi33k lora_4400 lora_4800 lora_5200 lora_5600" bash score_reading.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
E=$REPO/nv_lora/eval
PY=$PY_TTS_EVAL
MODELS="${MODELS:-baseline_8000_regen baseline_8000 lora_200 lora_400 lora_600 lora_800}"

for m in $MODELS; do
  n=$(find -L "$E/gen/${m}__vi_clone_300" -maxdepth 1 -name '*.wav' | wc -l)
  [[ $n -eq 300 ]] || { echo "not ready: $m $n/300" >&2; exit 3; }
done
mkdir -p "$E/runs" "$E/logs"
srun --overlap --jobid="$JOB" bash -c "cd $TE && for m in $MODELS; do
  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -m tts_eval.evaluate \
    --evalset evalsets/vi_clone_300.jsonl --wavs $E/gen/\${m}__vi_clone_300 \
    --run-dir $E/runs/\$m --model-name \$m --device cuda:0 > $E/logs/score_\$m.log 2>&1
  grep -h '^\[.*\] 1-WER=' $E/logs/score_\$m.log
done"
