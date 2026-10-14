#!/usr/bin/env bash
# Generate vi_clone_300 through the live server, then score it with tts-eval.
# Usage (on the worker, via srun): eval_quality.sh NAME
set -euo pipefail
NAME="${1:?run name}"
B=/mnt/data/users/minhth21/breeze-tts
TE=/mnt/data/users/anhnct/tts-eval
WAVS="$B/logs/eval/wavs/$NAME"
RUN="$B/logs/eval/runs/$NAME"
/mnt/data/users/minhth21/conda_env/breeze312/bin/python "$B/gen_evalset.py" \
  --evalset "$TE/evalsets/vi_clone_300.jsonl" --out "$WAVS"
# Score from the shared tts-eval checkout without writing into it: results go to $RUN,
# models load from local files only, no bytecode caches.
source /mnt/data/users/anhnct/conda_env/miniconda3/etc/profile.d/conda.sh
conda activate tts_eval
cd "$TE"
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python -m tts_eval.evaluate \
  --evalset evalsets/vi_clone_300.jsonl --wavs "$WAVS" --run-dir "$RUN" --model-name "$NAME" --device cuda:0
