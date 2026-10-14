#!/usr/bin/env bash
# Score every seed's generated audio (both probe evalsets) with tts-eval: 1-WER, pMOS,
# spk_sim, duration_sec per sample. Same invocation as score_reading.sh, generalized over
# two evalsets (tc_clone_100 has [throatclearing]; tts-eval's text_norm strips the tag
# before WER, so reading quality is still comparable) instead of one, into the SAME runs/
# dir as the checkpoint-sweep eval (names seed_<seed>__<evalset> don't collide with
# lora_<step>__/baseline_*__).
# Runs ON THE LOGIN NODE (detached: tmux new -d -s dpo_probe_read "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/dpo_probe_score_reading.sh > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/dpo_probe_read.log 2>&1");
# scoring runs inside job $JOB. Results: nv_lora/eval/runs/seed_<seed>__<evalset>/.
# SEEDS/EVALSETS/JOB are overridable env vars (SEEDS default 42..49, EVALSETS default the
# throatclearing + plain-reading probe; for the 4-tag probe's extra sets use e.g.
# SEEDS="42 43 44 45" EVALSETS="sn_clone_100 br_clone_100 la_clone_100"). Shared path config
# (REPO/PY_TTS_EVAL/TE/...) lives in nv_lora/env.sh; JOB is required, no default
# (shared-cluster job ids get repurposed — see env.sh):
#   JOB=22286 bash dpo_probe_score_reading.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
E=$REPO/nv_lora/eval
PY=$PY_TTS_EVAL
read -ra SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
read -ra EVALSETS <<< "${EVALSETS:-tc_clone_100 vi_probe_100}"

for seed in "${SEEDS[@]}"; do
  for evalset in "${EVALSETS[@]}"; do
    want=$(grep -c . "$E/evalsets/$evalset.jsonl")
    n=$(find -L "$E/gen/seed_${seed}__${evalset}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
    [[ $n -eq $want ]] || { echo "not ready: seed_${seed}__${evalset} $n/$want" >&2; exit 3; }
  done
done
mkdir -p "$E/runs" "$E/logs"
srun --overlap --jobid="$JOB" bash -c "cd $TE && for seed in ${SEEDS[*]}; do for evalset in ${EVALSETS[*]}; do
  name=seed_\${seed}__\${evalset}
  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -m tts_eval.evaluate \
    --evalset $E/evalsets/\${evalset}.jsonl --wavs $E/gen/\${name} \
    --run-dir $E/runs/\${name} --model-name \${name} --device cuda:0 > $E/logs/score_\${name}.log 2>&1
  grep -h '^\[.*\] 1-WER=' $E/logs/score_\${name}.log
done; done"
