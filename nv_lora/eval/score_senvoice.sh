#!/usr/bin/env bash
# Score one Breeze checkpoint on the Senvoice metrics (VI-WER, EN-WER, CS-Latin, CS-IPA,
# CS-Transliteration, UTMOSv2 -- see nv_lora/eval/senvoice_metrics.py's module docstring
# for exactly how each is computed offline on H100 and how that differs from anhnct's
# Senvoice portal numbers). STOI and SIM-O are intentionally not computed -- see the same
# docstring.
#
# Runs ON THE LOGIN NODE (start detached: tmux new -d -s sv_eval "JOB=22465 MODEL=... \
#   MODEL_NAME=... bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/score_senvoice.sh \
#   > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/sv_eval.log 2>&1");
# generation/ASR/UTMOS all run inside job $JOB via srun --overlap; only the final
# senvoice_metrics.py aggregation step runs on the login node (pure Python, no GPU).
#
# Required: MODEL (exported checkpoint dir, with export_info.json), JOB (Slurm job id --
# shared-cluster ids get repurposed, see nv_lora/env.sh -- check squeue --me for a free
# one). Optional: MODEL_NAME (defaults to MODEL's basename), PY_UTMOS (env with
# torch+torchvision+torchaudio+transformers+timm+librosa; breeze312 on H100 has all of
# them and is the default), WHISPER_DIR / UTMOS_DIR (local offline weight dirs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22465 — check squeue --me for a free one}"
MODEL="${MODEL:?set MODEL=<exported checkpoint dir>}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL")}"
[[ -f "$MODEL/export_info.json" ]] || { echo "not an export dir (no export_info.json): $MODEL" >&2; exit 1; }

E=$REPO/nv_lora/eval
PY_UTMOS="${PY_UTMOS:-/mnt/data/users/minhth21/conda_env/breeze312/bin/python}"
# breeze312's torch is a cu130 build; driver 550 needs the forward-compat libcuda from
# setup_cuda_compat.sh (breeze_tts/tts_eval's cu128 torch does not need this).
CUDA_COMPAT_LIB="${CUDA_COMPAT_LIB:-/mnt/data/users/minhth21/conda_env/cuda13compat/cuda-compat}"
WHISPER_DIR="${WHISPER_DIR:-/mnt/data/users/minhth21/models/whisper-large-v3}"
UTMOS_DIR="${UTMOS_DIR:-/mnt/data/users/minhth21/models/utmosv2}"
OUT=$E/runs_senvoice/$MODEL_NAME
SV_SETS=(sv_vi_200 sv_en_200 sv_cs_latin_200 sv_cs_ipa_200 sv_cs_translit_200)
UTMOS_SETS=(sv_vi_200 sv_en_200 sv_cs_latin_200 sv_cs_ipa_200 sv_cs_translit_200)
[[ -d $E/gen/${MODEL_NAME}__vi_clone_300 ]] && UTMOS_SETS+=(vi_clone_300)

mkdir -p "$E/logs" "$E/gen" "$OUT"
python3 "$E/make_sv_evalsets.py" "$E/evalsets"

srun --overlap --jobid="$JOB" env TMPDIR=/tmp/minhth21 bash -s <<EOF
set -euo pipefail
mkdir -p /tmp/minhth21
cd $REPO

echo "--- generation ---"
for evalset in ${SV_SETS[*]}; do
  $PY_BREEZE $GENERATOR $MODEL $E/evalsets/\$evalset.jsonl $E/gen/${MODEL_NAME}__\$evalset \
    --seed 42 >> $E/logs/gen_${MODEL_NAME}_\$evalset.log 2>&1
  echo "done \$evalset (\$(ls $E/gen/${MODEL_NAME}__\$evalset | grep -c '\.wav\$')/200 wavs)"
done

echo "--- VI-WER (tts-eval, ZipFormer) ---"
(cd $TE && HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY_TTS_EVAL -m tts_eval.evaluate \
  --evalset $E/evalsets/sv_vi_200.jsonl --wavs $E/gen/${MODEL_NAME}__sv_vi_200 \
  --run-dir $E/runs/${MODEL_NAME}__sv_vi_200 --model-name ${MODEL_NAME}__sv_vi_200 \
  --device cuda:0) > $E/logs/vi_wer_${MODEL_NAME}.log 2>&1
grep -h '^\[.*\] 1-WER=' $E/logs/vi_wer_${MODEL_NAME}.log

echo "--- EN-WER / CS-* transcripts (Whisper large-v3) ---"
for evalset in sv_en_200 sv_cs_latin_200 sv_cs_ipa_200 sv_cs_translit_200; do
  lang=vi; [[ \$evalset == sv_en_200 ]] && lang=en
  LD_LIBRARY_PATH=$CUDA_COMPAT_LIB $PY_UTMOS $E/whisper_transcribe.py --wavs $E/gen/${MODEL_NAME}__\$evalset \
    --evalset $E/evalsets/\$evalset.jsonl --language \$lang --model-dir $WHISPER_DIR \
    --out $OUT/asr_\$evalset.jsonl > $E/logs/whisper_${MODEL_NAME}_\$evalset.log 2>&1
  tail -n 1 $E/logs/whisper_${MODEL_NAME}_\$evalset.log
done

echo "--- UTMOSv2 ---"
for evalset in ${UTMOS_SETS[*]}; do
  PYTHONPATH=$UTMOS_DIR/python HF_HOME=$UTMOS_DIR/huggingface TRANSFORMERS_CACHE=$UTMOS_DIR/huggingface \
    LD_LIBRARY_PATH=$CUDA_COMPAT_LIB $PY_UTMOS $E/utmos_score.py --audio-dir $E/gen/${MODEL_NAME}__\$evalset \
    --checkpoint-dir $UTMOS_DIR --output $OUT/utmosv2_\$evalset.json \
    > $E/logs/utmos_${MODEL_NAME}_\$evalset.log 2>&1
  tail -n 1 $E/logs/utmos_${MODEL_NAME}_\$evalset.log
done
EOF

echo "--- aggregating ---"
python3 "$E/senvoice_metrics.py" --eval-dir "$E" --model "$MODEL_NAME"
echo "wrote $OUT/metrics.json"
