#!/usr/bin/env bash
set -euo pipefail
CKPT="/mnt/data/users/anhnct/breeze-tts/outputs/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k/checkpoint-8000"
BASE="/mnt/data/users/anhnct/breeze-tts-2-ckpt"
SERVE="/mnt/data/users/minhth21/breeze-tts/outputs/checkpoint-8000-serve"
mkdir -p "$SERVE"
# From the fine-tuned checkpoint: the actual model weights + its config.
for f in config.json generation_config.json model-00001-of-00003.safetensors \
         model-00002-of-00003.safetensors model-00003-of-00003.safetensors \
         model.safetensors.index.json; do
  ln -sfn "$CKPT/$f" "$SERVE/$f"
done
# From the base release: static assets training never touches.
ln -sfn "$BASE/audio_tokenizer" "$SERVE/audio_tokenizer"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
  ln -sfn "$BASE/$f" "$SERVE/$f"
done
echo "--- serve dir built ---"
ls -la "$SERVE"
