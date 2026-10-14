#!/usr/bin/env bash
# Serve checkpoint-8000 (via the symlinked serve dir) with SGLang-Omni.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
unset CONDA_PREFIX
# A conda-activated shell can export HOST=<build-triplet> (e.g. x86_64-conda-linux-gnu);
# serve.sh binds to ${HOST:-0.0.0.0} for both its port-free check and the actual server,
# so a leaked HOST makes the port-wait loop fail with a misleading "port already in use".
unset HOST
export PYTHON=/mnt/data/users/minhth21/conda_env/breeze312/bin/python
# Driver 550 (CUDA 12.4) can't run torch cu130; use NVIDIA's CUDA 13 forward-compat libcuda.
COMPAT_LIB=/mnt/data/users/minhth21/conda_env/cuda13compat/cuda-compat
export LD_LIBRARY_PATH="$COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH="$COMPAT_LIB"
export MODEL="${MODEL:-/mnt/data/users/minhth21/breeze-tts/outputs/checkpoint-8000-serve}"
export PORT="${PORT:-8000}"
# 1 (default) torch.compiles the depth-decoder CUDA graph (~2x faster per frame); 0 rolls back.
export BREEZE_COMPILE_DEPTH="${BREEZE_COMPILE_DEPTH:-1}"
# cuda (default) encodes clone reference audio on the GPU in fp32, like Breeze's infer.py; cpu rolls back.
export BREEZE_REF_ENCODE_DEVICE="${BREEZE_REF_ENCODE_DEVICE:-cuda}"
# GPU startup locks go to tempfile.gettempdir(); worker-5's /tmp holds another user's copies.
export TMPDIR=/tmp/minhth21
mkdir -p "$TMPDIR"
cd /mnt/data/users/minhth21/breeze-tts/serving/sglang_omni
# /tmp/sglang_omni on worker-5 belongs to another user; use a per-user IPC dir.
exec ./serve.sh --endpoints.base_path "/tmp/sglang_omni_$USER"
