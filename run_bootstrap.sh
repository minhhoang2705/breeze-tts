#!/usr/bin/env bash
# Create an isolated Python 3.12 conda env, then run bootstrap.sh into it.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never
CONDA=/mnt/data/users/minhth21/conda_env/miniconda3/bin/conda
ENV_PREFIX=/mnt/data/users/minhth21/conda_env/breeze312
[[ -x "$CONDA" ]] || { echo "run_bootstrap.sh: conda not found at $CONDA" >&2; exit 1; }
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.12
fi
"$ENV_PREFIX/bin/python" --version
export PYTHON="$ENV_PREFIX/bin/python"
cd /mnt/data/users/minhth21/breeze-tts/serving/sglang_omni
exec ./bootstrap.sh
