#!/usr/bin/env bash
# Install the isolated SGLang-Omni 0.1.4 runtime for Breeze TTS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Set PYTHON to use an existing environment (for example, after conda activate).
# Otherwise retain the repo-local Python 3.12 venv as the default.
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$(command -v "$PYTHON")"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  [ -d .venv ] || uv venv .venv -p 3.12
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "bootstrap.sh: Python not executable: $PYTHON_BIN" >&2; exit 1; }
SITE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

uv pip install --python "$PYTHON_BIN" --prerelease=allow "sglang-omni==0.1.4" soundfile

# qwen-tts pins an older Transformers release; do not resolve its dependencies.
uv pip install --python "$PYTHON_BIN" --no-deps "qwen-tts==0.1.1"

# Match CUDA runtime/compiler/headers: torch ships CUDA 13.0, while the nvcc
# wheels otherwise resolve to 13.4 and CCCL rejects the mixed toolkit.
uv pip install --python "$PYTHON_BIN" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
  "nvidia-cuda-cccl==13.0.*" "nvidia-nvvm==13.0.*"
CU="$SITE/nvidia/cu13"
ln -sfn lib "$CU/lib64"
ln -sfn libcudart.so.13 "$CU/lib/libcudart.so"
"$CU/bin/nvcc" --version | tail -1

# Import Breeze's registry hook in the CLI and every spawn-created interpreter.
printf '%s\n' 'import os; os.environ.get("BREEZE_OMNI_REGISTER") == "1" and __import__("breeze_omni.registration", fromlist=["register"]).register()' > "$SITE/breeze_omni_register.pth"
"$PYTHON_BIN" -c 'import sglang_omni, sglang; print("SGLang-Omni runtime imports OK")'
echo "bootstrap OK -> ./serve.sh"
