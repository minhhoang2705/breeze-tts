#!/usr/bin/env bash
# Create the isolated SGLang-Omni 0.1.4 runtime for Breeze TTS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[ -d .venv ] || uv venv .venv -p 3.12
source .venv/bin/activate
UV_HTTP_TIMEOUT=120 uv pip install --prerelease=allow "sglang-omni==0.1.4" soundfile

# qwen-tts pins an older Transformers release; do not resolve its dependencies.
uv pip install --no-deps "qwen-tts==0.1.1"

# Match CUDA runtime/compiler/headers: torch ships CUDA 13.0, while the nvcc
# wheels otherwise resolve to 13.4 and CCCL rejects the mixed toolkit.
uv pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
  "nvidia-cuda-cccl==13.0.*" "nvidia-nvvm==13.0.*"
CU="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
ln -sfn lib "$CU/lib64"
ln -sfn libcudart.so.13 "$CU/lib/libcudart.so"
"$CU/bin/nvcc" --version | tail -1

# Import Breeze's registry hook in the CLI and every spawn-created interpreter.
SITE="$ROOT/.venv/lib/python3.12/site-packages"
printf '%s\n' 'import os; os.environ.get("BREEZE_OMNI_REGISTER") == "1" and __import__("breeze_omni.registration", fromlist=["register"]).register()' > "$SITE/breeze_omni_register.pth"
"$ROOT/.venv/bin/python" -c 'import sglang_omni, sglang; print("SGLang-Omni runtime imports OK")'
echo "bootstrap OK -> ./serve.sh"
