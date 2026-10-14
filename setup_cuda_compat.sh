#!/usr/bin/env bash
# Install NVIDIA's CUDA 13.0 forward-compat userspace driver into its own prefix,
# then check that the breeze312 torch (cu130) can run on the GPU with it.
set -euo pipefail
CONDA=/mnt/data/users/minhth21/conda_env/miniconda3/bin/conda
COMPAT_PREFIX=/mnt/data/users/minhth21/conda_env/cuda13compat
PY=/mnt/data/users/minhth21/conda_env/breeze312/bin/python
if [[ ! -d "$COMPAT_PREFIX" ]]; then
  "$CONDA" create -y -p "$COMPAT_PREFIX" -c conda-forge --override-channels "cuda-compat=13.0"
fi
LIBCUDA="$(find "$COMPAT_PREFIX" -name 'libcuda.so.*' -print -quit)"
[[ -n "$LIBCUDA" ]] || { echo "setup_cuda_compat.sh: no libcuda.so in $COMPAT_PREFIX" >&2; exit 1; }
COMPAT_LIB="$(dirname "$LIBCUDA")"
echo "compat lib dir: $COMPAT_LIB"
ls -la "$COMPAT_LIB"
LD_LIBRARY_PATH="$COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
x = torch.ones(1024, device="cuda") * 2
torch.cuda.synchronize()
print("gpu op OK:", x.sum().item(), torch.cuda.get_device_name(0))
EOF
