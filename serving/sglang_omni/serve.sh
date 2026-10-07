#!/usr/bin/env bash
# Serve the local Breeze TTS 2 checkpoint with SGLang-Omni.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Match bootstrap.sh: PYTHON overrides, active Conda is detected, else use .venv.
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$(command -v "$PYTHON")"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "serve.sh: Python not executable: $PYTHON_BIN (run ./bootstrap.sh first)" >&2; exit 1; }
ENV_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix + "/bin")')"
SITE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PATH="$ENV_BIN:$PATH"

MODEL="${MODEL:-$(realpath "$ROOT/../../../breeze-tts-2")}"
PORT="${PORT:-8000}"
# 12 GB budget (one process, one GPU): backbone+depth 3.5 GB + KV inside SGLang's
# static pool, then the T5Gemma2 text encoder (1.9 GB), codec (0.3 GB), the depth
# CUDA graph and activations outside it. 0.56 gives ~26.8k KV tokens and leaves
# ~2.8 GB headroom (measured).
MEM_FRACTION="${MEM_FRACTION:-0.56}"
# Lanes, not requests: a CFG request takes two (cond + uncond). Keep
# MAX_RUNNING * 2048 (max_seq_len) within the KV pool so SGLang never retracts
# a request: a retracted request restarts generation after its audio has
# already been streamed.
MAX_RUNNING="${MAX_RUNNING:-12}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-$MAX_RUNNING}"
ATTENTION="${ATTENTION:-triton}"

CUDA_HOME="$SITE/nvidia/cu13"
export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BREEZE_OMNI_REGISTER=1

# sgl-omni silently switches to a random port when PORT cannot be bound (e.g. a
# previous server is still shutting down). Wait for the port, then fail loudly.
port_free() {
  python -c "import socket,sys; s=socket.socket(); s.bind(('${HOST:-0.0.0.0}', int(sys.argv[1])))" "$PORT" 2>/dev/null
}
for _ in $(seq 30); do port_free && break; sleep 1; done
port_free || { echo "serve.sh: port $PORT is still in use" >&2; exit 1; }

exec sgl-omni serve \
  --model-path "$MODEL" \
  --port "$PORT" \
  --host "${HOST:-0.0.0.0}" \
  --tts_engine.engine.mem_fraction_static "$MEM_FRACTION" \
  --tts_engine.engine.max_running_requests "$MAX_RUNNING" \
  --tts_engine.engine.cuda_graph_max_bs "$CUDA_GRAPH_MAX_BS" \
  --tts_engine.engine.attention_backend "$ATTENTION" \
  "$@"
