# nv_lora/env.sh — shared path config for nv_lora training/eval/DPO-probe scripts.
# Source from nv_lora/:        source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Source from nv_lora/eval/:   source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
#
# Everything here is `${VAR:-default}` against the current minhth21/anhnct path layout —
# override any single var per-invocation, e.g. PY_BREEZE=/some/other/python bash run.sh.
#
# JOB (the Slurm job id for `srun --overlap --jobid=$JOB`) is deliberately NOT set here:
# only scripts that actually touch the GPU need it, and shared-cluster job ids get
# repurposed over time (21775/worker-5 was this project's training sandbox, is now the
# live breeze_serve inference job — a stale default would silently target someone's live
# serving GPU). Each GPU-touching script declares its own required check:
#   JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"

REPO="${REPO:-/mnt/data/users/minhth21/breeze-tts}"
PY_BREEZE="${PY_BREEZE:-/mnt/data/users/minhth21/conda_env/miniconda3/envs/breeze_tts/bin/python}"
PY_TTS_EVAL="${PY_TTS_EVAL:-/mnt/data/users/minhth21/conda_env/miniconda3/envs/tts_eval/bin/python}"
GENERATOR="${GENERATOR:-/mnt/data/users/anhnct/breeze-tts-remote/remote/breeze_generate.py}"
EXPORTS="${EXPORTS:-/mnt/data/users/anhnct/breeze-tts/outputs/exported}"
TE="${TE:-/mnt/data/users/anhnct/tts-eval}"
BASE_CKPT="${BASE_CKPT:-/mnt/data/users/anhnct/breeze-tts-2-ckpt}"
