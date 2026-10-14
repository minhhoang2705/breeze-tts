#!/usr/bin/env bash
# Restart the Breeze SGLang-Omni server inside job 21775. Run on the login node.
# `tmux kill-session` alone does NOT stop the srun step (srun ignores the SIGHUP and the
# old server keeps holding port 8000), so stop the server processes on the worker first.
set -euo pipefail
B=/mnt/data/users/minhth21/breeze-tts
JOB=21775
srun --overlap --jobid="$JOB" bash -s <<'EOF'
set -uo pipefail
# Bracketed patterns so this script's own command line never matches.
roots=$(pgrep -u "$USER" -f 'sgl-omni[ ]serve|sglang_omni/serve[.]sh' || true)
tree=""
for p in $roots; do tree="$tree $p $(pgrep -u "$USER" -P "$p" | tr '\n' ' ')"; done
for p in $tree; do tree="$tree $(pgrep -u "$USER" -P "$p" | tr '\n' ' ')"; done
tree=$(echo $tree | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "stopping: ${tree:-none}"
[ -n "$tree" ] && kill -TERM $tree 2>/dev/null
for _ in $(seq 60); do
  alive=$(for p in $tree; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
  [ -z "$alive" ] && break
  sleep 1
done
[ -n "${alive:-}" ] && { echo "SIGKILL: $alive"; kill -KILL $alive 2>/dev/null; sleep 2; }
if ss -ltn | grep -q ':8000 '; then echo "port 8000 still busy" >&2; exit 1; fi
echo "port 8000 free"
EOF
tmux kill-session -t breeze_serve 2>/dev/null || true
# tmux sessions do not inherit this shell's variables; forward the BREEZE_* switches, MODEL
# and MEM_FRACTION (e.g. `MODEL=<export dir> MEM_FRACTION=0.2 bash restart_serve.sh` when other
# jobs already hold GPU memory); run_serve.sh / serve.sh supply defaults.
env_args=()
for v in BREEZE_COMPILE_DEPTH BREEZE_REF_ENCODE_DEVICE MODEL MEM_FRACTION; do
  [ -n "${!v:-}" ] && env_args+=(-e "$v=${!v}")
done
tmux new -d -s breeze_serve "${env_args[@]}" "srun --overlap --jobid=$JOB bash $B/run_serve.sh > $B/logs/serve.log 2>&1"
echo "started tmux breeze_serve ${env_args[*]}"
