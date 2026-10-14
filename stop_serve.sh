#!/usr/bin/env bash
# Stop the Breeze SGLang-Omni server inside job 21775 (no restart). Run on the login node.
# Same kill logic as restart_serve.sh: tmux kill-session alone does NOT stop the srun step.
set -euo pipefail
JOB=21775
srun --overlap --jobid="$JOB" bash -s <<'EOF'
set -uo pipefail
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
pgrep -u "$USER" -af 'sgl-omni[ ]serve' && { echo "server still alive" >&2; exit 1; }
if ss -ltn | grep -q ':8000 '; then echo "port 8000 still busy" >&2; exit 1; fi
echo "port 8000 free"
sleep 3
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
EOF
tmux kill-session -t breeze_serve 2>/dev/null || true
echo "stopped"
