#!/usr/bin/env bash
# DPO signal probe: generate K seeds x 2 evalsets (tc_clone_100, vi_probe_100, both built
# by make_evalsets.py) with ONE fixed checkpoint (nv_lora02 checkpoint-4400 by default), to
# see whether same-prompt samples differ enough to form DPO chosen/rejected pairs.
# breeze_generate.py takes one --seed per whole invocation (not per-row), so seed variation
# = one generator run per seed into its own out_dir, same pattern run_gen.sh uses per-
# checkpoint. Shares evalsets/, gen/, runs/ with the checkpoint-sweep eval (run_gen.sh):
# outputs are named seed_<seed>__<evalset>, distinct from that script's lora_<step>__/
# baseline_*__ names, so nothing collides.
#
# Runs ON THE LOGIN NODE (start detached: tmux new -d -s dpo_probe_gen "JOB=22286 bash /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/dpo_probe_run_gen.sh > /mnt/data/users/minhth21/breeze-tts/nv_lora/eval/logs/dpo_probe_gen.log 2>&1");
# generation itself runs inside job $JOB via srun --overlap. Resumable: breeze_generate.py
# skips wavs already written.
#
# MODEL/SEEDS/EVALSETS/JOB/STREAMS are overridable env vars (STREAMS caps concurrent
# generator processes on the one GPU, default 4 — eval/run_gen.sh runs ~5 streams for
# comparison). EVALSETS defaults to the throatclearing + plain-reading probe; the 4-tag
# probe adds the same-sentence sniff/breathing/laughter sets, e.g.
#   JOB=22292 SEEDS="42 43 44 45" EVALSETS="sn_clone_100 br_clone_100 la_clone_100" bash dpo_probe_run_gen.sh
# Shared path config (REPO/PY_BREEZE/GENERATOR/EXPORTS/...) lives in nv_lora/env.sh; JOB
# is required, no default (shared-cluster job ids get repurposed — see env.sh):
#   JOB=22286 bash dpo_probe_run_gen.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
JOB="${JOB:?set JOB=<slurm job id>, e.g. JOB=22286 — check squeue --me for a free one}"
E=$REPO/nv_lora/eval
PY=$PY_BREEZE
MODEL="${MODEL:-$EXPORTS/nv_lora02_p1-r32_from-vi33k_remap-v3-46k__checkpoint-4400}"
read -ra SEEDS <<< "${SEEDS:-42 43 44 45 46 47 48 49}"
read -ra EVALSETS <<< "${EVALSETS:-tc_clone_100 vi_probe_100}"
STREAMS="${STREAMS:-4}"

[[ -f $MODEL/export_info.json ]] || { echo "missing export: $MODEL" >&2; exit 1; }
python3 "$E/make_evalsets.py" "$E/evalsets"
for evalset in "${EVALSETS[@]}"; do
  [[ -f $E/evalsets/$evalset.jsonl ]] || { echo "unknown evalset: $evalset (not built by make_evalsets.py)" >&2; exit 1; }
done

mkdir -p "$E/logs" "$E/gen"
# Semaphore below: must use `wait -n || rc=1` in BOTH the cap loop and the final drain,
# never a pids[] array waited individually afterward — `wait -n` reaps a job from bash's
# table immediately, so a later `wait $specific_pid` on an already-reaped pid returns 127
# ("not a child of this shell") and falsely marks a fully successful run as failed.
srun --overlap --jobid="$JOB" env TMPDIR=/tmp/minhth21 bash -s <<EOF
set -euo pipefail
mkdir -p /tmp/minhth21
cd $REPO

gen() {  # gen SEED EVALSET
  local seed=\$1 evalset=\$2
  local name=seed_\${seed}__\${evalset}
  $PY $GENERATOR $MODEL $E/evalsets/\${evalset}.jsonl $E/gen/\${name} \
    --seed \$seed --tag-style vlsp >> $E/logs/gen_\${name}.log 2>&1
  echo "done \${name} (\$(ls $E/gen/\${name} | grep -c '\.wav\$') wavs)"
}
rc=0
for seed in ${SEEDS[*]}; do
  for evalset in ${EVALSETS[*]}; do
    while (( \$(jobs -rp | wc -l) >= $STREAMS )); do wait -n || rc=1; done
    gen \$seed \$evalset &
  done
done
while (( \$(jobs -p | wc -l) > 0 )); do wait -n || rc=1; done
exit \$rc
EOF
echo "all generation finished"
