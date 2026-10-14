#!/usr/bin/env bash
# Score one [throatclearing] evalset (tc_val_99 or tc_clone_100) of every model with
# nv_gemini_audit. Run LOCALLY (needs GEMINI_API_KEY + internet); each remote step is a
# single ssh call. Exits 3 without doing anything if that evalset's generation is unfinished.
#   GEMINI_API_KEY=... bash nv_lora/eval/score_tags.sh tc_val_99
# RUN_NAME/MODELS are overridable env vars (defaults reproduce nv_lora01). For nv_lora02:
#   RUN_NAME=nv_lora02 MODELS="baseline_vi33k lora_4400 lora_4800 lora_5200 lora_5600" \
#     GEMINI_API_KEY=... bash nv_lora/eval/score_tags.sh tc_val_99
# Audit run: nv_gemini_audit/runs/<RUN_NAME>_<evalset> (summary.md on H100 and locally).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
E=$REPO/nv_lora/eval
EVALSET="${1:?usage: score_tags.sh tc_val_99|tc_clone_100}"
RUN_NAME="${RUN_NAME:-nv_lora01}"
MODELS="${MODELS:-baseline_8000 lora_200 lora_400 lora_600 lora_800}"

echo "== check $EVALSET generation is complete"
ssh -n H100 "timeout 60 bash -c 'want=\$(grep -c . $E/evalsets/$EVALSET.jsonl); for m in $MODELS; do
  n=\$(ls $E/gen/\${m}__$EVALSET 2>/dev/null | grep -c \"\\.wav\$\" || true)
  [[ \$n -eq \$want ]] || { echo \"not ready: \$m \$n/\$want\"; exit 3; }
done; echo ready'"

echo "== build audit metadata on H100"
ssh -n H100 "timeout 60 python3 $E/to_audit_metadata.py $E $EVALSET $E/audit_${RUN_NAME}_$EVALSET.jsonl $MODELS"

"$HERE/../../nv_gemini_audit/run_audit.sh" "${RUN_NAME}_$EVALSET" \
  --vlsp "$E/audit_${RUN_NAME}_$EVALSET.jsonl" --tags throatclearing --per-tag 5000 --include-untagged 0 --seed 0
