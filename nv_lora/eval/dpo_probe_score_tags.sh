#!/usr/bin/env bash
# Score the probe's tagged evalsets for every seed with nv_gemini_audit (blind judge: it
# hears each tag position without being told the expected tag, labels it from
# laughter/breath/gasp/sigh/cough/throat_clear/sniff/other/none), then reduce to NV
# Placement Accuracy (NVPA) per compute_nvpa.py's strict accept-only definition. All
# evalsets go into ONE audit run (source "seed_<s>|<evalset>"), so one Gemini pass gives
# the full expected-tag x heard-label confusion picture across the four tags.
# Run LOCALLY (needs GEMINI_API_KEY + internet); each remote step is a single ssh call,
# same pattern as score_tags.sh (the checkpoint-sweep version). Reuses to_audit_metadata.py
# unmodified.
#   GEMINI_API_KEY=... bash nv_lora/eval/dpo_probe_score_tags.sh
# EVALSETS/SEEDS/RUN_NAME are overridable env vars (defaults: the four same-sentence tag
# sets x seeds 42..45, run dpo_probe_nv4). E.g. the throatclearing probe's extra seeds:
#   EVALSETS=tc_clone_100 SEEDS="46 47 48 49" RUN_NAME=dpo_probe_tc_extra bash ...
# Audit run: nv_gemini_audit/runs/<RUN_NAME> (summary.md + nvpa.jsonl locally,
# keep/drop manifests + summary.md on H100); nvpa_<RUN_NAME>.jsonl is pushed to H100 next
# to runs/ for dpo_probe_analyze_spread.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
E=$REPO/nv_lora/eval
EVALSETS="${EVALSETS:-tc_clone_100 sn_clone_100 br_clone_100 la_clone_100}"
SEEDS="${SEEDS:-42 43 44 45}"
RUN_NAME="${RUN_NAME:-dpo_probe_nv4}"
MODELS=""
for s in $SEEDS; do MODELS="$MODELS${MODELS:+ }seed_$s"; done

echo "== check generation is complete: [$EVALSETS] x [$MODELS]"
ssh -n H100 "timeout 60 bash -c 'for ev in $EVALSETS; do want=\$(grep -c . $E/evalsets/\$ev.jsonl); for m in $MODELS; do
  n=\$(ls $E/gen/\${m}__\$ev 2>/dev/null | grep -c \"\\.wav\$\" || true)
  [[ \$n -eq \$want ]] || { echo \"not ready: \${m}__\$ev \$n/\$want\"; exit 3; }
done; done; echo ready'"

echo "== build one audit metadata file on H100 covering every evalset"
ssh -n H100 "timeout 120 bash -c 'set -e; out=$E/audit_$RUN_NAME.jsonl; : > \$out; for ev in $EVALSETS; do
  python3 $E/to_audit_metadata.py $E \$ev $E/audit_${RUN_NAME}__\$ev.jsonl $MODELS
  cat $E/audit_${RUN_NAME}__\$ev.jsonl >> \$out; rm $E/audit_${RUN_NAME}__\$ev.jsonl
done; wc -l \$out'"

"$HERE/../../nv_gemini_audit/run_audit.sh" "$RUN_NAME" \
  --vlsp "$E/audit_$RUN_NAME.jsonl" --tags laughter,breathing,throatclearing,sniff \
  --per-tag 5000 --include-untagged 0 --seed 0

echo "== reduce to NVPA"
python3 "$HERE/compute_nvpa.py" "$HERE/../../nv_gemini_audit/runs/$RUN_NAME" \
  --tags "$HERE/../../nv_gemini_audit/tags.json" --out "$HERE/../../nv_gemini_audit/runs/$RUN_NAME/nvpa.jsonl"
cp "$HERE/../../nv_gemini_audit/runs/$RUN_NAME/nvpa.jsonl" "$HERE/nvpa_$RUN_NAME.jsonl"
echo "NVPA per sample -> $HERE/nvpa_$RUN_NAME.jsonl"

echo "== push NVPA to H100 (runs/seed_*/per_utt.jsonl live there, so the spread analysis runs there)"
scp "$HERE/nvpa_$RUN_NAME.jsonl" "H100:$E/nvpa_$RUN_NAME.jsonl"
echo "next: ssh -n H100 'timeout 120 python3 $E/dpo_probe_analyze_spread.py $E/runs $E/nvpa_$RUN_NAME.jsonl --seeds $SEEDS'"
