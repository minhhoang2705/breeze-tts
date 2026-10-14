#!/usr/bin/env python3
"""Per-model verdict breakdown from a nv_gemini_audit results.jsonl, matching the table
format in docs/NV_LORA01_TRAINING_REPORT.md §5 (keep/cough/none/other/breath/gate-fail
counts per model). decide.py's summary.md only aggregates by tag, not by model (source
is "<model>|<evalset>" but the per-source table there is gate-fail-rate only), so this
reads results.jsonl directly: id = "<model>__<evalset>__<item_id>", one row per item,
each with a transcript gate (`transcript.exact_match`) and one or more `markers`
(`label` in {throat_clear, cough, breath, sigh, sniff, laughter, other, none}).

Usage: per_model_tag_table.py RESULTS_JSONL
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> int:
    path = sys.argv[1]
    by_model: dict[str, Counter] = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            model = row["id"].split("__", 1)[0]
            c = by_model[model]
            c["total"] += 1
            if row.get("status") != "ok":
                c["error"] += 1
                continue
            if not row["transcript"]["exact_match"]:
                c["gate_fail"] += 1
                continue
            for m in row["markers"]:
                label = m["label"] if m["heard"] else "none"
                c[label] += 1

    labels = ["throat_clear", "cough", "none", "other", "breath", "sigh", "sniff", "laughter", "gate_fail", "error"]
    header = ["model", "n"] + labels
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for model in sorted(by_model):
        c = by_model[model]
        row = [model, str(c["total"])] + [str(c.get(l, 0)) for l in labels]
        print(" | ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
