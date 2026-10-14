#!/usr/bin/env python3
"""Per-model keep-rate by tag position (beginning/middle/end/before_comma/after_comma),
joining a nv_gemini_audit results.jsonl back to its evalset's `position` field. Matches
the breakdown in docs/NV_LORA01_TRAINING_REPORT.md §5.1.

Usage: per_model_position_table.py RESULTS_JSONL EVALSET_JSONL
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> int:
    results_path, evalset_path = sys.argv[1], sys.argv[2]
    position_by_id = {}
    with open(evalset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                position_by_id[row["id"]] = row["position"]

    by_model_position: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            model, _evalset, item_id = row["id"].split("__", 2)
            position = position_by_id.get(item_id, "?")
            c = by_model_position[(model, position)]
            c["total"] += 1
            if row.get("status") != "ok" or not row["transcript"]["exact_match"]:
                c["gate_fail"] += 1
                continue
            for m in row["markers"]:
                label = m["label"] if m["heard"] else "none"
                c["keep" if label == "throat_clear" else label] += 1

    positions = sorted({p for _, p in by_model_position})
    models = sorted({m for m, _ in by_model_position})
    print("model | " + " | ".join(positions))
    print("--- | " + " | ".join(["---"] * len(positions)))
    for model in models:
        cells = []
        for pos in positions:
            c = by_model_position.get((model, pos), Counter())
            n = c["total"]
            keep = c["keep"]
            cells.append(f"{keep}/{n}" if n else "-")
        print(f"{model} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
