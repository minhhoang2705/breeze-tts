#!/usr/bin/env python3
"""Pick the fixed prompts train.py --sample-prompts renders at every checkpoint.

One prompt per tag position (beginning / middle / end) from each throatclearing
evalset built by eval/make_evalsets.py: tc_val_99 (VieNeu voice seen in training,
unseen sentence) and tc_clone_100 (non-VieNeu clone reference, the production
setting). Six prompts in tts-eval row format.

Usage: make_sample_prompts.py EVALSETS_DIR OUT_JSONL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

POSITIONS = ("beginning", "middle", "end")
FIELDS = ("id", "text", "ref_audio", "ref_text")


def main() -> int:
    evalsets_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    prompts = []
    for evalset in ("tc_val_99", "tc_clone_100"):
        rows = [json.loads(l) for l in (evalsets_dir / f"{evalset}.jsonl").open() if l.strip()]
        for position in POSITIONS:
            row = next(r for r in rows if r["position"] == position)
            prompts.append({"id": f"{evalset}__{position}", **{k: row[k] for k in FIELDS[1:]}})
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts))
    for p in prompts:
        print(p["id"], "|", p["text"][:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
