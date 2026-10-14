#!/usr/bin/env python3
"""Turn generated eval audio into a VLSP-style metadata.jsonl for nv_gemini_audit.

One row per (model, item) of one evalset, with source "<model>|<evalset>", so
decide.py's per-source tables become the per-model comparison. For tc_val_99 it
also adds the original VieNeu clips as source "vieneu_gt|tc_val_99": the judge's
acceptance rate on the training data itself, i.e. the ceiling a model can be
expected to reach.

Usage: to_audit_metadata.py EVAL_DIR EVALSET OUT_JSONL MODEL [MODEL ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    eval_dir, evalset, out_path, models = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
    items = [json.loads(l) for l in (eval_dir / "evalsets" / f"{evalset}.jsonl").open() if l.strip()]
    rows, missing = [], []
    for model in models:
        for item in items:
            wav = eval_dir / "gen" / f"{model}__{evalset}" / f"{item['id']}.wav"
            if not wav.is_file():
                missing.append(str(wav))
                continue
            rows.append({"id": f"{model}__{evalset}__{item['id']}", "source": f"{model}|{evalset}",
                         "text": item["text"], "audio_path": str(wav), "duration": None})
    if evalset == "tc_val_99":
        for item in items:
            rows.append({"id": f"vieneu_gt__{evalset}__{item['id']}", "source": f"vieneu_gt|{evalset}",
                         "text": item["text"], "audio_path": item["target_audio"], "duration": None})
    if missing:
        print(f"ERROR: {len(missing)} generated wavs missing, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
