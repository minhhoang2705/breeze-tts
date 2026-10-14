#!/usr/bin/env python3
"""Oversample manifest rows whose audio was NOT part of a reference manifest.

Used when a new training CSV mostly overlaps data a checkpoint was already
fine-tuned on (e.g. exp10_train.csv vs. Nonverbal_VLSP4_Remap_v3, which
nv_lora02/checkpoint-4400 already trained on): rows whose audio_path is new
get repeated --factor times (each copy a distinct manifest id, same
audio_path/text) so they carry proportionally more weight in an epoch without
discarding the already-seen rows that still provide tag/voice diversity.
Repeats share one codec-cache entry (keyed by audio content hash), so the
--factor has no cache-size cost.

Apply this to a train.jsonl AFTER split_manifest.py, never before: splitting
first keeps val.jsonl a clean, non-duplicated sample.

Usage: oversample_new.py TRAIN_JSONL REF_MANIFEST OUT_JSONL --factor 4
  TRAIN_JSONL  nv01-shape manifest (needs "id", "audio_path")
  REF_MANIFEST nv01-shape manifest/jsonl to treat as "already trained on"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("ref_manifest", type=Path)
    parser.add_argument("out_jsonl", type=Path)
    parser.add_argument("--factor", type=int, default=4, help="repeat count for new rows")
    args = parser.parse_args()
    if args.factor < 1:
        sys.exit("--factor must be >= 1")

    with args.ref_manifest.open() as f:
        ref_paths = {json.loads(line)["audio_path"] for line in f if line.strip()}

    with args.train_jsonl.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out_rows = []
    n_old, n_new_unique = 0, 0
    for row in rows:
        if row["audio_path"] in ref_paths:
            out_rows.append(row)
            n_old += 1
        else:
            n_new_unique += 1
            for k in range(args.factor):
                copy = dict(row, id=f"{row['id']}_ovs{k}")
                out_rows.append(copy)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    print(json.dumps({
        "already_trained_rows": n_old,
        "new_unique_rows": n_new_unique,
        "new_repeated_rows": n_new_unique * args.factor,
        "total_out_rows": len(out_rows),
        "new_share": round(n_new_unique * args.factor / len(out_rows), 3) if out_rows else 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
