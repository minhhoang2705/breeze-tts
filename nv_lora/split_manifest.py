#!/usr/bin/env python3
"""Split an nv01-format manifest (id, audio_path, text, speaker, ...) into train/val.

Used for nv_lora02, whose input Nonverbal_VLSP4_Remap_v3/metadata.jsonl is already
in the nv01 record shape (text carries the [tag]s, speaker is "<source>:<voice>"),
so no row is filtered here: the remap already decided what to keep. val holds a
fixed sha256(id) slice, stable across reruns and independent of row order.

Usage: split_manifest.py MANIFEST OUT_DIR [--val-frac 0.006]
Writes OUT_DIR/{train,val}.jsonl and OUT_DIR/stats.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def is_val(row_id: str, val_frac: float) -> bool:
    digest = hashlib.sha256(row_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_frac


def summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "hours": round(sum(r.get("duration") or 0 for r in rows) / 3600, 3),
        "by_tag": dict(Counter(t for r in rows for t in r.get("tags", [])).most_common()),
        "by_source": dict(Counter(r.get("source") for r in rows).most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.006)
    args = parser.parse_args()

    with args.manifest.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    train = [r for r in rows if not is_val(r["id"], args.val_frac)]
    val = [r for r in rows if is_val(r["id"], args.val_frac)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in split)
    stats = {
        "input": str(args.manifest),
        "val_frac": args.val_frac,
        "train": summary(train),
        "val": summary(val),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps({k: stats[k]["rows"] for k in ("train", "val")}), f"of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
