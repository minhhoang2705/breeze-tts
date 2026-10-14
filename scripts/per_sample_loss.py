"""Per-sample loss of a checkpoint over a manifest, for data and memorization analysis.

Runs every record as its own batch (no padding effects), in eval mode, and
writes one JSONL row per record: the manifest row's fields plus `loss`,
`backbone_loss`, `depth_decoder_loss`. Prints means grouped by the
manifest's `position` and `speaker` fields (when present) and the records
with the highest loss -- the usual suspects for clips whose audio does not
match their text, or for what the model has not learned.

Comparing the same manifest across checkpoints (or train vs. held-out
manifests at one checkpoint) separates learning from memorizing.

Usage (repo root, training env; codec cache must already hold the manifest):
  python scripts/per_sample_loss.py BASE_DIR CHECKPOINT_DIR MANIFEST CACHE_DIR OUT.jsonl
CHECKPOINT_DIR may be a PEFT checkpoint (merged onto BASE_DIR) or a full model dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from breeze_train.collator import BreezeDataCollator  # noqa: E402
from breeze_train.dataset import BreezeTrainDataset  # noqa: E402
from breeze_train.export import load_finetuned  # noqa: E402


def group_means(rows: list[dict], field: str) -> dict[str, tuple[int, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if field in row:
            groups[str(row[field])].append(row["loss"])
    return {k: (len(v), sum(v) / len(v)) for k, v in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    model = load_finetuned(args.base_dir, args.checkpoint_dir).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_dir), fix_mistral_regex=False)
    dataset = BreezeTrainDataset(args.manifest, args.cache_dir, tokenizer, model.config)
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=model.config)
    raw = {r["id"]: r for r in (json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip())}

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f, torch.no_grad():
        for index, record in enumerate(dataset.records):
            batch = {k: v.to(args.device) for k, v in collator([dataset[index]]).items()}
            outputs = model(**batch)
            row = {
                **raw[record.id],
                "loss": outputs.loss.float().item(),
                "backbone_loss": outputs.backbone_loss.float().item(),
                "depth_decoder_loss": outputs.depth_decoder_loss.float().item(),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (index + 1) % 500 == 0:
                print(f"{index + 1}/{len(dataset)}", flush=True)

    mean = sum(r["loss"] for r in rows) / len(rows)
    print(f"n={len(rows)} mean_loss={mean:.4f} "
          f"mean_backbone={sum(r['backbone_loss'] for r in rows) / len(rows):.4f} "
          f"mean_depth={sum(r['depth_decoder_loss'] for r in rows) / len(rows):.4f} -> {args.out}")
    for field in ("position", "speaker"):
        groups = group_means(rows, field)
        if groups:
            print(f"by {field}: " + ", ".join(f"{k}={m:.3f} (n={n})" for k, (n, m) in groups.items()))
    print(f"top {args.top} by loss:")
    for row in sorted(rows, key=lambda r: r["loss"], reverse=True)[: args.top]:
        print(f"  {row['loss']:.3f}  {row['id']}  {row.get('text', '')[:90]}")


if __name__ == "__main__":
    main()
