#!/usr/bin/env python3
"""Paired comparison of vi_clone_300 reading metrics against the regenerated baseline.

Every model reads the same 300 prompts, so resampling *prompts* and recomputing each
metric for model and baseline on the same resample cancels prompt difficulty; the
95% CI of the difference is far tighter than comparing two independent means.
Metrics match tts_eval's summary.json: 1-WER is corpus-level (1 - sum word_errors /
sum ref_words), pMOS is mean DNSMOS OVRL, spk_sim is mean ECAPA cosine vs ref_audio.
Reads runs/<model>/per_utt.jsonl written by tts_eval.evaluate.

Usage: paired_reading.py EVAL_DIR [BASELINE] [MODEL ...]
(default BASELINE: baseline_8000_regen; default MODELS: the nv_lora01 checkpoint sweep)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DEFAULT_MODELS = ("baseline_8000", "lora_200", "lora_400", "lora_600", "lora_800")


def one_minus_wer(rows: list[dict]) -> float:
    return 1 - sum(r["word_errors"] for r in rows) / sum(r["ref_words"] for r in rows)


def pmos(rows: list[dict]) -> float:
    return sum(r["dnsmos_ovrl"] for r in rows) / len(rows)


def spk_sim(rows: list[dict]) -> float:
    return sum(r["spk_sim"] for r in rows) / len(rows)


METRICS = {"1-WER": one_minus_wer, "pMOS": pmos, "spk_sim": spk_sim}


def load(eval_dir: Path, model: str) -> dict[str, dict]:
    path = eval_dir / "runs" / model / "per_utt.jsonl"
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def main() -> int:
    eval_dir = Path(sys.argv[1])
    base_name = sys.argv[2] if len(sys.argv) > 2 else "baseline_8000_regen"
    models = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT_MODELS
    base = load(eval_dir, base_name)
    rng = random.Random(0)
    print(f"paired vs {base_name}; model - baseline, bootstrap over prompts (5000), 95% CI; * = CI excludes 0")
    for model in models:
        rows = load(eval_dir, model)
        ids = sorted(i for i in base if i in rows and not base[i]["missing"] and not rows[i]["missing"])
        resamples = [rng.choices(ids, k=len(ids)) for _ in range(5000)]
        cells = []
        for name, fn in METRICS.items():
            diff = fn([rows[i] for i in ids]) - fn([base[i] for i in ids])
            boots = sorted(fn([rows[i] for i in s]) - fn([base[i] for i in s]) for s in resamples)
            lo, hi = boots[125], boots[4874]
            cells.append(f"{name} {diff:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{model:14s} n={len(ids)} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
