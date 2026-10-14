#!/usr/bin/env python3
"""Build breeze_train manifests from the VieNeu [throatclearing] candidate run.

The run directory holds one row per *candidate* in samples/results.jsonl
(5,000 logical slots, 1..81 attempts each, every attempt reading the same
sentence). Training on all candidates would duplicate sentences and
overweight the voices that needed many retries, so this keeps at most one
candidate per slot:

  1. gate-v5-fallback-standard-30.jsonl status == "pass"
     (token reachability + audio activity; the run's final gate);
  2. acoustic-calibration-full-run-strict-v6-20261009.json
     acoustic_prediction == "nv" (ridge screen trained on human listening
     labels; the only check here that looks at whether the cue is audible);
  3. among those, the highest acoustic feature_score;
  4. drop duration / speaking-rate outliers (cap-length runaways, clips too
     short to hold a sentence plus a throat-clear).

Text is `original_text`, which carries the `[throatclearing]` tag in the same
`[tag]` syntax the nv01 checkpoints were trained on. Record shape follows the
nv01 manifest (Nonverbal_VLSP4_Mapped/metadata.jsonl): id, audio_path, text,
speaker, plus passthrough metadata that `breeze_train.manifest` ignores.

val.jsonl holds unseen *sentences* of the training voices (split by slot hash).
--holdout-voice keeps whole presets out of training into val_voices.jsonl, the only
split that measures generalization to voices the model never trained on.

Usage: build_manifest.py OUT_DIR [--run RUN_DIR] [--val-frac 0.02]
                          [--holdout-voice VOICE ...] [--holdout-per-voice 100]
Writes OUT_DIR/{train,val}.jsonl (+ val_voices.jsonl) and OUT_DIR/stats.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_RUN = Path(
    "/mnt/data/users/anhnct/non-verbal-TTS/20261008-vieneu-nv-active21-audit-v5-refill"
)
GATE_FILE = "gate-v5-fallback-standard-30.jsonl"
ACOUSTIC_FILE = "acoustic-calibration-full-run-strict-v6-20261009.json"
TAG = "[throatclearing]"
BRACKET_RE = re.compile(r"\[[^\]]*\]")

MIN_DURATION_S = 1.5
CAP_DURATION_S = 24.0  # generator's max length; clips at the cap are runaways
MIN_CHARS_PER_S = 5.0
MAX_CHARS_PER_S = 30.0


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_val(slot_id: str, val_frac: float) -> bool:
    digest = hashlib.sha256(slot_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_frac


def reject_reason(row: dict) -> str | None:
    duration = row["duration_s"]
    if duration < MIN_DURATION_S:
        return "too_short"
    if duration >= CAP_DURATION_S:
        return "length_cap"
    rate = len(row["base_text"]) / duration
    if rate < MIN_CHARS_PER_S:
        return "too_slow"
    if rate > MAX_CHARS_PER_S:
        return "too_fast"
    text = row["original_text"]
    if text.count(TAG) != 1 or set(BRACKET_RE.findall(text)) != {TAG}:
        return "tag_text"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--holdout-voice", action="append", default=[], metavar="VOICE",
                        help="VieNeu preset kept out of training entirely; its clips go to "
                             "val_voices.jsonl (at most --holdout-per-voice each). Repeatable.")
    parser.add_argument("--holdout-per-voice", type=int, default=100)
    args = parser.parse_args()

    samples_dir = args.run / "samples"
    results = read_jsonl(samples_dir / "results.jsonl")
    gate = {r["candidate_id"]: r["status"] for r in read_jsonl(args.run / GATE_FILE)}
    acoustic_doc = json.loads((args.run / ACOUSTIC_FILE).read_text())
    acoustic = {r["candidate_id"]: r for r in acoustic_doc["rows"]}

    missing = [r["candidate_id"] for r in results if r["candidate_id"] not in gate
               or r["candidate_id"] not in acoustic]
    if missing:
        print(f"ERROR: {len(missing)} candidates lack a gate/acoustic verdict, "
              f"e.g. {missing[:3]}", file=sys.stderr)
        return 1

    funnel: Counter = Counter()
    rejects: Counter = Counter()
    best: dict[str, tuple[float, dict]] = {}
    for row in results:
        cid = row["candidate_id"]
        funnel["candidates"] += 1
        if row["status"] != "ok":
            rejects["generation_status"] += 1
            continue
        if gate[cid] != "pass":
            rejects["gate_fail"] += 1
            continue
        verdict = acoustic[cid]
        if verdict["acoustic_prediction"] != "nv":
            rejects["acoustic_non_nv"] += 1
            continue
        reason = reject_reason(row)
        if reason:
            rejects[reason] += 1
            continue
        funnel["eligible_candidates"] += 1
        score = verdict["feature_score"]
        slot = row["slot_id"]
        if slot not in best or score > best[slot][0]:
            best[slot] = (score, row)

    holdout = set(args.holdout_voice)
    train: list[dict] = []
    val: list[dict] = []
    val_voices: list[dict] = []
    for slot in sorted(best):
        score, row = best[slot]
        audio_path = (samples_dir / row["audio"]).resolve()
        if not audio_path.is_file():
            print(f"ERROR: missing audio {audio_path}", file=sys.stderr)
            return 1
        record = {
            "id": f"vieneu_tc_{row['candidate_id']}",
            "audio_path": str(audio_path),
            "text": row["original_text"],
            "speaker": f"vieneu:{row['voice']}",
            "source": "vieneu_tc_active21_v5",
            "tags": row["tags"],
            "position": row["position"],
            "duration": row["duration_s"],
            "acoustic_score": round(score, 4),
            "source_dataset": row["source_dataset"],
        }
        if row["voice"] in holdout:
            val_voices.append(record)
        else:
            (val if is_val(slot, args.val_frac) else train).append(record)

    unknown = holdout - {r["speaker"].removeprefix("vieneu:") for r in train + val + val_voices}
    if unknown:
        print(f"ERROR: --holdout-voice not in the run: {sorted(unknown)}", file=sys.stderr)
        return 1
    capped: list[dict] = []
    for voice in sorted(holdout):
        rows = [r for r in val_voices if r["speaker"] == f"vieneu:{voice}"]
        capped += sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[: args.holdout_per_voice]
    val_voices = capped

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = [("train", train), ("val", val)] + ([("val_voices", val_voices)] if holdout else [])
    for name, records in splits:
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    kept = train + val + val_voices
    slots_total = len({r["slot_id"] for r in results})
    stats = {
        "run": str(args.run),
        "gate_file": GATE_FILE,
        "acoustic_file": ACOUSTIC_FILE,
        "filters": {
            "min_duration_s": MIN_DURATION_S,
            "cap_duration_s": CAP_DURATION_S,
            "chars_per_s": [MIN_CHARS_PER_S, MAX_CHARS_PER_S],
        },
        "candidates": funnel["candidates"],
        "candidate_rejects": dict(rejects),
        "eligible_candidates": funnel["eligible_candidates"],
        "slots_total": slots_total,
        "slots_kept": len(kept),
        "train": len(train),
        "val": len(val),
        "holdout_voices": sorted(holdout),
        "val_voices": len(val_voices),
        "hours": round(sum(r["duration"] for r in kept) / 3600, 2),
        "by_position": dict(Counter(r["position"] for r in kept)),
        "by_speaker": dict(Counter(r["speaker"] for r in kept).most_common()),
        "by_source_dataset": dict(Counter(r["source_dataset"] for r in kept)),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
