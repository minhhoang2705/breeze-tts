#!/usr/bin/env python3
"""Convert a `data_path,label` CSV (e.g. exp10_train.csv) into an nv01-shape manifest.

Each CSV row is one clip: `data_path` is an absolute audio path on H100, `label` the
transcript carrying the VLSP `[tag]`s. Output rows: id, audio_path, text, speaker,
plus passthrough metadata (source, tags, duration) that `breeze_train.manifest` ignores.

speaker: every row in Nonverbal_VLSP4_Remap_v3 carries a real per-clip tag of the form
"{source}:{voice_or_file_id}" (confirmed for all 46,053 rows, every source bucket,
including non-VieNeu ones -- NOT the "S0" default). Continuing LoRA training from a
checkpoint fine-tuned on that manifest while using a blanket "S0" speaker for new rows
would (a) retrain the 1,484 rows this CSV shares with Remap_v3 under a different tag than
the one the base checkpoint learned them with, and (b) since breeze_infer production
clone-mode requests default to speaker="S0" too, teach "[S0]" to mean "many different
unrelated voices" instead of leaving it to the ref-audio clone signal. So:
  --speaker-ref REMAP_JSONL   exact speaker for rows whose audio_path is in that manifest
  --vieneu-run DIR            DIR/samples/results.jsonl supplies {audio (relative), voice}
                               for VieNeu rows not in --speaker-ref (same "vieneu:{voice}"
                               format Remap_v3 uses)
  anything else               falls back to "{source}:{file_id}", file_id = the leading
                               spk_NNNN token if present (e.g. nhandt23's train_spk_0000_*),
                               else the filename stem up to the first "_" (the random
                               per-clip id the Nonverbal/Nonverbal_38K generators use,
                               matching Remap_v3's own "nonverbal:<id>" convention)
Without --speaker-ref, every row gets "S0" (original, simpler behavior).

Rows are dropped (and counted) when the audio is missing/unreadable, the text is
empty after stripping tags, the label holds a bracket that is not one of the four
VLSP tags, or --drop-path is given and the row's audio_path matches (used to exclude
rows that overlap an eval set's ref/target audio).

Usage: csv_to_manifest.py CSV OUT_JSONL [--id-prefix exp10] [--speaker-ref REMAP_JSONL]
                           [--vieneu-run DIR] [--drop-path PATH ...]
Then:  split_manifest.py OUT_JSONL DATA_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

TAGS = ("laughter", "breathing", "sniff", "throatclearing")
BRACKET_RE = re.compile(r"\[([^\]]*)\]")
SPK_RE = re.compile(r"(spk_\d+)")

REMAP_SOURCE_BY_DIR = {
    "Nonverbal": "nonverbal",
    "Nonverbal_38K_Elevenlab_Synthetic": "nonverbal_38k",
}


def source_of(path: str) -> str:
    if "/vietnamese/" in path:
        return path.split("/vietnamese/", 1)[1].split("/", 1)[0]
    if "/non-verbal-TTS/" in path and "vieneu" in path:
        return "vieneu_tc_active21_v5"
    parts = Path(path).parts
    return "/".join(parts[4:-1]) if len(parts) > 5 else str(Path(path).parent)


def load_speaker_ref(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("speaker"):
                    out[r["audio_path"]] = r["speaker"]
    return out


def load_vieneu_voices(run_dir: Path | None) -> dict[str, str]:
    """-> {absolute audio path suffix match key: voice}, keyed by the audio path's tail
    (run_dir/samples/<audio>), since results.jsonl stores `audio` relative to samples/."""
    if run_dir is None:
        return {}
    out = {}
    results = run_dir / "samples" / "results.jsonl"
    with results.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                abs_path = str(run_dir / "samples" / r["audio"])
                out[abs_path] = r["voice"]
    return out


def derive_speaker(path: str, csv_source: str, vieneu_voices: dict[str, str]) -> str:
    if path in vieneu_voices:
        return f"vieneu:{vieneu_voices[path]}"
    remap_source = REMAP_SOURCE_BY_DIR.get(csv_source, Path(csv_source).name)
    stem = Path(path).stem
    m = SPK_RE.search(stem)
    file_id = m.group(1) if m else stem.split("_", 1)[0]
    return f"{remap_source}:{file_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--id-prefix", default="exp10")
    parser.add_argument("--speaker-ref", type=Path, default=None)
    parser.add_argument("--vieneu-run", type=Path, default=None)
    parser.add_argument("--drop-path", action="append", default=[])
    args = parser.parse_args()

    speaker_ref = load_speaker_ref(args.speaker_ref)
    vieneu_voices = load_vieneu_voices(args.vieneu_run)
    drop_paths = set(args.drop_path)

    drops: Counter = Counter()
    seen: set[str] = set()
    out_rows = []
    speaker_source = Counter()
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path, text = row["data_path"].strip(), row["label"].strip()
            if path in drop_paths:
                drops["excluded (eval-set overlap)"] += 1
                continue
            if path in seen:
                drops["duplicate audio path"] += 1
                continue
            seen.add(path)
            brackets = BRACKET_RE.findall(text)
            if any(b not in TAGS for b in brackets):
                drops["non-VLSP bracket in label"] += 1
                continue
            if not BRACKET_RE.sub("", text).strip(" .,!?"):
                drops["empty text"] += 1
                continue
            try:
                duration = round(sf.info(path).duration, 3)
            except Exception:  # missing file, permission, undecodable
                drops["audio missing/unreadable"] += 1
                continue
            source = source_of(path)
            if path in speaker_ref:
                speaker = speaker_ref[path]
                speaker_source["joined from --speaker-ref"] += 1
            elif args.speaker_ref is not None:
                speaker = derive_speaker(path, source, vieneu_voices)
                speaker_source["derived (new row)"] += 1
            else:
                speaker = "S0"
                speaker_source["S0 (no --speaker-ref given)"] += 1
            out_rows.append({
                "id": f"{args.id_prefix}_{hashlib.sha256(path.encode()).hexdigest()[:16]}",
                "audio_path": path,
                "text": text,
                "speaker": speaker,
                "source": source,
                "tags": brackets,
                "duration": duration,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    print(json.dumps({
        "kept": len(out_rows),
        "hours": round(sum(r["duration"] for r in out_rows) / 3600, 3),
        "dropped": dict(drops),
        "speaker_source": dict(speaker_source),
        "tags": dict(Counter(t for r in out_rows for t in r["tags"]).most_common()),
        "sources": dict(Counter(r["source"] for r in out_rows).most_common()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
