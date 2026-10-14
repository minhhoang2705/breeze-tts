#!/usr/bin/env python3
"""Build the evalsets (tts-eval row schema) used by nv_lora checkpoint eval and the
DPO signal probe (nv_lora/eval/dpo_probe_*.{py,sh}).

tc_val_99      held-out VieNeu sentences (nv_lora/data/val.jsonl). Voices are the same
               21 VieNeu presets seen in training (val is split by slot, not voice), so
               this measures the tag on unseen *sentences* in seen voices. Each row is
               a zero-shot clone prompted with one fixed train clip of the same voice.
               Every VieNeu clip contains a throat-clear, so the reference itself does
               too; that priming is identical for every model compared.
tc_clone_100   100 vi_clone_300 rows (non-VieNeu voices, the production clone setting)
               with [throatclearing] inserted at the beginning / middle / end, cycling.
               Measures whether the tag generalizes beyond the training voices.
sn_clone_100   the SAME 100 sentences, positions, voices and refs as tc_clone_100 with
br_clone_100   [sniff] / [breathing] / [laughter] instead. Text, voice and position are
la_clone_100   held constant across the four sets, so any difference in what is heard
               at the tag is caused by the tag alone — the controlled comparison the
               4-tag DPO probe needs to measure tag confusion (e.g. [sniff] heard as
               breath, [throatclearing] heard as cough).
vi_probe_100   100 further vi_clone_300 rows (non-VieNeu voices, untagged, disjoint from
               the *_clone_100 sentences) used only by the DPO probe to measure
               plain-reading spread (1-WER/spk_sim) independent of any NV tag.

Row schema matches tts-eval evalsets: id, source, speaker, text, ref_audio, ref_text,
target_audio (tc_val_99 only), plus position (all tagged sets).

Usage: make_evalsets.py OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

NV_LORA = Path("/mnt/data/users/minhth21/breeze-tts/nv_lora")
VI_CLONE = Path("/mnt/data/users/anhnct/tts-eval/evalsets/vi_clone_300.jsonl")
TAG = "[throatclearing]"  # tc_val_99's tag (VieNeu data is throatclearing-only)
# prefix -> tag for the four same-sentence clone evalsets; "tc" must stay first and keep
# its exact ids/text — run_gen.sh outputs and audits already exist for tc_clone_100.
CLONE_TAGS = {"tc": "[throatclearing]", "sn": "[sniff]", "br": "[breathing]", "la": "[laughter]"}
ANY_TAG_RE = re.compile(r"\[[a-z_]+\]")
TRAILING_PUNCT_RE = re.compile(r"[\s.!?…,;:]+$")
N_CLONE = 100
N_PROBE = 100
POSITIONS = ("beginning", "middle", "end")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def insert_tag(text: str, position: str, tag: str) -> str:
    text = text.strip()
    if position == "beginning":
        return f"{tag} {text}"
    if position == "end":
        return f"{TRAILING_PUNCT_RE.sub('', text)} {tag}"
    words = text.split()
    cut = len(words) // 2
    return " ".join(words[:cut] + [tag] + words[cut:])


def tc_val_rows() -> list[dict]:
    train = read_jsonl(NV_LORA / "data/train.jsonl")
    refs: dict[str, dict] = {}
    for row in sorted(train, key=lambda r: r["id"]):
        if row["speaker"] not in refs and 3.0 <= row["duration"] <= 10.0:
            refs[row["speaker"]] = row
    rows = []
    for row in read_jsonl(NV_LORA / "data/val.jsonl"):
        ref = refs[row["speaker"]]
        rows.append({
            "id": row["id"],
            "source": "vieneu_tc_val",
            "speaker": row["speaker"],
            "text": row["text"],
            "position": row["position"],
            "target_audio": row["audio_path"],
            "ref_id": ref["id"],
            "ref_audio": ref["audio_path"],
            "ref_text": ref["text"],
        })
    return rows


def clone_rows(prefix: str, tag: str) -> list[dict]:
    """100 vi_clone_300 rows with `tag` inserted; sentence choice and position cycling do
    not depend on the tag, so every prefix gets the same sentences at the same positions."""
    candidates = [r for r in read_jsonl(VI_CLONE) if not ANY_TAG_RE.search(r["text"])
                  and len(r["text"].split()) >= 6]
    chosen = sorted(candidates, key=lambda r: sha(r["id"]))[:N_CLONE]
    rows = []
    for i, row in enumerate(sorted(chosen, key=lambda r: r["id"])):
        position = POSITIONS[i % len(POSITIONS)]
        rows.append({
            "id": f"{prefix}_{row['id']}",
            "source": f"{prefix}_clone:{row['source']}",
            "speaker": row["speaker"],
            "text": insert_tag(row["text"], position, tag),
            "position": position,
            "ref_id": row["ref_id"],
            "ref_audio": row["ref_audio"],
            "ref_text": row["ref_text"],
        })
    return rows


def vi_probe_rows(exclude_ids: set[str]) -> list[dict]:
    """100 further vi_clone_300 rows, untagged, disjoint from the *_clone_100 sentences —
    same sha256(id) selection convention as clone_rows(), just over the remaining pool."""
    candidates = [r for r in read_jsonl(VI_CLONE) if r["id"] not in exclude_ids
                  and not ANY_TAG_RE.search(r["text"]) and len(r["text"].split()) >= 6]
    chosen = sorted(candidates, key=lambda r: sha(r["id"]))[:N_PROBE]
    rows = []
    for row in sorted(chosen, key=lambda r: r["id"]):
        rows.append({
            "id": row["id"],
            "source": f"vi_probe:{row['source']}",
            "speaker": row["speaker"],
            "text": row["text"],
            "ref_id": row["ref_id"],
            "ref_audio": row["ref_audio"],
            "ref_text": row["ref_text"],
        })
    return rows


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    clones = {prefix: clone_rows(prefix, tag) for prefix, tag in CLONE_TAGS.items()}
    clone_ids = {row["id"].removeprefix("tc_") for row in clones["tc"]}
    vi_probe = vi_probe_rows(clone_ids)
    overlap = clone_ids & {row["id"] for row in vi_probe}
    if overlap:
        print(f"ERROR: vi_probe_100 overlaps the clone sets: {sorted(overlap)[:3]}", file=sys.stderr)
        return 1
    evalsets = [("tc_val_99", tc_val_rows(), TAG)]
    evalsets += [(f"{prefix}_clone_100", clones[prefix], tag) for prefix, tag in CLONE_TAGS.items()]
    evalsets.append(("vi_probe_100", vi_probe, None))
    for name, rows, tag in evalsets:
        for row in rows:
            tag_ok = row["text"].count(tag) == 1 if tag else not ANY_TAG_RE.search(row["text"])
            if not tag_ok or not Path(row["ref_audio"]).is_file():
                print(f"ERROR: bad row {row['id']}", file=sys.stderr)
                return 1
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        by_pos: dict[str, int] = {}
        for row in rows:
            by_pos[row.get("position", "n/a")] = by_pos.get(row.get("position", "n/a"), 0) + 1
        print(name, len(rows), by_pos)
        for row in rows[:2]:
            print("  ", row["text"][:140])
    return 0


if __name__ == "__main__":
    sys.exit(main())
