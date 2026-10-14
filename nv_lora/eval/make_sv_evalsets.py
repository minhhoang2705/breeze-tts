#!/usr/bin/env python3
"""Build the sv_* evalsets (tts-eval/breeze_generate.py row schema) from anhnct's
"Senvoice Evaluation" dataset
(/mnt/data/users/anhnct/Higgs-tts-3-finetune/benchmark_portal_20260912/data), read-only.

Five evalsets, 200 rows each (the first 200 of each 1000-row Senvoice metadata.jsonl,
in file order):
  sv_vi_200          data/vi_wer/metadata.jsonl              VI-WER
  sv_en_200          data/en_wer/metadata.jsonl               EN-WER
  sv_cs_latin_200     data/cs_latin/metadata.jsonl             CS-Latin
  sv_cs_ipa_200       data/cs_ipa/metadata.jsonl               CS-IPA
  sv_cs_translit_200  data/cs_transliteration/metadata.jsonl    CS-Transliteration

Every row is a zero-shot clone prompt (ref_audio/ref_text -> text), cycling through the
100 Senvoice reference prompts (all the same speaker, Thuy_Duyen, from
data/vi_wer/reference_prompts/ -- identical audio is duplicated per metric subfolder
upstream, so this script reads only the vi_wer copy). CS rows additionally carry
raw_text/english_tokens (senvoice_metrics.py's CS-WER/EN-recall scoring inputs); VI/EN
rows carry raw_text too, though it is identical to text for those two forms in this
dataset -- Senvoice's "raw_text" is the always-Latin scoring reference, "text" is what
gets synthesized (IPA-marked for cs_ipa, Vietnamese-phonetic for cs_transliteration).

Usage: make_sv_evalsets.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SENVOICE_DATA = Path("/mnt/data/users/anhnct/Higgs-tts-3-finetune/benchmark_portal_20260912/data")
N = 200
FORMS = {
    "sv_vi_200": "vi_wer",
    "sv_en_200": "en_wer",
    "sv_cs_latin_200": "cs_latin",
    "sv_cs_ipa_200": "cs_ipa",
    "sv_cs_translit_200": "cs_transliteration",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = read_jsonl(SENVOICE_DATA / "vi_wer" / "reference_prompts" / "metadata.jsonl")
    ref_audio_dir = SENVOICE_DATA / "vi_wer" / "reference_prompts" / "audio"
    for ref in refs:
        if not (ref_audio_dir / f"{ref['id']}.wav").is_file():
            print(f"ERROR: missing reference audio {ref['id']}", file=sys.stderr)
            return 1

    for evalset_name, source in FORMS.items():
        source_rows = read_jsonl(SENVOICE_DATA / source / "metadata.jsonl")[:N]
        rows = []
        for i, row in enumerate(source_rows):
            ref = refs[i % len(refs)]
            out_row = {
                "id": row["id"],
                "source": source,
                "speaker": ref["speaker"],
                "text": row["text"],
                "raw_text": row["raw_text"],
                "ref_id": ref["id"],
                "ref_audio": str(ref_audio_dir / f"{ref['id']}.wav"),
                "ref_text": ref["text"],
            }
            if "english_tokens" in row:
                out_row["english_tokens"] = row["english_tokens"]
            rows.append(out_row)
        if len(rows) != N:
            print(f"ERROR: {evalset_name} has {len(rows)} rows, expected {N}", file=sys.stderr)
            return 1
        write_jsonl(out_dir / f"{evalset_name}.jsonl", rows)
        print(evalset_name, len(rows), "e.g.", rows[0]["id"], "->", rows[0]["ref_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
