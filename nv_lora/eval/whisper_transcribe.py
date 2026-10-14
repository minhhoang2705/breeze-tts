#!/usr/bin/env python3
"""Transcribe an evalset's generated wavs with Whisper large-v3 (offline weights),
for senvoice_metrics.py's EN-WER / CS-* scoring.

Run in an env with transformers+torch+torchaudio+accelerate (breeze312 on H100 has all
four); weights are local (H100 has no internet) at --model-dir, default
/mnt/data/users/minhth21/models/whisper-large-v3 (openai/whisper-large-v3, config/
tokenizer/model.safetensors only, no internet fetch -- see nv_lora/eval/senvoice_metrics.py
module docstring for why Whisper is used instead of Senvoice's ElevenLabs+Gemini).

Usage:
  whisper_transcribe.py --wavs DIR --evalset E.jsonl --language en --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--evalset", type=Path, required=True)
    parser.add_argument("--language", required=True, choices=("en", "vi"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/data/users/minhth21/models/whisper-large-v3"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=str(args.model_dir),
        torch_dtype=torch.float16,
        device=args.device,
        chunk_length_s=30,
        generate_kwargs={"language": args.language, "task": "transcribe"},
    )

    items = [json.loads(line) for line in args.evalset.read_text().splitlines() if line.strip()]
    paths, ids = [], []
    for item in items:
        wav = args.wavs / f"{item['id']}.wav"
        if wav.is_file() and wav.stat().st_size > 0:
            paths.append(str(wav))
            ids.append(item["id"])
    print(f"{len(ids)}/{len(items)} wavs found in {args.wavs}", flush=True)

    rows = []
    done = 0
    for out in asr(paths, batch_size=args.batch_size):
        rows.append({"id": ids[done], "hyp": out["text"].strip()})
        done += 1
        if done % 20 == 0 or done == len(ids):
            print(f"transcribed {done}/{len(ids)}", flush=True)

    missing = set(item["id"] for item in items) - set(ids)
    rows.extend({"id": i, "hyp": None} for i in sorted(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {args.out} ({len(rows)} rows, {len(missing)} missing)")


if __name__ == "__main__":
    main()
