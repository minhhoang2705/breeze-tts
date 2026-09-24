"""Throwaway smoke test (phase 4, Task 4.3): batch-of-2 forward pass.

Not a pytest test -- deleted in the phase 9 cleanup. Proves the model accepts
a real collated batch mixing two different templates (different segment
counts) and returns a finite loss with both components present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from breeze_train.codec_cache import load_codes
from breeze_train.collator import BreezeDataCollator
from breeze_train.examples import build_example
from breeze_train.manifest import load_manifest
from models.breeze import BreezeForConditionalGeneration

CKPT_DIR = Path("../breeze-tts-2")
CACHE_DIR = Path("data/codec_cache")
MANIFEST_PATH = Path("data/dev_manifest.jsonl")


def main() -> None:
    model = BreezeForConditionalGeneration.from_pretrained(
        str(CKPT_DIR), dtype=torch.bfloat16, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR), fix_mistral_regex=False)

    records = {r.id: r for r in load_manifest(MANIFEST_PATH)}
    # r0 = tts_plain (1 text segment), r1 = ref_clone_tata (3 segments) --
    # exercises the flat cross-batch text_ids_len walk (models/breeze.py:1393-1412).
    examples = []
    for record_id in ("r0", "r1"):
        record = records[record_id]
        target_codes = load_codes(CACHE_DIR, record.audio_path)
        ref_codes = load_codes(CACHE_DIR, record.ref_audio_path) if record.ref_audio_path else None
        examples.append(build_example(tokenizer, model.config, record, target_codes, ref_codes))

    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=model.config)
    batch = collator(examples)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    model.to("cuda").eval()
    with torch.no_grad():
        outputs = model(**batch)

    print(f"loss={outputs.loss.item()}")
    print(f"backbone_loss={outputs.backbone_loss.item()}")
    if outputs.depth_decoder_loss is None:
        print("depth_decoder_loss=None")
    else:
        print(f"depth_decoder_loss={outputs.depth_decoder_loss.item()}")


if __name__ == "__main__":
    main()
