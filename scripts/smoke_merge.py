"""Throwaway smoke test (phase 8, Task 8.2b): prove the merge survived save/load.

Not a pytest test -- deleted in the phase 9 cleanup. Prints three losses (not
one) to isolate which stage any discrepancy comes from, per kongming's
counsel at the phase 3-6 -> 7-8 checkpoint:
  (a) unmerged PeftModel eval-mode loss on the training batch
  (b) merged-in-memory loss (right after merge_and_unload(), before save)
  (c) loss from the reloaded outputs/overfit/merged directory
`a - final_train_loss` isolates train-vs-eval mode; `b - a` isolates bf16
merge rounding; `c - b` isolates save/load. The gate is unchanged:
`|c - final_train_loss| <= 0.1`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peft
import torch
from transformers import AutoTokenizer

from breeze_train.codec_cache import load_codes
from breeze_train.collator import BreezeDataCollator
from breeze_train.examples import build_example
from breeze_train.manifest import load_manifest
from models.breeze import BreezeForConditionalGeneration

CKPT_DIR = Path("../breeze-tts-2")
CACHE_DIR = Path("data/codec_cache")
MANIFEST_PATH = Path("data/overfit_one.jsonl")
ADAPTER_DIR = Path("outputs/overfit/checkpoint-300")
MERGED_DIR = Path("outputs/overfit/merged")
TRAIN_LOG = Path("outputs/overfit/train.log")


def final_train_loss() -> float:
    log = TRAIN_LOG.read_text()
    matches = re.findall(r"'loss': ([0-9.eE+-]+)", log)
    return float(matches[-1])


def build_eval_batch(tokenizer, config):
    record = load_manifest(MANIFEST_PATH)[0]
    target_codes = load_codes(CACHE_DIR, record.audio_path)
    example = build_example(tokenizer, config, record, target_codes, None)
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=config)
    batch = collator([example])
    return {k: v.to("cuda") for k, v in batch.items()}


def eval_loss(model, batch) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    return outputs.loss.item()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR), fix_mistral_regex=False)

    # (a) unmerged PeftModel, eval mode, on the training batch.
    base = BreezeForConditionalGeneration.from_pretrained(
        str(CKPT_DIR), dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    peft_model = peft.PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    batch = build_eval_batch(tokenizer, peft_model.config)
    loss_unmerged = eval_loss(peft_model, batch)

    # (b) merged in memory (right after merge_and_unload(), before save).
    merged_in_memory = peft_model.merge_and_unload()
    loss_merged_in_memory = eval_loss(merged_in_memory, batch)

    del base, peft_model, merged_in_memory
    torch.cuda.empty_cache()

    # (c) reloaded from the exported directory.
    reloaded = BreezeForConditionalGeneration.from_pretrained(
        str(MERGED_DIR), dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    loss_reloaded = eval_loss(reloaded, batch)

    train_loss = final_train_loss()
    delta = abs(loss_reloaded - train_loss)

    print(f"unmerged_eval_loss={loss_unmerged}")
    print(f"merged_in_memory_loss={loss_merged_in_memory}")
    print(f"exported_loss={loss_reloaded}")
    print(f"final_train_loss={train_loss}")
    print(f"delta(a-train)={loss_unmerged - train_loss}")
    print(f"delta(b-a)={loss_merged_in_memory - loss_unmerged}")
    print(f"delta(c-b)={loss_reloaded - loss_merged_in_memory}")
    print(
        f"merge ok: exported_loss={loss_reloaded} final_train_loss={train_loss} "
        f"delta={delta}"
    )
    if delta > 0.1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
