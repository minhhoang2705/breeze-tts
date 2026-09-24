"""Throwaway smoke test (phase 5, Task 5.3): scoping and gradient-flow gate.

Not a pytest test -- deleted in the phase 9 cleanup. Mechanically proves that
every parameter apply_policy("p1") claims to train is actually reachable by
autograd, and that nothing outside the intended scope is trainable.
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
from breeze_train.policy import apply_policy, enable_memory_savings, trainable_summary
from models.breeze import BreezeForConditionalGeneration

CKPT_DIR = Path("../breeze-tts-2")
CACHE_DIR = Path("data/codec_cache")
MANIFEST_PATH = Path("data/dev_manifest.jsonl")


def build_batch(tokenizer, config):
    records = {r.id: r for r in load_manifest(MANIFEST_PATH)}
    examples = []
    for record_id in ("r0", "r1"):
        record = records[record_id]
        target_codes = load_codes(CACHE_DIR, record.audio_path)
        ref_codes = load_codes(CACHE_DIR, record.ref_audio_path) if record.ref_audio_path else None
        examples.append(build_example(tokenizer, config, record, target_codes, ref_codes))
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=config)
    batch = collator(examples)
    return {k: v.to("cuda") for k, v in batch.items()}


def main() -> None:
    torch.cuda.reset_peak_memory_stats()

    model = BreezeForConditionalGeneration.from_pretrained(
        str(CKPT_DIR), dtype=torch.bfloat16, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR), fix_mistral_regex=False)
    base_config = model.config

    model = apply_policy(model, "p1")
    enable_memory_savings(model)
    model.to("cuda")
    model.train()

    summary = trainable_summary(model)
    print("trainable per_module:", summary["per_module"])
    print("trainable total (deduplicated):", summary["total"])

    backbone_class = type(model.base_model.model.backbone_model).__name__
    print(f"backbone_class={backbone_class}")

    violations = 0

    lora_param_count = summary["total"]
    print(f"lora_param_count={lora_param_count}")
    if lora_param_count != 48037888:
        violations += 1

    lora_outside_allowed_prefix = 0
    allowed_prefixes = ("base_model.model.backbone_model.", "base_model.model.depth_decoder.")
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if not name.startswith(allowed_prefixes):
                lora_outside_allowed_prefix += 1
    print(f"lora_outside_allowed_prefix={lora_outside_allowed_prefix}")
    violations += lora_outside_allowed_prefix

    frozen_tree_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad and (".text_encoder." in name or ".codec_model." in name):
            frozen_tree_trainable += 1
    print(f"frozen_tree_trainable={frozen_tree_trainable}")
    violations += frozen_tree_trainable

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )

    batch = build_batch(tokenizer, base_config)
    outputs = model(**batch)
    outputs.loss.backward()

    lora_a_grad_allzero_step1 = 0
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            if param.grad is None or bool(param.grad.abs().sum().item() == 0):
                lora_a_grad_allzero_step1 += 1
    print(f"lora_a_grad_allzero_step1={lora_a_grad_allzero_step1}  (information only, no pass condition)")

    optimizer.step()
    optimizer.zero_grad()

    batch = build_batch(tokenizer, base_config)
    outputs = model(**batch)
    outputs.loss.backward()

    trainable_grad_none = 0
    trainable_grad_allzero_step2 = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            trainable_grad_none += 1
        elif bool(param.grad.abs().sum().item() == 0):
            trainable_grad_allzero_step2 += 1

    print(f"trainable_grad_none={trainable_grad_none}")
    print(f"trainable_grad_allzero_step2={trainable_grad_allzero_step2}")
    violations += trainable_grad_none + trainable_grad_allzero_step2

    peak_vram_gib = torch.cuda.max_memory_reserved() / 2**30
    peak_vram_alloc_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak_vram_gib={peak_vram_gib}")
    print(f"peak_vram_alloc_gib={peak_vram_alloc_gib}")

    if violations != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
