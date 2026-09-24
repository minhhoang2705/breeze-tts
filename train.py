"""Training CLI for Breeze TTS 2 LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

# Import before AutoConfig.from_pretrained: this registers the "breeze"
# model type with AutoConfig as a side effect (models/breeze_config.py:54).
# A bare AutoConfig.from_pretrained(...) in a fresh process otherwise raises
# "Unrecognized model type".
import models.breeze  # noqa: F401
from breeze_train.collator import BreezeDataCollator
from breeze_train.dataset import BreezeTrainDataset
from breeze_train.policy import apply_policy, enable_memory_savings
from breeze_train.trainer import BreezeTrainer
from models.breeze import BreezeForConditionalGeneration

logger = logging.getLogger("breeze_train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Breeze TTS 2 (LoRA fine-tuning)")
    parser.add_argument("model", type=Path, help="Checkpoint directory")
    parser.add_argument("--config", type=Path, required=True, help="JSON training config")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", default=None, help="Overrides the config's policy")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None, help="Overrides the config value")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    policy = args.policy or cfg.get("policy", "p1")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(str(args.output_dir), "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    model_config = AutoConfig.from_pretrained(str(args.model))
    model_config.depth_header_loss_weight = cfg.get("depth_header_loss_weight", 1.0)

    model = BreezeForConditionalGeneration.from_pretrained(
        str(args.model),
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), fix_mistral_regex=False)

    model = apply_policy(
        model,
        policy,
        lora_rank=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )
    enable_memory_savings(model)

    dataset = BreezeTrainDataset(args.manifest, args.cache_dir, tokenizer, model.config)
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=model.config)

    save_steps = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 300)
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps", -1)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=False,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=save_steps,
        max_steps=max_steps,
        report_to=cfg.get("report_to", []),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        seed=args.seed,
    )

    trainer = BreezeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    # Trainer's default save_strategy="steps" writes a checkpoint only when
    # global_step % save_steps == 0 and performs no unconditional final
    # save, so a short run can finish having written nothing at all.
    checkpoint_dir = os.path.join(str(args.output_dir), f"checkpoint-{trainer.state.global_step}")
    trainer.save_model(checkpoint_dir)

    peak_vram_gib = torch.cuda.max_memory_reserved() / 2**30
    peak_vram_alloc_gib = torch.cuda.max_memory_allocated() / 2**30
    logger.info(f"peak_vram_gib={peak_vram_gib}")
    logger.info(f"peak_vram_alloc_gib={peak_vram_alloc_gib}")
    print(f"peak_vram_gib={peak_vram_gib}")
    print(f"peak_vram_alloc_gib={peak_vram_alloc_gib}")
    print(f"saved {checkpoint_dir}")


if __name__ == "__main__":
    main()
