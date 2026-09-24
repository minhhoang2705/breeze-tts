"""Diagnostic: warm-restart training from outputs/vietnamese_scale3/checkpoint-3000.

Loads the TRAINED adapter weights (not a fresh LoRA init) and continues
training with a fresh cosine schedule (LR back to peak) on the same 30k
manifest. Throwaway diagnostic script -- tests whether scale3's apparent
plateau at steps 2700-3000 was a schedule artifact (LR decayed to ~2.9e-11)
or a genuine capacity ceiling. If val loss keeps improving here, it was
the schedule; if it stays flat under fresh LR, that is real evidence of a
ceiling.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peft
import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

import models.breeze  # noqa: F401 -- registers the "breeze" AutoConfig architecture
from breeze_train.collator import BreezeDataCollator
from breeze_train.dataset import BreezeTrainDataset
from breeze_train.policy import enable_memory_savings
from breeze_train.trainer import BreezeTrainer
from models.breeze import BreezeForConditionalGeneration

logger = logging.getLogger("breeze_train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="outputs/vietnamese_scale3/checkpoint-3000")
    parser.add_argument("--manifest", default="data/vivoice_scale3/train.jsonl")
    parser.add_argument("--cache-dir", default="data/codec_cache")
    parser.add_argument("--output-dir", default="outputs/vietnamese_scale3_restart")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=43)  # different seed: fresh data order
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    base_dir = "../breeze-tts-2"
    model_config = AutoConfig.from_pretrained(base_dir)
    model_config.depth_header_loss_weight = 1.0

    model = BreezeForConditionalGeneration.from_pretrained(
        base_dir, config=model_config, dtype=torch.bfloat16, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_dir, fix_mistral_regex=False)

    # Load the TRAINED adapter weights, not a fresh LoRA init, with
    # is_trainable=True so gradients still flow -- this is what makes it a
    # continuation rather than a from-scratch run.
    model = peft.PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=True)
    enable_memory_savings(model)

    dataset = BreezeTrainDataset(args.manifest, args.cache_dir, tokenizer, model.config)
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=model.config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        bf16=True,
        gradient_checkpointing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,  # already warm; a fresh warmup would waste the restart
        logging_steps=30,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = BreezeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{trainer.state.global_step}")
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
