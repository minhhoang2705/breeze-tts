"""Training CLI for Breeze TTS 2 LoRA fine-tuning.

Single GPU: `python train.py ...`. Multi-GPU (HF Accelerate): `accelerate
launch --config_file configs/accelerate_<multi_gpu|fsdp|deepspeed_zero2|
deepspeed_zero3>.yaml --num_processes <N> train.py ...`; HF Trainer picks the
DDP/FSDP/DeepSpeed backend up from the launcher. The config's
`gradient_accumulation_steps` is the *single-GPU* value: it is divided by the
number of processes so the global batch (and therefore the tuned learning
rate and the meaning of `max_steps`/`save_steps`) stays identical while each
step finishes ~N× faster.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import accelerate
import peft
import torch
import transformers
from accelerate import PartialState
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# Import before AutoConfig.from_pretrained: this registers the "breeze"
# model type with AutoConfig as a side effect (models/breeze_config.py:54).
# A bare AutoConfig.from_pretrained(...) in a fresh process otherwise raises
# "Unrecognized model type".
import models.breeze  # noqa: F401
from breeze_train.collator import BreezeDataCollator
from breeze_train.dataset import BreezeTrainDataset
from breeze_train.policy import apply_policy, enable_memory_savings, trainable_summary, upcast_trainable_to_fp32
from breeze_train.trainer import BreezeTrainer
from models.breeze import BreezeForConditionalGeneration

logger = logging.getLogger("breeze_train")


def resolve_attn_implementation(requested: str) -> str:
    """`auto` -> flash_attention_2 when flash-attn is installed on an Ampere+
    GPU, else PyTorch sdpa. Both match eager within bf16 noise on padded
    batches (loss 8.3517 eager / 8.3533 sdpa / 8.3528 flash_attention_2) and
    never materialize the L x L attention matrix."""
    if requested != "auto":
        return requested
    if (
        torch.cuda.is_available()
        and importlib.util.find_spec("flash_attn") is not None
        and torch.cuda.get_device_capability()[0] >= 8
    ):
        return "flash_attention_2"
    return "sdpa"


def parse_eval_manifests(specs: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    """`NAME=PATH` pairs -> {NAME: PATH}; NAME becomes the metric prefix `eval_<NAME>_loss`."""
    manifests: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_]+", name) or not path:
            parser.error(f"--eval-manifest expects NAME=PATH with NAME in [A-Za-z0-9_], got {spec!r}")
        if name in manifests:
            parser.error(f"--eval-manifest: duplicate name {name!r}")
        manifests[name] = Path(path)
    return manifests


def file_digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "lines": sum(1 for line in data.splitlines() if line.strip())}


def git_state(repo: Path) -> dict:
    def run(*cmd: str) -> str:
        result = subprocess.run(["git", "--no-pager", "-C", str(repo), *cmd],
                                capture_output=True, text=True, check=False)
        # rstrip only: porcelain lines start with a 2-char status that may be a space.
        return result.stdout.rstrip() if result.returncode == 0 else ""
    changed = [line[3:] for line in run("status", "--porcelain").splitlines()]
    return {"commit": run("rev-parse", "HEAD") or None, "dirty_files": changed}


def write_run_meta(output_dir: Path, meta: dict) -> Path:
    """run_meta.json on the first launch; run_meta.<UTC time>.json on later ones (e.g. --resume)."""
    path = output_dir / "run_meta.json"
    if path.exists():
        path = output_dir / f"run_meta.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Breeze TTS 2 (LoRA fine-tuning)")
    parser.add_argument("model", type=Path, help="Checkpoint directory")
    parser.add_argument("--config", type=Path, required=True, help="JSON training config")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Held-out manifest evaluated every eval_steps (config, default save_steps) and before "
        "step 1; logged as eval_<NAME>_loss plus its backbone/depth components. Repeatable. "
        "Its clips must already be in --cache-dir.",
    )
    parser.add_argument(
        "--sample-prompts",
        type=Path,
        default=None,
        help="JSONL of fixed prompts (id, text, optional ref_audio + ref_text) rendered to "
        "<output-dir>/samples/step-<N>/ before step 1 and at every checkpoint save",
    )
    parser.add_argument("--policy", default=None, help="Overrides the config's policy")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None, help="Overrides the config value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Per-process accumulation used verbatim (skips the automatic "
        "division of the config value by the number of processes)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume model, optimizer, scheduler, RNG and data position from a "
        "checkpoint directory; bare --resume picks the newest checkpoint-* in --output-dir",
    )
    parser.add_argument("--wandb-project", default=None, help="Enables Weights & Biases logging")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated tags")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    policy = args.policy or cfg.get("policy", "p1")
    eval_manifests = parse_eval_manifests(args.eval_manifest, parser)

    # Initializes torch.distributed under `accelerate launch`; a no-op
    # single-process state under plain `python train.py`.
    state = PartialState()
    num_processes = state.num_processes

    os.makedirs(args.output_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    # Every rank runs this script; only rank 0 writes train.log, otherwise
    # N processes interleave duplicate lines into the same file.
    if state.is_main_process:
        file_handler = logging.FileHandler(os.path.join(str(args.output_dir), "train.log"))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    model_config = AutoConfig.from_pretrained(str(args.model))
    model_config.depth_header_loss_weight = cfg.get("depth_header_loss_weight", 1.0)

    attn_implementation = resolve_attn_implementation(cfg.get("attn_implementation", "auto"))
    logger.info(f"attn_implementation={attn_implementation}")
    model = BreezeForConditionalGeneration.from_pretrained(
        str(args.model),
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), fix_mistral_regex=False)

    model = apply_policy(
        model,
        policy,
        lora_rank=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )
    upcast = upcast_trainable_to_fp32(model)
    logger.info(f"policy={policy} upcast_trainable_params_to_fp32={upcast}")
    trainable = trainable_summary(model)
    logger.info(f"trainable_params={trainable['total']} per_module={trainable['per_module']}")
    # Default on (fits 12 GB). Large-memory GPUs (H100 80 GB) can turn it off
    # for ~one fewer forward recompute per step; LoRA params then require
    # grad directly, so enable_input_require_grads() is not needed either.
    if cfg.get("gradient_checkpointing", True):
        enable_memory_savings(model)

    dataset = BreezeTrainDataset(args.manifest, args.cache_dir, tokenizer, model.config)
    eval_datasets = {
        name: BreezeTrainDataset(path, args.cache_dir, tokenizer, model.config)
        for name, path in eval_manifests.items()
    }
    collator = BreezeDataCollator(pad_token_id=tokenizer.pad_token_id, config=model.config)

    save_steps = args.save_steps if args.save_steps is not None else cfg.get("save_steps", 300)
    eval_steps = cfg.get("eval_steps", save_steps)
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps", -1)

    if args.gradient_accumulation_steps is not None:
        grad_accum = args.gradient_accumulation_steps
    else:
        single_gpu_accum = cfg.get("gradient_accumulation_steps", 8)
        if single_gpu_accum % num_processes != 0:
            parser.error(
                f"config gradient_accumulation_steps={single_gpu_accum} is not divisible by "
                f"num_processes={num_processes}; pass --gradient-accumulation-steps explicitly"
            )
        grad_accum = single_gpu_accum // num_processes
    per_device_batch = cfg.get("per_device_train_batch_size", 2)
    logger.info(
        f"num_processes={num_processes} per_device_train_batch_size={per_device_batch} "
        f"gradient_accumulation_steps={grad_accum} "
        f"global_batch_size={per_device_batch * grad_accum * num_processes}"
    )

    resume_from_checkpoint = args.resume
    if resume_from_checkpoint == "latest":
        resume_from_checkpoint = get_last_checkpoint(str(args.output_dir))
        if resume_from_checkpoint is None:
            parser.error(f"--resume: no checkpoint-* directory in {args.output_dir}")
    if resume_from_checkpoint is not None:
        logger.info(f"resume_from_checkpoint={resume_from_checkpoint}")

    if state.is_main_process:
        meta_path = write_run_meta(Path(args.output_dir), {
            "argv": sys.argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_processes": num_processes,
            "git": git_state(Path(__file__).resolve().parent),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__, "accelerate": accelerate.__version__},
            "init_model": str(args.model),
            "policy": policy,
            "config": cfg,
            "config_path": str(args.config),
            "manifest": file_digest(args.manifest),
            "eval_manifests": {name: file_digest(path) for name, path in eval_manifests.items()},
            "sample_prompts": file_digest(args.sample_prompts) if args.sample_prompts else None,
            "trainable_params": trainable,
            "global_batch_size": per_device_batch * grad_accum * num_processes,
            "resume_from_checkpoint": resume_from_checkpoint,
        })
        logger.info(f"run_meta={meta_path}")

    report_to = list(cfg.get("report_to", []))
    if args.wandb_project:
        # The wandb integration reads these at wandb.init() on rank 0.
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        if "wandb" not in report_to:
            report_to.append("wandb")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=False,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        # AdamW. Defaults equal the TrainingArguments defaults the earlier
        # recipes ran with (torch >= 2.8 -> adamw_torch_fused); set e.g.
        # weight_decay 0.1 / adam_beta2 0.95 per config.
        optim=cfg.get("optim", "adamw_torch_fused"),
        weight_decay=cfg.get("weight_decay", 0.0),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.999),
        adam_epsilon=cfg.get("adam_epsilon", 1e-8),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=save_steps,
        # Full fine-tune checkpoints are large (fp32 trainable weights plus
        # AdamW state); null keeps every checkpoint.
        save_total_limit=cfg.get("save_total_limit"),
        max_steps=max_steps,
        report_to=report_to,
        run_name=args.wandb_run_name,
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        # Held-out loss. PeftModel.forward(*args, **kwargs) hides the `labels`
        # parameter from Trainer's signature inspection, so without explicit
        # label_names prediction_step would return no loss at all.
        eval_strategy="steps" if eval_datasets else "no",
        eval_steps=eval_steps if eval_datasets else None,
        eval_on_start=bool(eval_datasets) and cfg.get("eval_on_start", True),
        per_device_eval_batch_size=per_device_batch,
        prediction_loss_only=True,
        label_names=["labels"],
        # tqdm's carriage-return bar interleaves with the loss records on stdout;
        # turn it off for detached runs (train.log in --output-dir is clean either way).
        disable_tqdm=cfg.get("disable_tqdm", False),
        seed=args.seed,
        # The Trainer receives a PeftModel (not a PreTrainedModel), so it
        # would default to find_unused_parameters=True, which crashes DDP
        # combined with (reentrant) gradient checkpointing and costs an extra
        # graph traversal per step. Every trainable parameter is used in each
        # forward pass, so the unused-parameter search is never needed.
        ddp_find_unused_parameters=False,
    )

    trainer = BreezeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_datasets or None,
        data_collator=collator,
    )
    if args.sample_prompts is not None:
        from breeze_train.samples import CheckpointSampler

        trainer.add_callback(CheckpointSampler(trainer, args.model, tokenizer, args.sample_prompts, seed=42))
    # With save_strategy="steps" the Trainer also writes a full checkpoint
    # (adapter + optimizer + scheduler + RNG + trainer_state) at the final
    # step, so every run ends with a checkpoint --resume can continue from.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    checkpoint_dir = os.path.join(str(args.output_dir), f"checkpoint-{trainer.state.global_step}")

    if state.is_main_process:
        peak_vram_gib = torch.cuda.max_memory_reserved() / 2**30
        peak_vram_alloc_gib = torch.cuda.max_memory_allocated() / 2**30
        logger.info(f"peak_vram_gib={peak_vram_gib}")
        logger.info(f"peak_vram_alloc_gib={peak_vram_alloc_gib}")
        print(f"peak_vram_gib={peak_vram_gib}")
        print(f"peak_vram_alloc_gib={peak_vram_alloc_gib}")
        print(f"saved {checkpoint_dir}")


if __name__ == "__main__":
    main()
