"""Inference-ready checkpoint export.

Turns a training checkpoint into a directory `infer.py` can load unmodified.
Two checkpoint kinds, told apart by `adapter_config.json`:

- LoRA (p1): merges the adapters into the base weights (L13 -- PEFT
  `save_pretrained` alone writes only `adapter_model.safetensors` /
  `adapter_config.json`, and `breeze_infer/runtime.py` does a plain
  `from_pretrained(ckpt_dir)`).
- Full fine-tune (p0/p1b/p2): the checkpoint already holds every weight, with
  the trained ones as fp32 master copies; it is reloaded and saved in bf16
  like the base checkpoint, dropping optimizer/scheduler state.

Either way it copies the companion files `save_pretrained` does not write
(L14), including `LICENSE` -- the BreezeBlue Research and Non-Commercial
License governs model weights, checkpoints, adapters, and derivative models
separately from the Apache-2.0 source (README.md:184), and every artifact
this exporter produces is such a derivative. `NOTICE` and a model card are
written by `breeze_train.release` so the directory can be published as-is.

Never edits `breeze_infer/runtime.py` or `infer.py` to work around a missing
file -- the fix always belongs here.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import peft
import torch

from breeze_train.release import validate_release_name, write_release_files
from models.breeze import BreezeForConditionalGeneration

COMPANION_FILES = (
    "audio_tokenizer/config.json",
    "audio_tokenizer/configuration.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "LICENSE",
)


def is_lora_checkpoint(checkpoint_dir: str | Path) -> bool:
    return (Path(checkpoint_dir) / "adapter_config.json").exists()


def load_finetuned(base_dir: str | Path, checkpoint_dir: str | Path):
    # Load on CPU: merging/casting is weight arithmetic and does not need the
    # GPU; keeping it off the 12 GB card avoids competing with anything
    # resident there.
    if not is_lora_checkpoint(checkpoint_dir):
        return BreezeForConditionalGeneration.from_pretrained(
            str(checkpoint_dir),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=None,
        )
    model = BreezeForConditionalGeneration.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    peft_model = peft.PeftModel.from_pretrained(model, str(checkpoint_dir))
    return peft_model.merge_and_unload()


def export_checkpoint(
    base_dir: str | Path, checkpoint_dir: str | Path, out_dir: str | Path
) -> None:
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in COMPANION_FILES:
        src = base_dir / rel_path
        if not src.exists():
            raise FileNotFoundError(
                f"Companion file missing from base checkpoint: {src}"
            )

    model = load_finetuned(base_dir, checkpoint_dir)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    del model
    gc.collect()

    for rel_path in COMPANION_FILES:
        src = base_dir / rel_path
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    missing_after_copy = [p for p in COMPANION_FILES if not (out_dir / p).exists()]
    if missing_after_copy:
        raise RuntimeError(
            f"Companion files still missing after copy: {missing_after_copy}"
        )

    reloaded, loading_info = BreezeForConditionalGeneration.from_pretrained(
        str(out_dir),
        output_loading_info=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    del reloaded
    if loading_info["missing_keys"] or loading_info["unexpected_keys"]:
        raise RuntimeError(
            "Exported checkpoint failed to reload cleanly: "
            f"missing_keys={loading_info['missing_keys']} "
            f"unexpected_keys={loading_info['unexpected_keys']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a LoRA or full fine-tune checkpoint as a loadable Breeze TTS 2 directory"
    )
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("checkpoint_dir", type=Path, help="Trainer checkpoint-* directory")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--policy", default="p1")
    parser.add_argument(
        "--name", help="Model card name (default: out_dir name); must not use 'Breeze'/'BreezeBlue'"
    )
    parser.add_argument("--dataset", help="Fine-tuning dataset id, e.g. capleaf/viVoice")
    parser.add_argument("--dataset-license", help="License of the fine-tuning dataset")
    args = parser.parse_args()
    name = args.name or args.out_dir.name
    # Fail before the multi-minute merge, not after it.
    validate_release_name(name)

    export_checkpoint(args.base_dir, args.checkpoint_dir, args.out_dir)
    write_release_files(
        args.out_dir,
        args.base_dir,
        name=name,
        merged=True,
        adapter_dir=args.checkpoint_dir,
        dataset=args.dataset,
        dataset_license=args.dataset_license,
    )

    lora_rank = None
    if is_lora_checkpoint(args.checkpoint_dir):
        adapter_config = json.loads((args.checkpoint_dir / "adapter_config.json").read_text())
        lora_rank = adapter_config.get("r")

    export_info = {
        "policy": args.policy,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_kind": "lora" if is_lora_checkpoint(args.checkpoint_dir) else "full",
        "base_dir": str(args.base_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lora_rank": lora_rank,
        "weights_license": "BreezeBlue Research and Non-Commercial License",
    }
    (args.out_dir / "export_info.json").write_text(json.dumps(export_info, indent=2))

    print(f"exported {args.out_dir}")


if __name__ == "__main__":
    main()
