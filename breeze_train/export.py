"""Merged checkpoint export.

Turns an adapter checkpoint into a directory `infer.py` can load unmodified:
merges the LoRA adapters into the base weights (L13 -- PEFT `save_pretrained`
alone writes only `adapter_model.safetensors` / `adapter_config.json`, and
`breeze_infer/runtime.py` does a plain `from_pretrained(ckpt_dir)`), and
copies the companion files `save_pretrained` does not write (L14), including
`LICENSE` -- the BreezeBlue Research and Non-Commercial License governs
model weights, checkpoints, adapters, and derivative models separately from
the Apache-2.0 source (README.md:184), and every artifact this exporter
produces is such a derivative.

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


def merge_adapter(base_dir: str | Path, adapter_dir: str | Path):
    # Load on CPU: the merge is weight arithmetic and does not need the GPU;
    # keeping it off the 12 GB card avoids competing with anything resident
    # there.
    model = BreezeForConditionalGeneration.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    peft_model = peft.PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    return merged


def export_checkpoint(
    base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path
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

    merged = merge_adapter(base_dir, adapter_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    del merged
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
        description="Merge a LoRA adapter checkpoint into a loadable Breeze TTS 2 directory"
    )
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--policy", default="p1")
    args = parser.parse_args()

    export_checkpoint(args.base_dir, args.adapter_dir, args.out_dir)

    adapter_config_path = args.adapter_dir / "adapter_config.json"
    lora_rank = None
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text())
        lora_rank = adapter_config.get("r")

    export_info = {
        "policy": args.policy,
        "adapter_dir": str(args.adapter_dir),
        "base_dir": str(args.base_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lora_rank": lora_rank,
        "weights_license": "BreezeBlue Research and Non-Commercial License",
    }
    (args.out_dir / "export_info.json").write_text(json.dumps(export_info, indent=2))

    print(f"exported {args.out_dir}")


if __name__ == "__main__":
    main()
