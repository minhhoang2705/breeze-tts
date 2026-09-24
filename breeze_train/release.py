"""Distribution files for derivative weights.

Section 4 of the BreezeBlue Research and Non-Commercial License requires every
distributed Derivative Model (a LoRA adapter or a merged checkpoint) to ship
with a complete copy of the license, a NOTICE file carrying a fixed notice, a
prominent "Derived from Breeze TTS 2 ..." statement in its model card, and a
description of the modifications. It also forbids "BreezeBlue" or
"Breeze TTS 2" as the primary name of a derivative.

`write_release_files` writes LICENSE, NOTICE and a model card README.md into a
directory so it can be published as-is. `python -m breeze_train.release` does
this for an unmerged adapter directory; `breeze_train.export` calls it for
every merged checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

NOTICE_TEXT = (
    "Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial\n"
    "License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved.\n"
)
DERIVED_STATEMENT = (
    "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and "
    "non-commercial use only."
)
BASE_MODEL_ID = "BreezeBlue/Breeze-TTS-2"


def validate_release_name(name: str) -> None:
    """Reject names that use BreezeBlue marks as the primary name (license §4)."""
    compact = "".join(ch for ch in name.lower() if ch.isalnum())
    if "breeze" in compact:
        raise ValueError(
            f"Release name {name!r} uses 'Breeze'/'BreezeBlue'; the license forbids "
            "BreezeBlue marks as the primary name of a derivative model."
        )


def _lora_summary(adapter_dir: Path) -> list[str]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text())
    lines = []
    if config.get("r") is not None:
        lines.append(f"- LoRA rank: {config['r']}, alpha: {config.get('lora_alpha')}")
    targets = config.get("target_modules")
    if targets:
        if isinstance(targets, str):
            lines.append(f"- LoRA target modules (regex): `{targets}`")
        else:
            lines.append(f"- LoRA target modules: {', '.join(f'`{t}`' for t in sorted(targets))}")
    return lines


def _model_card(
    name: str,
    merged: bool,
    adapter_dir: Path,
    dataset: str | None,
    dataset_license: str | None,
) -> str:
    front_matter = [
        "---",
        "license: other",
        "license_name: breezeblue-research-non-commercial",
        "license_link: LICENSE",
        f"base_model: {BASE_MODEL_ID}",
        "pipeline_tag: text-to-speech",
        "tags:",
        "- text-to-speech",
        "- lora",
        "- non-commercial",
    ]
    if dataset:
        front_matter += ["datasets:", f"- {dataset}"]
    front_matter.append("---")

    kind = (
        "LoRA adapter merged into the base weights; loads with the unmodified "
        "Breeze TTS 2 `infer.py`."
        if merged
        else f"Unmerged LoRA adapter; load it on top of `{BASE_MODEL_ID}` with PEFT."
    )
    if dataset:
        data_line = f"- Fine-tuning data: `{dataset}`"
        if dataset_license:
            data_line += f" ({dataset_license}); its terms apply in addition to this license"
    else:
        data_line = "- Fine-tuning data: not recorded"

    body = [
        f"# {name}",
        "",
        f"> **{DERIVED_STATEMENT}**",
        "",
        "## Modifications",
        "",
        f"- Base model: [{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID})",
        f"- {kind}",
        *_lora_summary(adapter_dir),
        data_line,
        "",
        "## License",
        "",
        "These weights are a Derivative Model of Breeze TTS 2 and are governed by the",
        "BreezeBlue Research and Non-Commercial License Agreement (see `LICENSE` and",
        "`NOTICE`). No commercial use is permitted without a separate written license",
        "from BreezeBlue. Do not use these weights to clone or imitate a real person's",
        "voice without that person's explicit consent.",
        "",
        "This model is not affiliated with or endorsed by BreezeBlue.",
        "",
    ]
    return "\n".join(front_matter + [""] + body)


def write_release_files(
    target_dir: str | Path,
    base_dir: str | Path,
    *,
    name: str,
    merged: bool,
    adapter_dir: str | Path,
    dataset: str | None = None,
    dataset_license: str | None = None,
) -> None:
    """Write LICENSE, NOTICE and README.md (model card) into `target_dir`.

    README.md is overwritten: PEFT's `save_pretrained` leaves a generic stub
    there that carries none of the required statements.
    """
    validate_release_name(name)
    target_dir = Path(target_dir)
    base_dir = Path(base_dir)

    license_src = base_dir / "LICENSE"
    if not license_src.exists():
        raise FileNotFoundError(f"Base checkpoint has no LICENSE: {license_src}")
    if license_src.resolve() != (target_dir / "LICENSE").resolve():
        shutil.copy2(license_src, target_dir / "LICENSE")

    # §4(a): carry any NOTICE shipped with the base checkpoint, then the
    # notice §4(c) requires.
    notice_parts = []
    base_notice = base_dir / "NOTICE"
    if base_notice.exists():
        notice_parts.append(base_notice.read_text().rstrip() + "\n")
    if NOTICE_TEXT not in "".join(notice_parts):
        notice_parts.append(NOTICE_TEXT)
    (target_dir / "NOTICE").write_text("\n".join(notice_parts))

    (target_dir / "README.md").write_text(
        _model_card(name, merged, Path(adapter_dir), dataset, dataset_license)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add LICENSE, NOTICE and a model card to a LoRA adapter directory before publishing"
    )
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("--name", required=True, help="Release name; must not use 'Breeze'/'BreezeBlue'")
    parser.add_argument("--dataset", help="Fine-tuning dataset id, e.g. capleaf/viVoice")
    parser.add_argument("--dataset-license", help="License of the fine-tuning dataset")
    args = parser.parse_args()

    write_release_files(
        args.adapter_dir,
        args.base_dir,
        name=args.name,
        merged=False,
        adapter_dir=args.adapter_dir,
        dataset=args.dataset,
        dataset_license=args.dataset_license,
    )
    print(f"release files written to {args.adapter_dir}")


if __name__ == "__main__":
    main()
