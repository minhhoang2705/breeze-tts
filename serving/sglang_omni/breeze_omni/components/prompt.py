"""Token-exact Breeze prompts; behavior ported from breeze_infer/templates.py:31-145,176-228,285-360."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

AUDIO_TAG = "<|AUDIO|>"
AUDIO_EOS = "<|audio_eos|>"
INSTRUCTION_BOS = "<ins_bos>"
INSTRUCTION_EOS = "<ins_eos>"


@dataclass(frozen=True)
class BreezePrompt:
    input_ids: list[int]
    text_mask: list[bool]
    ref_codes: torch.LongTensor | None


def _speaker_prefix(speaker: str | None) -> str:
    if speaker in (None, ""):
        return ""
    if speaker.startswith("[") and speaker.endswith("]"):
        return speaker
    return f"[{speaker}]"


def _text_segment(tokenizer: Any, text: str) -> tuple[str, int]:
    # Match _prepare_one: round-trip each text segment with tokenizer-added BOS,
    # then tokenize the concatenated prompt without adding another BOS.
    token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    rendered = tokenizer.decode(token_ids, skip_special_tokens=False)
    return rendered, len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _assemble(tokenizer: Any, segments: list[tuple[str, str]], codes: torch.LongTensor | None) -> BreezePrompt:
    rendered: list[str] = []
    per_segment: list[tuple[str, int]] = []
    for kind, value in segments:
        if kind == "text":
            decoded, length = _text_segment(tokenizer, value)
            rendered.append(decoded)
            per_segment.append((kind, length))
        else:
            rendered.append(value)
            per_segment.append((kind, len(tokenizer(value, add_special_tokens=False)["input_ids"])))
    ids = tokenizer("".join(rendered), add_special_tokens=False)["input_ids"]
    mask = [flag for kind, length in per_segment for flag in [kind == "text"] * length]
    if len(ids) != len(mask):
        raise ValueError(f"Tokenizer segment lengths ({len(mask)}) do not match prompt ids ({len(ids)})")
    return BreezePrompt(input_ids=list(ids), text_mask=mask, ref_codes=codes)


def build_prompts(
    tokenizer: Any,
    *,
    text: str,
    instruction: str | None = None,
    ref_text: str | None = None,
    ref_codes: torch.LongTensor | None = None,
    speaker: str = "S0",
    cfg_scale: float = 1.0,
) -> tuple[BreezePrompt, BreezePrompt | None, str]:
    """Build the positive prompt and optional negative branch.

    The negative prompt removes only instruction text. Reference-edit negatives
    retain transcript and audio. Plain and clone prompts have no CFG negative.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("text is required")
    if (ref_text is None) != (ref_codes is None):
        raise ValueError("ref_text and ref_codes must be provided together")
    if ref_text is not None and (not isinstance(ref_text, str) or not ref_text):
        raise ValueError("ref_text must be non-empty")
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    instruction = instruction if instruction and instruction.strip() else None
    prefix = _speaker_prefix(speaker)
    is_reference = ref_text is not None
    if is_reference:
        if ref_codes.ndim != 2 or ref_codes.shape[1] != 16:
            raise ValueError("ref_codes must have shape [frames, 16]")
        ref_segments = [("text", prefix + ref_text), ("audio", AUDIO_TAG * ref_codes.shape[0] + AUDIO_EOS)]
    else:
        ref_segments = []
    if is_reference and instruction:
        template = "ref_edit_tata"
        positive = ref_segments + [("text", f"{prefix}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}")]
        negative_segments = ref_segments + [("text", prefix + text)]
    elif is_reference:
        template = "ref_clone_tata"
        positive = ref_segments + [("text", prefix + text)]
        negative_segments = None
    elif instruction:
        template = "tts_instruction"
        positive = [("text", f"{prefix}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}")]
        negative_segments = [("text", prefix + text)]
    else:
        template = "tts_plain"
        positive = [("text", prefix + text)]
        negative_segments = None

    if cfg_scale == 1.0:
        return _assemble(tokenizer, positive, ref_codes), None, "cond"
    if negative_segments is None:
        raise ValueError(f"Template '{template}' does not define a negative prompt but cfg_scale={cfg_scale}")
    negative = _assemble(tokenizer, negative_segments, ref_codes)
    if cfg_scale == 0.0:
        return negative, None, "neg_only"
    return _assemble(tokenizer, positive, ref_codes), negative, "cfg"
