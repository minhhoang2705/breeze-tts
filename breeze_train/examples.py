"""Training example builder.

Builds the single training example the model expects: the inference prompt
prefix (reused byte-for-byte from `breeze_infer.templates`), the target audio
appended as `<|AUDIO|>` placeholders plus `<|audio_eos|>`, and a 1-D `labels`
row using the model's four label classes documented at
`models/breeze.py:1741-1748`.

| Position                          | label      | Trains                    |
|------------------------------------|-----------|----------------------------|
| text token                         | -100      | nothing                    |
| reference `<\\|AUDIO\\|>`            | -101      | backbone only (cb0)        |
| target `<\\|AUDIO\\|>`               | audio_token_id | backbone + depth decoder |
| `<\\|audio_eos\\|>` (ref or target) | -100      | backbone (stop token)      |
"""

from __future__ import annotations

from typing import Any

import torch

from breeze_infer.templates import (
    AUDIO_EOS,
    AUDIO_TAG,
    _prepare_one,
    get_template,
    select_template_name,
)
from breeze_train import manifest as manifest_module


class _CachedCodesTokenizer:
    """Audio-tokenizer shim: serves cached codes instead of re-encoding audio.

    `_prepare_one` resolves each audio segment through
    `_resolve_segment_audio_codes` -> `encode_prompt_audio`, which calls
    `audio_tokenizer.encode(wav, sr=...)`. Left alone with the real
    `Qwen3TTSTokenizer` this re-encodes the reference wav from disk on every
    call, defeating the phase-2 cache. This shim ignores its arguments and
    returns the next queued cached-code array, in segment order.
    """

    def __init__(self, codes_queue: list[torch.Tensor]) -> None:
        self._queue = list(codes_queue)

    def encode(self, wav: Any, sr: Any) -> dict:
        if not self._queue:
            raise RuntimeError(
                "_CachedCodesTokenizer: no cached codes left to serve an audio "
                "segment. The template requested more audio segments than "
                "codes were supplied for."
            )
        codes = self._queue.pop(0)
        return {"audio_codes": [codes.numpy()]}


def _assert_example_invariants(
    example: dict, model_config: Any, ref_frames: int, target_frames: int
) -> None:
    """Refuse to emit a malformed example rather than letting the model assert later."""
    text_ids_mask_sum = int(example["text_ids_mask"].sum())
    text_ids_len_sum = int(example["text_ids_len"].sum())
    if text_ids_mask_sum != text_ids_len_sum:
        raise ValueError(
            f"text_ids_mask.sum() ({text_ids_mask_sum}) != text_ids_len.sum() "
            f"({text_ids_len_sum}) -- this is the exact condition the model "
            "asserts at models/breeze.py:1371"
        )

    audio_token_id = model_config.audio_token_id
    audio_count = int((example["input_ids"] == audio_token_id).sum())
    expected_audio = ref_frames + target_frames
    if audio_count != expected_audio:
        raise ValueError(
            f"audio placeholder count in input_ids ({audio_count}) != "
            f"ref_frames + target_frames ({expected_audio})"
        )

    if int(example["input_values"].shape[0]) != expected_audio:
        raise ValueError(
            f"input_values.shape[0] ({int(example['input_values'].shape[0])}) "
            f"!= ref_frames + target_frames ({expected_audio})"
        )

    if int(example["input_values"].shape[1]) != model_config.num_codebooks:
        raise ValueError(
            f"input_values.shape[1] ({int(example['input_values'].shape[1])}) "
            f"!= model_config.num_codebooks ({model_config.num_codebooks}); "
            "a value of 32 means the wrong codec was used (L10)"
        )

    audio_eos_token_id = model_config.audio_eos_token_id
    last_token = int(example["input_ids"][-1])
    if last_token != audio_eos_token_id:
        raise ValueError(
            f"last input_ids token ({last_token}) is not audio_eos_token_id "
            f"({audio_eos_token_id})"
        )


def build_example(
    tokenizer: Any,
    model_config: Any,
    record: manifest_module.ManifestRecord,
    target_codes: torch.Tensor,
    ref_codes: torch.Tensor | None,
) -> dict:
    """Build one training example.

    There is deliberately no `audio_tokenizer` parameter: this function
    constructs the `_CachedCodesTokenizer` shim itself from `ref_codes`, so
    the real codec can never be accidentally invoked here.
    """
    request = manifest_module.to_request(record)
    template_name = select_template_name(request)
    template = get_template(template_name)
    segments = template.build_segments(request)

    ref_queue = [ref_codes] if ref_codes is not None else []
    shim = _CachedCodesTokenizer(ref_queue)

    # Reuse the repo's own prompt builder; do not re-render the prompt.
    prepared = _prepare_one(tokenizer, shim, model_config, segments)

    prompt_len = int(prepared["input_ids"].shape[1])
    prompt_frames = int(prepared["audio_tokens"].shape[1])

    target_text = (AUDIO_TAG * int(target_codes.shape[0])) + AUDIO_EOS
    target_encoded = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")
    target_ids = target_encoded["input_ids"]
    target_len = int(target_ids.shape[1])

    input_ids = torch.cat([prepared["input_ids"], target_ids], dim=1).squeeze(0).to(
        torch.long
    )
    text_ids_mask = (
        torch.cat(
            [prepared["text_ids_mask"], torch.zeros((1, target_len), dtype=torch.bool)],
            dim=1,
        )
        .squeeze(0)
    )
    text_ids_len = prepared["text_ids_len"].to(torch.long)

    audio_token_id = model_config.audio_token_id
    labels = torch.full_like(input_ids, -100)
    audio_positions = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    is_reference = audio_positions < prompt_len
    labels[audio_positions[is_reference]] = -101
    labels[audio_positions[~is_reference]] = audio_token_id

    prompt_audio_tokens = prepared["audio_tokens"].squeeze(0)
    input_values = torch.cat([prompt_audio_tokens, target_codes], dim=0).to(torch.long)

    example = {
        "input_ids": input_ids,
        "labels": labels,
        "text_ids_mask": text_ids_mask,
        "text_ids_len": text_ids_len,
        "input_values": input_values,
        "prompt_len": prompt_len,
        "prompt_frames": prompt_frames,
    }

    ref_frames = prompt_frames
    target_frames = int(target_codes.shape[0])
    _assert_example_invariants(example, model_config, ref_frames, target_frames)
    return example
