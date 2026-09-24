"""Freeze/LoRA policy application.

Neutralizes L11 (LoRA target-module collision across backbone_model,
depth_decoder, text_encoder, and codec_model), L12 (gradients never reach
adapters without `enable_input_require_grads()`), L15 (`use_reentrant=False`
is a silent no-op in this repo's `gradient_checkpointing_enable` override),
and L7 (`train()` deliberately re-freezes `codec_model` and, when not
trainable, `text_encoder`).
"""

from __future__ import annotations

from typing import Any

import peft

# Module-anchored: only linears under backbone_model.* or depth_decoder.*,
# never text_encoder.* or codec_model.* (both of which also contain
# q_proj/v_proj/etc.). Passed as a *string* -- PEFT treats a string
# target_modules as a full-name regex, which is what anchors it; a list of
# bare suffixes would match by substring across all four module trees (L11).
LORA_TARGET_REGEX = (
    r"^(backbone_model|depth_decoder)\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def _freeze_all(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def apply_policy(
    model: Any,
    name: str,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
) -> Any:
    _freeze_all(model)

    if name == "p0":
        model.depth_decoder.requires_grad_(True)
        # depth_decoder.model.embed_tokens is tied to
        # backbone_model.embed_tokens.embed_audio_tokens (models/breeze.py:913-916,
        # 1102-1107); this also makes that shared parameter trainable, reported
        # under backbone_model.* by named_parameters() (see trainable_summary).
        # It is not extra parameters -- the 434,280,448 depth_decoder figure
        # already includes it -- and it must not be frozen back: the parameter
        # is shared, so that would also freeze the depth decoder's own input
        # embedding.
        return model

    if name == "p1":
        cfg = peft.LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=LORA_TARGET_REGEX,
            task_type=None,
        )
        return peft.get_peft_model(model, cfg)

    if name == "p1b":
        model = apply_policy(model, "p1", lora_rank, lora_alpha, lora_dropout)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.depth_decoder.requires_grad_(True)
        return model

    if name == "p2":
        model.backbone_model.requires_grad_(True)
        model.depth_decoder.requires_grad_(True)
        model.lm_head.requires_grad_(True)
        model.text_encoder_proj.requires_grad_(True)
        # Deliberately NOT embed_text_tokens: it is used only in the
        # `else:` branch at models/breeze.py:1578, reached when
        # self.text_encoder is None. This checkpoint ships a text encoder, so
        # every forward takes convert_input_ids_to_embeds at :1569 instead,
        # and embed_text_tokens can never receive a gradient on this
        # checkpoint.
        return model

    if name == "p3":
        raise NotImplementedError(
            "p3 is not shipped in v1. It requires two prerequisites that do "
            "not exist yet: (1) a --text-encoder-trainable flag on train.py "
            "that sets config.text_encoder_config.requires_grad = True "
            "*before* from_pretrained, because models/breeze.py:1061-1062 "
            "latches text_encoder_trainable once in __init__ and never "
            "re-reads it; and (2) a cloud config that can actually be run. "
            "Ship both together when a cloud run is scheduled."
        )

    raise ValueError(f"Unknown policy: {name!r}")


def trainable_summary(model: Any) -> dict[str, Any]:
    """Two different correct answers to "how many trainable parameters":

    - `per_module`: built from `named_parameters(remove_duplicate=False)`, so
      the tied audio embedding is reported under *both*
      `backbone_model.*` and `depth_decoder.*`. Summing this dict for p2
      gives 1,917,459,456.
    - `total`: `sum(p.numel() for p in model.parameters() if p.requires_grad)`,
      which deduplicates. For p2 this gives 1,850,252,288, matching the
      safetensors header sum.
    """
    per_module: dict[str, int] = {}
    peft_prefix = "base_model.model."
    for name, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        unwrapped_name = name[len(peft_prefix):] if name.startswith(peft_prefix) else name
        top_module = unwrapped_name.split(".", 1)[0]
        per_module[top_module] = per_module.get(top_module, 0) + param.numel()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {"per_module": per_module, "total": total}


def enable_memory_savings(model: Any) -> None:
    """Enable gradient checkpointing and make gradients reach frozen-base adapters.

    L15: `gradient_checkpointing_kwargs={"use_reentrant": False}` is a no-op
    in this repo -- `models/breeze.py:1187` pops the dict and `:1203` never
    forwards it back to the base class. Call with no arguments.

    L12: with a fully frozen base, `enable_input_require_grads()` is the only
    mechanism that lets gradients reach LoRA adapters through gradient
    checkpointing. It is mandatory, not conditional.
    """
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
