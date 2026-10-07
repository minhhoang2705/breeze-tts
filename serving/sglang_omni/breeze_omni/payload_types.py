"""Serializable request state shared by Breeze pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class BreezeTtsState(DeclarativeStateBase):
    text: str = wire("", codec="str")
    instruction: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    speaker: str = wire("S0", codec="str")
    cfg_scale: float = wire(1.0, codec="float")
    mode: str = wire("cond", codec="str")
    sampling: dict[str, Any] = wire(default_factory=dict, codec="dict")
    seed: int | None = None
    max_new_tokens: int = wire(1500, codec="int")
    max_seq_len: int = wire(2048, codec="int")
    input_ids: Any | None = wire(None, codec="tensor_list")
    text_mask: Any | None = wire(None, codec="tensor_list")
    ref_codes: Any | None = wire(None, codec="tensor_list")
    negative_input_ids: Any | None = wire(None, codec="tensor_list")
    negative_text_mask: Any | None = wire(None, codec="tensor_list")
    negative_ref_codes: Any | None = wire(None, codec="tensor_list")
    generated_codes: Any | None = wire(None, codec="tensor_list")
    sample_rate: int = wire(24000, codec="int")
    audio_samples: Any | None = wire(None, codec="tensor_list")
    finish_reason: str | None = None
