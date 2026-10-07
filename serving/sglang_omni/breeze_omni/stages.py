"""Preprocessing, SGLang engine, and Qwen codec stage factories."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.device import resolve_device_spec

from .request_builders import build_breeze_state

logger = logging.getLogger(__name__)
_PREPROCESS_CONTEXT: tuple[Any, Any] | None = None


def set_preprocessing_context(tokenizer: Any, audio_tokenizer: Any) -> None:
    global _PREPROCESS_CONTEXT
    _PREPROCESS_CONTEXT = (tokenizer, audio_tokenizer)


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int = 4
) -> ThreadedSimpleScheduler:
    del model_path
    return ThreadedSimpleScheduler(
        preprocess_breeze_payload,
        max_concurrency=max_concurrency,
    )


def preprocess_breeze_payload(payload: StagePayload) -> StagePayload:
    context = _PREPROCESS_CONTEXT
    if context is None:
        raise RuntimeError("Breeze preprocessing context was not initialized by the engine")
    tokenizer, audio_tokenizer = context
    raw_inputs = payload.request.inputs or {}
    if isinstance(raw_inputs, str):
        inputs = {"text": raw_inputs}
    elif isinstance(raw_inputs, dict):
        inputs = raw_inputs
    else:
        raise ValueError("speech prompt must be a string or an object")
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts = metadata.get("tts_params") or {}
    stage = params.get("stage_params") or {}
    engine = stage.get("tts_engine") or {}
    if not isinstance(tts, dict) or not isinstance(engine, dict):
        raise ValueError("tts_params and stage_params.tts_engine must be objects")
    references = inputs.get("references") or params.get("references") or tts.get("references") or []
    ref_audio = (
        inputs.get("ref_audio")
        or inputs.get("reference_audio")
        or tts.get("ref_audio")
        or engine.get("ref_audio")
    )
    ref_text = inputs.get("ref_text") or tts.get("ref_text") or engine.get("ref_text")
    if not ref_audio and references:
        ref = references[0]
        if isinstance(ref, dict):
            ref_audio = ref.get("audio_path") or ref.get("ref_audio") or ref.get("audio") or ref.get("data")
            ref_text = ref_text or ref.get("text")
        else:
            ref_audio = ref
    if ref_text is not None:
        inputs = dict(inputs)
        inputs["ref_text"] = ref_text
        payload.request.inputs = inputs
    ref_codes = inputs.get("ref_codes")
    if ref_codes is None and ref_audio is not None:
        waveform, sample_rate = _read_audio(ref_audio)
        with torch.inference_mode():
            encoded = audio_tokenizer.encode(waveform.numpy(), sr=sample_rate)
        ref_codes = encoded.audio_codes[0].to(dtype=torch.long, device="cpu").contiguous()
    elif ref_codes is not None:
        ref_codes = torch.as_tensor(ref_codes, dtype=torch.long)
    state = build_breeze_state(payload, tokenizer, ref_codes=ref_codes)
    state.ref_audio = None
    state_data = state.to_dict()
    return StagePayload(request_id=payload.request_id, request=payload.request, data=state_data)


def _read_audio(value: Any) -> tuple[torch.Tensor, int]:
    data_uri = False
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            value = data
        elif value.get("audio_path") or value.get("path"):
            value = value.get("audio_path") or value.get("path")
        elif value.get("base64") or value.get("data"):
            value = value.get("base64") or value.get("data")
            data_uri = True
        else:
            value = value.get("audio") or value.get("ref_audio")
    if isinstance(value, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(value)
    elif isinstance(value, str) and value.startswith("data:"):
        audio_bytes = base64.b64decode(value.partition(",")[2])
    elif isinstance(value, str) and data_uri:
        audio_bytes = base64.b64decode(value)
    elif isinstance(value, str):
        waveform, rate = sf.read(Path(value), dtype="float32", always_2d=True)
        mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        return torch.from_numpy(mono), int(rate)
    else:
        raise ValueError("ref_audio must be a path, data URI, or encoded audio bytes")
    waveform, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
    return torch.from_numpy(mono), int(rate)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from .engine_builder import BreezeTtsEngineBuilder

    return BreezeTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    **kwargs: Any,
):
    del kwargs
    device = resolve_device_spec(device, gpu_id)
    from .engine_builder import load_breeze_codec
    from .vocoder import BreezeStreamingVocoder

    tokenizer = load_breeze_codec(model_path, device=device, dtype=dtype)
    return BreezeStreamingVocoder(tokenizer, device=device)


