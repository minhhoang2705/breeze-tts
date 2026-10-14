"""Preprocessing, SGLang engine, and Qwen codec stage factories."""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import logging
import os
import threading
from collections import OrderedDict
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


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int = 4
) -> ThreadedSimpleScheduler:
    """Load the prompt tokenizer and the reference-audio codec for this stage.

    The stage runs in its own process (config.py), so reference encoding does
    not compete for the GIL with the engine's decode loop. The codec encodes in
    fp32 on the GPU like Breeze's infer.py; BREEZE_REF_ENCODE_DEVICE=cpu
    encodes on CPU instead.
    """
    from transformers import AutoTokenizer

    from .engine_builder import load_breeze_codec

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, tokenizer_type="gemma", fix_mistral_regex=False, trust_remote_code=True
    )
    device = "cpu" if os.environ.get("BREEZE_REF_ENCODE_DEVICE", "cuda") == "cpu" else "cuda"
    audio_tokenizer = load_breeze_codec(model_path, device=device, dtype="float32")
    logger.info("Breeze preprocessing: reference codec on %s, %d workers", device, max_concurrency)
    return ThreadedSimpleScheduler(
        functools.partial(
            preprocess_breeze_payload, tokenizer=tokenizer, audio_tokenizer=audio_tokenizer
        ),
        max_concurrency=max_concurrency,
    )


def preprocess_breeze_payload(
    payload: StagePayload, *, tokenizer: Any, audio_tokenizer: Any
) -> StagePayload:
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
        ref_codes = _encode_reference(audio_tokenizer, ref_audio)
    elif ref_codes is not None:
        ref_codes = torch.as_tensor(ref_codes, dtype=torch.long)
    state = build_breeze_state(payload, tokenizer, ref_codes=ref_codes)
    state.ref_audio = None
    state_data = state.to_dict()
    return StagePayload(request_id=payload.request_id, request=payload.request, data=state_data)


# Clone requests may reuse a reference voice. Encoding a reference with the 12 Hz
# codec costs ~10 ms on the GPU (~0.7 s on CPU, BREEZE_REF_ENCODE_DEVICE=cpu) for a
# 7 s clip, so cache the codes by the hash of the audio bytes: identical bytes give
# identical codes, so outputs do not change.
_REF_CODE_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_REF_CODE_CACHE_SIZE = 128
_REF_CODE_LOCK = threading.Lock()


def _encode_reference(audio_tokenizer: Any, ref_audio: Any) -> torch.Tensor:
    audio_bytes = _audio_bytes(ref_audio)
    key = hashlib.sha256(audio_bytes).hexdigest()
    with _REF_CODE_LOCK:
        cached = _REF_CODE_CACHE.get(key)
        if cached is not None:
            _REF_CODE_CACHE.move_to_end(key)
            return cached.clone()
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
    with torch.inference_mode():
        encoded = audio_tokenizer.encode(mono, sr=int(sample_rate))
    codes = encoded.audio_codes[0].to(dtype=torch.long, device="cpu").contiguous()
    with _REF_CODE_LOCK:
        _REF_CODE_CACHE[key] = codes
        _REF_CODE_CACHE.move_to_end(key)
        while len(_REF_CODE_CACHE) > _REF_CODE_CACHE_SIZE:
            _REF_CODE_CACHE.popitem(last=False)
    return codes.clone()


def _audio_bytes(value: Any) -> bytes:
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
        return bytes(value)
    if isinstance(value, str) and value.startswith("data:"):
        return base64.b64decode(value.partition(",")[2])
    if isinstance(value, str) and data_uri:
        return base64.b64decode(value)
    if isinstance(value, str):
        return Path(value).read_bytes()
    raise ValueError("ref_audio must be a path, data URI, or encoded audio bytes")


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


