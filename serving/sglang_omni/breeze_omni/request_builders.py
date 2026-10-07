"""OpenAI speech request normalization and SGLang request adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .payload_types import BreezeTtsState


_DEFAULTS: dict[str, Any] = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "depth_temperature": 0.9,
    "depth_top_p": 1.0,
    "depth_top_k": 50,
    "repetition_penalty": 1.1,
}


def request_options(payload: StagePayload) -> tuple[dict[str, Any], dict[str, Any]]:
    request = payload.request
    raw_inputs = request.inputs or {}
    if isinstance(raw_inputs, str):
        inputs = {"text": raw_inputs}
    elif isinstance(raw_inputs, dict):
        inputs = raw_inputs
    else:
        raise ValueError("speech prompt must be a string or an object")
    params = request.params or {}
    metadata = request.metadata or {}
    tts = metadata.get("tts_params") or {}
    stage = params.get("stage_params") or {}
    engine = stage.get("tts_engine") or {}
    if not isinstance(tts, dict) or not isinstance(engine, dict):
        raise ValueError("tts_params and stage_params.tts_engine must be objects")
    merged = {**params, **tts, **engine}
    merged["__engine_keys"] = tuple(engine)
    merged["__has_tts_metadata"] = "tts_params" in metadata
    return inputs, merged


def build_breeze_state(
    payload: StagePayload,
    tokenizer: Any,
    *,
    ref_codes: torch.Tensor | None = None,
) -> BreezeTtsState:
    """Build exact positive/negative prompts from OpenAI-compatible inputs.

    Parameters may be supplied directly, in metadata.tts_params, or in the
    tts_engine stage_params object; the latter two are merged over endpoint
    defaults, matching the Qwen3-TTS and Higgs request adapters.
    """
    from .components.prompt import build_prompts

    inputs, params = request_options(payload)
    text = _text_value(inputs.get("text", params.get("text")))
    instruction = _optional_text(
        inputs.get("instruction", params.get("instruction", params.get("instructions")))
    )
    ref_text = _optional_text(inputs.get("ref_text", params.get("ref_text")))
    speaker_value = inputs.get("speaker", params.get("speaker", params.get("voice", "S0")))
    speaker = str(speaker_value)
    if speaker not in {f"S{i}" for i in range(10)}:
        speaker = "S0"
    cfg_scale = float(params.get("cfg_scale", inputs.get("cfg_scale", 1.0)))
    main, negative, mode = build_prompts(
        tokenizer,
        text=text,
        instruction=instruction,
        ref_text=ref_text,
        ref_codes=ref_codes,
        speaker=speaker,
        cfg_scale=cfg_scale,
    )
    explicit = set(params.get("explicit_generation_params") or ())
    engine_keys = set(params.get("__engine_keys") or ())
    has_tts_metadata = bool(params.get("__has_tts_metadata"))

    def generation_value(name: str, default: Any, *aliases: str) -> Any:
        names = (name, *aliases)
        for candidate in names:
            if candidate in engine_keys:
                return params[candidate]
        if has_tts_metadata and not any(candidate in explicit for candidate in names):
            return default
        for candidate in names:
            if candidate in params:
                return params[candidate]
        return default

    sampling = {
        "temperature": float(generation_value("temperature", _DEFAULTS["temperature"])),
        "top_p": float(generation_value("top_p", _DEFAULTS["top_p"])),
        "top_k": int(generation_value("top_k", _DEFAULTS["top_k"])),
        "depth_temperature": float(generation_value(
            "depth_temperature", _DEFAULTS["depth_temperature"], "subtalker_temperature"
        )),
        "depth_top_p": float(generation_value(
            "depth_top_p", _DEFAULTS["depth_top_p"], "subtalker_top_p"
        )),
        "depth_top_k": int(generation_value(
            "depth_top_k", _DEFAULTS["depth_top_k"], "subtalker_top_k"
        )),
        "repetition_penalty": float(generation_value(
            "repetition_penalty", _DEFAULTS["repetition_penalty"]
        )),
    }
    seed = params.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or int(seed) != seed:
            raise ValueError("seed must be an integer")
        seed = int(seed)
    max_new_tokens = int(generation_value("max_new_tokens", 1500))
    max_seq_len = int(params.get("max_seq_len", 2048))
    return BreezeTtsState(
        text=text,
        instruction=instruction,
        ref_text=ref_text,
        ref_audio=inputs.get("ref_audio", inputs.get("reference_audio")),
        speaker=speaker,
        cfg_scale=cfg_scale,
        mode=mode,
        sampling=sampling,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        input_ids=torch.tensor(main.input_ids, dtype=torch.long),
        text_mask=torch.tensor(main.text_mask, dtype=torch.bool),
        ref_codes=main.ref_codes,
        negative_input_ids=(torch.tensor(negative.input_ids, dtype=torch.long) if negative else None),
        negative_text_mask=(torch.tensor(negative.text_mask, dtype=torch.bool) if negative else None),
        negative_ref_codes=(negative.ref_codes if negative else None),
    )


def _text_value(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("speech request requires non-empty text")


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("text and instruction fields must be strings")
    return value


@dataclass
class BreezeSGLangRequestData(SGLangARRequestData):
    state: BreezeTtsState | None = None
    cfg_uncond: "BreezeSGLangRequestData | None" = None
    is_cfg_uncond: bool = False
    cfg_cond_rid: str | None = None
    cfg_scale: float = 1.0
    rng_seed: int | None = None
    generated_frames: list[torch.Tensor] | None = None
    # Frames already shipped to the vocoder (engine_builder.CodeChunkStream).
    stream_frames_sent: int = 0


def cfg_uncond_rid(request_id: str) -> str:
    return f"{request_id}_uncond"


def is_cfg_uncond_rid(request_id: str) -> bool:
    return request_id.endswith("_uncond")


def build_sglang_breeze_request(
    payload: StagePayload, *, model: Any, state: BreezeTtsState
) -> BreezeSGLangRequestData:
    """Create the conditional request and, when needed, its negative twin."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from .components.prompt import BreezePrompt

    if is_cfg_uncond_rid(payload.request_id):
        raise ValueError("request ids ending in '_uncond' are reserved for CFG lanes")

    # Both CFG lanes share one token stream, so they must share one length limit
    # and finish on the same step. The limit comes from the (longer) cond prompt.
    max_new_tokens = min(state.max_new_tokens, state.max_seq_len - len(state.input_ids))
    if max_new_tokens <= 0:
        raise ValueError(
            f"Breeze prompt length {len(state.input_ids)} reaches max_seq_len={state.max_seq_len}"
        )

    def make_lane(
        rid: str,
        ids: torch.Tensor,
        mask: torch.Tensor,
        ref_codes: torch.Tensor | None,
        *,
        is_uncond: bool,
    ) -> BreezeSGLangRequestData:
        ids = torch.as_tensor(ids, dtype=torch.long)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if ref_codes is not None:
            ref_codes = torch.as_tensor(ref_codes, dtype=torch.long)
        prompt = BreezePrompt(
            input_ids=ids.tolist(),
            text_mask=mask.tolist(),
            ref_codes=ref_codes,
        )
        prefill = model.text_encoder.build_prefill_embeds(prompt, model.embed_frames)
        params = state.sampling
        temp = float(params["temperature"])
        if len(ids) + max_new_tokens > state.max_seq_len:
            raise ValueError("CFG negative prompt is longer than the conditional prompt")
        sampling = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=float(params["top_p"]),
            top_k=int(params["top_k"]),
            repetition_penalty=float(params["repetition_penalty"]),
            stop_token_ids=[int(model.eos_token_id)],
            sampling_seed=state.seed,
        )
        sampling.normalize(None)
        sampling.verify(int(model.eos_token_id) + 1)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=ids.long().tolist(),
            sampling_params=sampling,
            eos_token_ids={int(model.eos_token_id)},
            vocab_size=int(model.eos_token_id) + 1,
            extra_key=f"breeze:prompt:v1:{rid}",
        )
        req.tokenizer = None
        req._input_embeds_are_projected = True
        req._omni_prompt_only_radix = True
        req._omni_prompt_cache_key = req.extra_key
        return BreezeSGLangRequestData(
            req=req,
            input_ids=ids.long().unsqueeze(0),
            output_ids=req.output_ids,
            state=state,
            stage_payload=payload,
            prefill_input_embeds=prefill,
            input_embeds_are_projected=True,
            is_cfg_uncond=is_uncond,
            cfg_scale=state.cfg_scale,
            rng_seed=state.seed,
            generated_frames=[],
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=float(params["top_p"]),
            top_k=int(params["top_k"]),
            repetition_penalty=float(params["repetition_penalty"]),
        )

    cond = make_lane(
        payload.request_id, state.input_ids, state.text_mask, state.ref_codes,
        is_uncond=False,
    )
    if state.mode != "cfg" or state.negative_input_ids is None:
        return cond
    uncond = make_lane(
        cfg_uncond_rid(payload.request_id),
        state.negative_input_ids,
        state.negative_text_mask,
        state.negative_ref_codes,
        is_uncond=True,
    )
    cond.cfg_uncond = uncond
    uncond.cfg_cond_rid = payload.request_id
    return cond
