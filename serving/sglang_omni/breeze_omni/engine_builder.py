"""SGLang engine construction and checkpoint-side Breeze resources."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.pipeline_state import store_state

from .model_runner import BreezeTTSModelRunner
from .payload_types import BreezeTtsState
from .request_builders import build_sglang_breeze_request
from .scheduler import BreezeScheduler

logger = logging.getLogger(__name__)

_CODEC_CACHE: dict[tuple[str, str, str], Any] = {}


def load_breeze_codec(
    model_path: str, *, device: str, dtype: str, attn_implementation: str | None = None
) -> Any:
    key = (str(model_path), str(device), str(dtype))
    if key not in _CODEC_CACHE:
        from sglang_omni.models.qwen3_tts.compat import (
            apply_qwen_tts_transformers_compatibility_patches,
        )
        apply_qwen_tts_transformers_compatibility_patches()
        from qwen_tts import Qwen3TTSTokenizer
        kwargs: dict[str, Any] = {
            "device_map": device,
            "dtype": getattr(torch, dtype) if isinstance(dtype, str) else dtype,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        _CODEC_CACHE[key] = Qwen3TTSTokenizer.from_pretrained(
            f"{model_path.rstrip('/')}/audio_tokenizer", **kwargs
        )
    return _CODEC_CACHE[key]


class CodeChunkStream:
    """Stream codec frames to the vocoder in chunks rather than one per decode step.

    The vocoder decodes request by request, so one message per 80 ms frame per
    stream saturates it long before the engine does. A small first chunk keeps
    time-to-first-audio low; later chunks amortize the vocoder call. OmniScheduler
    calls ``flush`` at request end, which ships whatever is still pending.
    """

    def __init__(self, first_frames: int = 4, chunk_frames: int = 12) -> None:
        self.first_frames = first_frames
        self.chunk_frames = chunk_frames

    def __call__(self, request_id: str, data: Any, req_output: Any) -> list[Any]:
        del req_output
        if not self._streams(data):
            return []
        pending = len(data.generated_frames) - data.stream_frames_sent
        threshold = self.chunk_frames if data.stream_frames_sent else self.first_frames
        return self._emit(request_id, data) if pending >= threshold else []

    def flush(self, request_id: str, data: Any) -> list[Any]:
        if not self._streams(data) or len(data.generated_frames) == data.stream_frames_sent:
            return []
        return self._emit(request_id, data)

    @staticmethod
    def _streams(data: Any) -> bool:
        params = data.stage_payload.request.params or {}
        return bool(params.get("stream")) and not data.is_cfg_uncond

    @staticmethod
    def _emit(request_id: str, data: Any) -> list[Any]:
        from sglang_omni.scheduling.messages import OutgoingMessage

        codes = torch.stack(data.generated_frames[data.stream_frames_sent:]).to(dtype=torch.long)
        data.stream_frames_sent = len(data.generated_frames)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=codes,
                target="vocoder",
                metadata={"modality": "audio_codes", "stream": True, "num_quantizers": 16},
            )
        ]


class BreezeTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Breeze TTS 2"
    context_length = 2048
    model_arch_override = "BreezeForConditionalGeneration"

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": 12,
            "max_queued_requests": 32,
            "cuda_graph_max_bs": 12,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.56,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del gpu_id, server_args
        from .components.text_encoder import BreezeTextEncoder

        model = model_worker.model_runner.model
        dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        model.text_encoder = BreezeTextEncoder.from_checkpoint(
            checkpoint_dir, device=device, dtype=dtype
        )
        model.depth_decoder.capture_graph(device, dtype)
        logger.info("Breeze depth decoder: captured CUDA graph for %d lanes", model.depth_decoder.max_rows)

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return BreezeTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        def build_request(payload: Any) -> Any:
            raw_state = payload.data
            state = (
                BreezeTtsState.from_dict(raw_state)
                if isinstance(raw_state, dict)
                else raw_state
            )
            if not isinstance(state, BreezeTtsState):
                raise TypeError("Breeze preprocessing did not produce BreezeTtsState")
            return build_sglang_breeze_request(payload, model=model, state=state)

        def apply_result(data: Any) -> Any:
            if data.state is None:
                raise RuntimeError("Breeze SGLang request has no pipeline state")
            return store_state(data.stage_payload, data.state)

        self._stream_output_builder = CodeChunkStream()
        return build_request, apply_result

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
        }
        kwargs.update(self.extra_scheduler_callbacks())
        kwargs.update(self.extra_scheduler_kwargs())
        return BreezeScheduler(**kwargs)

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": self._stream_output_builder,
            "request_build_max_workers": 4,
            "request_build_max_pending": 16,
        }

