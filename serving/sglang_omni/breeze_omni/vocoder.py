"""Stateful 24 kHz streaming decode for Breeze's Qwen3-TTS codec frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import torch


from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

from .payload_types import BreezeTtsState



_LOGGER = logging.getLogger(__name__)

@dataclass
class BreezeCodecState:
    decoder_state: Qwen3TTSIncrementalCodecState = field(default_factory=Qwen3TTSIncrementalCodecState)
    pending_codes: list[torch.Tensor] = field(default_factory=list)
    audio_chunks: list[torch.Tensor] = field(default_factory=list)


class BreezeStreamingVocoder(StreamingVocoderBase[BreezeCodecState, None]):
    """Decode incoming frame chunks incrementally, with abort-owned codec state."""

    def __init__(self, tokenizer: Any, *, device: str, max_batch_size: int = 8):
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.incremental_decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
        super().__init__(
            self._decode_buffered,
            sample_rate=24000,
            stream_source_hint="Breeze TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=1,
        )

    @property
    def row_pool_occupancy(self) -> int:
        return len(self._stream_states)

    def create_stream_state(self, request_id: str) -> BreezeCodecState:
        del request_id
        return BreezeCodecState()

    def latch_stream_contract(self, request_id: str, state: BreezeCodecState, source: Any, *, origin: str) -> None:
        del request_id, state, source, origin

    def validate_chunk(self, request_id: str, state: BreezeCodecState, codes: torch.Tensor) -> torch.Tensor:
        del state
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        if codes.ndim != 2 or codes.shape[-1] != 16:
            raise ValueError(f"Breeze codec chunks must have shape [T,16], got {tuple(codes.shape)}")
        codes = codes.to(device=self.device, dtype=torch.long)
        if codes.numel() and bool(((codes < 0) | (codes >= 2048)).any()):
            raise ValueError(f"Breeze request {request_id} emitted a code outside [0,2048)")
        return codes

    def ingest(self, request_id: str, state: BreezeCodecState, codes: torch.Tensor) -> None:
        del request_id
        state.pending_codes.append(codes)

    # Codec state tensors persist across chunks; with autograd on, every chunk's
    # graph stays reachable from the state and GPU memory grows ~28 MiB per frame.
    @torch.inference_mode()
    def decode_delta(self, request_id: str, state: BreezeCodecState, *, is_final: bool) -> torch.Tensor | None:
        del is_final
        if not state.pending_codes:
            return None
        fresh = torch.cat(state.pending_codes, dim=0)
        state.pending_codes.clear()
        waveform = self.incremental_decoder.decode(
            fresh.transpose(0, 1).unsqueeze(0), state.decoder_state
        )
        waveform = waveform.reshape(-1).detach().float()
        state.audio_chunks.append(waveform)
        return waveform

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: BreezeCodecState
    ) -> dict[str, Any]:
        waveform = (
            torch.cat(state.audio_chunks).cpu()
            if state.audio_chunks
            else self.fallback_full_decode(request_id, payload, state)
        )
        return audio_waveform_payload(
            waveform if waveform is not None else [],
            sample_rate=24000,
            modality="audio",
            source_hint="Breeze TTS",
        )

    @torch.inference_mode()
    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: BreezeCodecState
    ) -> torch.Tensor | None:
        del request_id, state
        codes = _get_state(payload).generated_codes
        if codes is None or not torch.as_tensor(codes).numel():
            return None
        audio_codes = torch.as_tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
        decoded, _ = self.tokenizer.decode({"audio_codes": audio_codes})
        return torch.as_tensor(decoded[0], dtype=torch.float32).reshape(-1)

    def release_stream_resources(self, request_id: str, state: BreezeCodecState) -> None:
        state.pending_codes.clear()
        state.audio_chunks.clear()
        state.decoder_state = Qwen3TTSIncrementalCodecState()
        _LOGGER.info(
            "Breeze vocoder row pool occupancy=%d after request=%s cleanup",
            self.row_pool_occupancy,
            request_id,
        )
    @torch.inference_mode()
    def _decode_buffered(self, payload: StagePayload) -> StagePayload:
        state = _get_state(payload)
        codes = state.generated_codes
        waveform = torch.empty(0, dtype=torch.float32)
        if codes is not None:
            frames = torch.as_tensor(codes, dtype=torch.long, device=self.device)
            if frames.numel():
                audio_codes = frames.unsqueeze(0)
                decoded, sample_rate = self.tokenizer.decode(
                    {"audio_codes": audio_codes}
                )
                waveform = torch.as_tensor(
                    decoded[0], dtype=torch.float32
                ).cpu()
                state.sample_rate = int(sample_rate)
        state.sample_rate = 24000
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=audio_waveform_payload(
                waveform,
                sample_rate=state.sample_rate,
                modality="audio",
                source_hint="Breeze TTS",
            ),
        )


def _get_state(payload: StagePayload) -> BreezeTtsState:
    if isinstance(payload.data, BreezeTtsState):
        return payload.data
    if isinstance(payload.data, dict):
        return BreezeTtsState.from_dict(payload.data)
    raise TypeError("Breeze vocoder requires BreezeTtsState payload data")
