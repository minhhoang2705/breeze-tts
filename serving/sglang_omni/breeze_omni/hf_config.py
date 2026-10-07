"""Transformers configuration for the exported Breeze TTS checkpoint."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class BreezeHFConfig(PretrainedConfig):
    """Keep Breeze's composite checkpoint configuration intact on Transformers 5."""

    model_type = "breeze"
    is_composition = True

    def __init__(self, **kwargs: Any) -> None:
        # SGLang constructs the Qwen3 decoder from the nested backbone config;
        # retain the original nested dicts rather than coercing Breeze-only
        # codec/depth configs through AutoConfig.
        audio_vocab_size = kwargs.get("audio_vocab_size")

        for name in (
            "backbone_model_type",
            "backbone_config",
            "depth_decoder_config",
            "text_encoder_config",
            "codec_config",
            "text_vocab_size",
            "audio_vocab_size",
            "audio_num_codebooks",
            "num_codebooks",
            "codec_codebook_size",
            "audio_embed_size",
            "tie_codebooks_embeddings",
            "codebook_pad_token_id",
            "codebook_eos_token_id",
            "audio_token_id",
            "audio_eos_token_id",
            "text_encoder_proj_type",
            "text_encoder_layer_idx",
        ):
            setattr(self, name, kwargs.pop(name, None))
        if audio_vocab_size is not None:
            kwargs["vocab_size"] = int(audio_vocab_size) + 1
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        for name in (
            "backbone_model_type", "backbone_config", "depth_decoder_config",
            "text_encoder_config", "codec_config", "text_vocab_size",
            "audio_vocab_size", "audio_num_codebooks", "num_codebooks",
            "codec_codebook_size", "audio_embed_size", "tie_codebooks_embeddings",
            "codebook_pad_token_id", "codebook_eos_token_id", "audio_token_id",
            "audio_eos_token_id", "text_encoder_proj_type", "text_encoder_layer_idx",
        ):
            value = getattr(self, name, None)
            if value is not None:
                result[name] = value
        return result
