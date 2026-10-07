"""SGLang-native Qwen3 backbone and Breeze codec-frame heads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from transformers import Qwen3Config
from torch import nn

from .components.depth_decoder import BreezeDepthDecoder


class BreezeTTSModel(nn.Module):
    """Breeze frame model; the causal backbone KV is SGLang's paged KV pool."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        raw_backbone = getattr(config, "backbone_config", None)
        if not isinstance(raw_backbone, dict):
            raise ValueError("Breeze checkpoint is missing backbone_config")
        backbone_values = dict(raw_backbone)
        # Breeze prompts and generated frames always use precomputed embeds;
        # keep the unused base text-token embedding tiny.
        backbone_values["vocab_size"] = 1
        backbone_values["tie_word_embeddings"] = True
        backbone_config = Qwen3Config(**backbone_values)
        self.backbone = Qwen3ForCausalLM(
            backbone_config, quant_config=quant_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        self.hidden_size = int(backbone_config.hidden_size)
        self.audio_vocab_size = int(getattr(config, "audio_vocab_size", None) or 2051)
        self.num_codebooks = int(
            getattr(config, "audio_num_codebooks", None)
            or getattr(config, "num_codebooks", None)
            or 16
        )
        self.codec_codebook_size = int(
            getattr(config, "codec_codebook_size", None) or 2048
        )
        if self.num_codebooks != 16 or self.audio_vocab_size != 2051:
            raise ValueError(
                f"This Breeze checkpoint requires 16 codebooks and vocab 2051; got "
                f"{self.num_codebooks} and {self.audio_vocab_size}"
            )
        self.eos_token_id = self.audio_vocab_size
        self.lm_head = nn.Linear(self.hidden_size, self.audio_vocab_size + 1, bias=False)
        depth_config = getattr(config, "depth_decoder_config", None)
        if depth_config is None:
            raise ValueError("Breeze checkpoint is missing depth_decoder_config")
        server_args = _server_args()
        self.depth_decoder = BreezeDepthDecoder(
            depth_config, max_rows=int(getattr(server_args, "max_running_requests", 4))
        )
        self.text_encoder: nn.Module | None = None
        max_running = int(getattr(server_args, "max_running_requests", 4))
        parameter = self.backbone.model.embed_tokens.weight
        self.decode_feedback_buffer = torch.zeros(
            (max_running + 1, self.hidden_size),
            dtype=parameter.dtype,
            device=parameter.device,
        )


    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    @property
    def frame_embedding(self) -> nn.Embedding:
        # tie_codebooks_embeddings: the checkpoint stores the backbone audio-frame
        # table once, as depth_decoder.model.embed_tokens (models/breeze.py:1102-1107).
        return self.depth_decoder.model.embed_tokens

    def embed_frames(self, codes: torch.LongTensor) -> torch.Tensor:
        return self.depth_decoder.embed_frames(codes)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del kwargs
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_decode():
            batch = int(input_ids.shape[0])
            input_embeds = self.decode_feedback_buffer[:batch]
        hidden = self.backbone.model(input_ids, positions, forward_batch, input_embeds)
        if hidden.ndim == 2:
            extend_lens = getattr(forward_batch, "extend_seq_lens", None)
            if (
                extend_lens is not None
                and hidden.shape[0] != len(extend_lens)
                and forward_mode is not None
                and forward_mode.is_extend()
            ):
                last = torch.cumsum(extend_lens, dim=0).long() - 1
                hidden = hidden.index_select(0, last)
        elif hidden.ndim == 3:
            hidden = hidden[:, -1]
        # Raw logits. CFG pairing, codec-token suppression and sampling happen in
        # BreezeTTSModelRunner, outside the decode CUDA graph, where rids are known.
        logits = self.lm_head(hidden).float()
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not getattr(self.config, "tie_codebooks_embeddings", False):
            raise ValueError("Breeze serving requires tie_codebooks_embeddings=True checkpoints")
        depth_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        backbone_keys: list[str] = []

        def backbone_weights():
            for name, tensor in weights:
                if name.startswith(("codec_model.", "text_encoder", "embed_text_tokens.")):
                    continue
                if name.startswith("backbone_model.embed_tokens."):
                    raise ValueError(f"unexpected untied Breeze frame embedding {name}")
                if name.startswith("backbone_model."):
                    key = name[len("backbone_model."):]
                    if key.startswith(("layers.", "norm.")):
                        key = "model." + key
                    backbone_keys.append(key)
                    yield key, tensor
                    continue
                if name.startswith("depth_decoder."):
                    depth_weights.append((name[len("depth_decoder."):], tensor))
                    continue
                if name == "lm_head.weight":
                    _copy_exact(self.lm_head.weight, tensor, name)
                    loaded.add("lm_head.weight")
                    continue
                raise ValueError(f"unmapped Breeze checkpoint tensor {name}")

        self.backbone.load_weights(backbone_weights())
        # 11 tensors per Qwen3 decoder layer plus the final norm.
        expected = 11 * len(self.backbone.model.layers) + 1
        if len(set(backbone_keys)) != expected:
            raise ValueError(f"Breeze backbone loaded {len(set(backbone_keys))} tensors, expected {expected}")
        if "lm_head.weight" not in loaded:
            raise ValueError("Breeze checkpoint is missing lm_head.weight")
        self.depth_decoder.load_weights(iter(depth_weights))
        # Inference only: runner hooks (frame embeddings, sampling) run outside
        # SGLang's no-grad forward and must not build autograd graphs.
        self.requires_grad_(False)
        return loaded


def _copy_exact(param: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if param.shape != tensor.shape:
        raise ValueError(f"Breeze weight {name} shape {tuple(tensor.shape)} != {tuple(param.shape)}")
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _server_args() -> Any:
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except Exception:
        return None
