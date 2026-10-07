"""Checkpoint-compatible T5Gemma2 encoder, porting Breeze model logic.

Weight layout mirrors models/breeze.py:947-1006 and segment encoding/merging
mirrors models/breeze.py:1243-1350,1352-1460,1544-1628. Transformers 5.12's
native T5Gemma2TextEncoder keys/shapes match the checkpoint's indexed
``text_encoder.*`` names; only those tensors and ``text_encoder_proj.*`` are read.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

from .prompt import BreezePrompt


class BreezeTextEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projection: nn.Linear, *, audio_token_id: int, audio_eos_token_id: int, codebook_eos_token_id: int, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.audio_token_id = audio_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.hidden_size = hidden_size

    @classmethod
    def from_checkpoint(cls, model_dir: str | Path, device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> "BreezeTextEncoder":
        model_dir = Path(model_dir)
        with (model_dir / "config.json").open() as f:
            model_cfg = json.load(f)
        raw_cfg = dict(model_cfg["text_encoder_config"])
        raw_cfg.pop("model_type", None)
        encoder_cfg = T5Gemma2TextConfig(**raw_cfg)
        encoder = T5Gemma2TextEncoder(encoder_cfg)
        projection = nn.Linear(encoder_cfg.hidden_size, model_cfg["hidden_size"], bias=False)

        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        wanted = {name for name in weight_map if name.startswith("text_encoder.") or name.startswith("text_encoder_proj.")}
        shards = sorted({weight_map[name] for name in wanted})
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in wanted:
                        local_name = key.removeprefix("text_encoder.") if key.startswith("text_encoder.") else key.removeprefix("text_encoder_proj.")
                        state[local_name] = f.get_tensor(key)
        encoder_names = set(encoder.state_dict())
        enc_state = {k: v for k, v in state.items() if k in encoder_names}
        if set(enc_state) != encoder_names:
            missing = sorted(encoder_names - set(enc_state))
            raise ValueError(f"Checkpoint/native T5Gemma2 mismatch; missing encoder tensors: {missing[:8]}")
        if "weight" not in state or state["weight"].shape != projection.weight.shape:
            raise ValueError("text_encoder_proj.weight is absent or has an incompatible shape")
        encoder.load_state_dict(enc_state, strict=True)
        projection.load_state_dict({"weight": state["weight"]}, strict=True)
        module = cls(encoder, projection, audio_token_id=model_cfg["audio_token_id"], audio_eos_token_id=model_cfg["audio_eos_token_id"], codebook_eos_token_id=model_cfg["codebook_eos_token_id"], hidden_size=model_cfg["hidden_size"])
        module.to(device=device, dtype=dtype)
        module.eval()
        return module

    @torch.inference_mode()
    def _encode_segments(self, segments: list[torch.Tensor]) -> list[torch.Tensor]:
        if not segments:
            return []
        device = self.projection.weight.device
        dtype = self.projection.weight.dtype
        lengths = [x.numel() for x in segments]
        max_len = max(lengths)
        ids = torch.zeros((len(segments), max_len), dtype=torch.long, device=device)
        mask = torch.zeros((len(segments), max_len), dtype=torch.long, device=device)
        positions = torch.zeros_like(ids)
        for row, segment in enumerate(segments):
            n = lengths[row]
            ids[row, :n] = segment.to(device=device, dtype=torch.long)
            mask[row, :n] = 1
            positions[row, :n] = torch.arange(n, device=device)
        encoded = self.encoder(input_ids=ids, attention_mask=mask, position_ids=positions).last_hidden_state
        return [self.projection(encoded[i, :n]).to(dtype=dtype) for i, n in enumerate(lengths)]

    @torch.inference_mode()
    def build_prefill_embeds(self, prompt: BreezePrompt, embed_frames: Callable[[torch.LongTensor], torch.Tensor]) -> torch.Tensor:
        ids = torch.as_tensor(prompt.input_ids, dtype=torch.long, device=self.projection.weight.device)
        mask = torch.as_tensor(prompt.text_mask, dtype=torch.bool, device=ids.device)
        if ids.ndim != 1 or mask.shape != ids.shape:
            raise ValueError("prompt input_ids and text_mask must be aligned 1-D sequences")
        out = torch.zeros((ids.numel(), self.hidden_size), dtype=self.projection.weight.dtype, device=ids.device)
        # Preserve each contiguous text span as an independent encoder sample.
        boundaries = torch.where(mask[1:] != mask[:-1])[0] + 1
        spans = torch.tensor_split(torch.arange(ids.numel(), device=ids.device), boundaries.tolist())
        text_spans = [span for span in spans if span.numel() and bool(mask[span[0]])]
        encoded = self._encode_segments([ids[span] for span in text_spans])
        for span, features in zip(text_spans, encoded):
            out[span] = features
        audio_positions = torch.where(ids == self.audio_token_id)[0]
        codes = prompt.ref_codes
        if audio_positions.numel():
            if codes is None or codes.shape != (audio_positions.numel(), 16):
                raise ValueError("reference audio placeholder count does not match ref_codes [T,16]")
            audio_embeds = embed_frames(codes.to(device=ids.device, dtype=torch.long))
            if audio_embeds.shape != (audio_positions.numel(), self.hidden_size):
                raise ValueError("embed_frames must return [N, hidden_size]")
            out[audio_positions] = audio_embeds.to(dtype=out.dtype)
        elif codes is not None and codes.numel():
            raise ValueError("ref_codes supplied but prompt has no <|AUDIO|> positions")
        eos_positions = torch.where(ids == self.audio_eos_token_id)[0]
        if eos_positions.numel():
            eos_codes = torch.full((1, 16), self.codebook_eos_token_id, dtype=torch.long, device=ids.device)
            eos_embed = embed_frames(eos_codes)
            if eos_embed.shape != (1, self.hidden_size):
                raise ValueError("embed_frames must return [N, hidden_size]")
            out[eos_positions] = eos_embed.to(dtype=out.dtype)
        return out

    def build_prefill_embeds_batch(self, prompts: Iterable[BreezePrompt], embed_frames: Callable[[torch.LongTensor], torch.Tensor]) -> list[torch.Tensor]:
        return [self.build_prefill_embeds(prompt, embed_frames) for prompt in prompts]
