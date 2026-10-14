"""Fixed audio samples rendered by the model being trained.

`CheckpointSampler` generates the same prompts with the same seed before the
first step (the init model; skipped when resuming) and at every checkpoint
save, writing `<output_dir>/samples/step-<N>/<id>.wav`. Listening across steps
shows when a behaviour (e.g. a non-verbal tag) starts to come out and when
reading or voice quality starts to drift -- neither of which the loss shows.

Generation mirrors infer.py (speaker "S0", template chosen per request, its
CFG scale / token limits / repetition penalty, per-item seed, no CUDA-graph
fast path) but runs on the in-training model: the PEFT adapters stay
unmerged and active, the model is switched to eval mode for the duration and
restored afterwards. Only the main process generates; under DDP the other
ranks simply wait at their next collective.

Prompt file: JSONL with `id`, `text` and optionally `ref_audio` + `ref_text`
(zero-shot clone) -- the tts-eval evalset row format.

When a wandb run is active (train.py --wandb-project) each render is also logged
as `samples/<id>` audio plus `samples/frames/<id>` and `samples/hit_max_frames`
(the share of prompts that ran into MAX_NEW_TOKENS instead of stopping on EOS),
at the run's `train/global_step`.

The inference runtime assumes things the training setup does not provide, so
for the duration of a render: attention runs `eager` (training may use
flash_attention_2, which the runtime's static KV cache cannot drive), fp32
master weights of p0/p1b/p2 run as bf16 (the runtime mixes them with bf16
frozen weights), and every parameter gets its training dtype back afterwards
(building the runtime upcasts lm_head and the depth codebooks head to fp32 in
place).
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import PretrainedConfig

from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from breeze_train.codec_cache import load_audio_tokenizer
from infer import DEFAULT_CFG_SCALE, MAX_NEW_TOKENS, MAX_SEQ_LEN, REPETITION_PENALTY
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

logger = logging.getLogger("breeze_train")


def load_prompts(path: str | Path) -> list[dict]:
    prompts = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate prompt ids")
    for prompt in prompts:
        if ("ref_audio" in prompt) != ("ref_text" in prompt):
            raise ValueError(f"{path}: prompt {prompt['id']!r} needs both ref_audio and ref_text or neither")
        if "ref_audio" in prompt and not Path(prompt["ref_audio"]).is_file():
            raise ValueError(f"{path}: prompt {prompt['id']!r}: ref_audio not found: {prompt['ref_audio']}")
    return prompts


def _request(prompt: dict) -> dict:
    request = {"id": prompt["id"], "text": prompt["text"], "speaker": "S0"}
    if "ref_audio" in prompt:
        request["ref_audio_path"] = prompt["ref_audio"]
        request["ref_text"] = prompt["ref_text"].strip()
    return request


class CheckpointSampler(transformers.TrainerCallback):
    def __init__(self, trainer: Any, model_dir: str | Path, tokenizer: Any, prompts_path: str | Path,
                 seed: int = 42) -> None:
        self.trainer = trainer
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.prompts = load_prompts(prompts_path)
        self.seed = seed
        self._audio_tokenizer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            self._render(args, state)

    def on_save(self, args, state, control, **kwargs):
        self._render(args, state)

    @torch.no_grad()
    def _render(self, args, state) -> None:
        if not self.trainer.is_world_process_zero():
            return
        started = time.perf_counter()
        out_dir = Path(args.output_dir) / "samples" / f"step-{state.global_step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._audio_tokenizer is None:
            self._audio_tokenizer = load_audio_tokenizer(self.model_dir, str(args.device))

        wrapped = self.trainer.model
        model = wrapped.get_base_model() if hasattr(wrapped, "get_base_model") else wrapped
        was_training = model.training
        configs = {id(c): c for m in model.modules() if isinstance(c := getattr(m, "config", None), PretrainedConfig)}
        attn = {key: c._attn_implementation for key, c in configs.items()}
        dtypes = [(param, param.dtype) for param in model.parameters()]
        # LoRA adapters stay fp32: PEFT casts around them itself.
        masters = [(param, param.data.to("cpu", copy=True)) for name, param in model.named_parameters()
                   if param.dtype == torch.float32 and "lora_" not in name]
        frames: dict[str, int] = {}
        # Sampling reseeds every RNG per prompt and edits generation_config for inference;
        # both are put back so training (dropout streams, --resume reproducibility, saved
        # full-model configs) is unaffected by having rendered samples.
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        generation_config = copy.deepcopy(model.generation_config)
        model.eval()
        try:
            for config in configs.values():
                config._attn_implementation = "eager"
            for param, _ in masters:
                param.data = param.data.to(torch.bfloat16)
            update_generation_config_for_breeze(model)
            runtime = FastBreezeStreamingRuntime(
                model, self._audio_tokenizer,
                FastStreamingConfig(
                    max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=None,
                    fast_text_encoder=False, fast_backbone_prefill=False, fast_backbone_decode=False,
                    fast_depth_decoder=False, fast_codec=False, repetition_penalty=REPETITION_PENALTY,
                ),
                tokenizer=self.tokenizer,
            )
            for prompt in self.prompts:
                request = _request(prompt)
                set_all_seeds(self.seed)
                inputs = prepare_inputs(
                    self.tokenizer, self._audio_tokenizer, model, [request],
                    get_template(select_template_name(request)),
                    guidance_scale=DEFAULT_CFG_SCALE, guidance_scale_ref=None, guidance_scale_ins=None,
                )
                with sf.SoundFile(out_dir / f"{prompt['id']}.wav", mode="w", samplerate=runtime.sample_rate,
                                  channels=1, subtype="PCM_16") as out:
                    frames[prompt["id"]] = 0
                    for chunk in runtime.iter_audio_chunks(inputs, request_id=prompt["id"], seed=self.seed):
                        out.write(chunk.audio)
                        frames[prompt["id"]] += chunk.codec_frames
        finally:
            for key, config in configs.items():
                config._attn_implementation = attn[key]
            for param, master in masters:
                param.data = master.to(param.device)
            for param, dtype in dtypes:
                if param.dtype != dtype:
                    param.data = param.data.to(dtype)
            model.train(was_training)
            model.generation_config = generation_config
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])
        self._log_to_wandb(state.global_step, out_dir, frames)
        elapsed = time.perf_counter() - started
        self.trainer.exclude_from_throughput(elapsed)
        logger.info(f"samples step={state.global_step} n={len(self.prompts)} dir={out_dir} seconds={elapsed:.1f}")

    def _log_to_wandb(self, step: int, out_dir: Path, frames: dict[str, int]) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        texts = {prompt["id"]: prompt["text"] for prompt in self.prompts}
        logs: dict[str, Any] = {"train/global_step": step}
        for prompt_id, count in frames.items():
            logs[f"samples/{prompt_id}"] = wandb.Audio(str(out_dir / f"{prompt_id}.wav"), caption=texts[prompt_id])
            logs[f"samples/frames/{prompt_id}"] = count
        logs["samples/hit_max_frames"] = sum(c >= MAX_NEW_TOKENS for c in frames.values()) / max(len(frames), 1)
        wandb.log(logs)
