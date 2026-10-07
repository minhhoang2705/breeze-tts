#!/usr/bin/env python3
"""Generate Breeze TTS golden prompts, greedy traces, CFG traces, and audio."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = Path("/home/minhtranh/works/SLP/TTS/research/breeze-tts-2")
OUT_DIR = Path(__file__).resolve().parents[1] / "goldens"
REF_AUDIO = REPO / "data/vivoice_scale3/audio/vivoice_s0_0001491.wav"
REF_TEXT = "Rồi rồi, tới công chuyện rồi."
CASES: dict[str, dict[str, str]] = {
    "plain_vi": {"text": "Xin chào, hôm nay trời thật đẹp."},
    "plain_en": {"text": "Good morning. I hope you have a wonderful day."},
    "instruction_en": {
        "text": "The train leaves in ten minutes.",
        "instruction": "Speak warmly and clearly, like a friendly station announcer.",
    },
    "ref_clone": {
        "text": "Chúng ta sẽ bắt đầu cuộc họp ngay bây giờ.",
        "ref_text": REF_TEXT,
        "ref_audio_path": str(REF_AUDIO),
    },
    "ref_edit": {
        "text": "The meeting will begin in just a moment.",
        "instruction": "Speak in a calm, reassuring tone.",
        "ref_text": REF_TEXT,
        "ref_audio_path": str(REF_AUDIO),
    },
}


def cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32 if value.is_floating_point() else value.dtype).contiguous()


def prepare_prompt(tokenizer, audio_tokenizer, model, values: dict[str, str]) -> dict[str, Any]:
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name

    request = {**values, "speaker": "S0"}
    template = get_template(select_template_name(request))
    inputs = prepare_inputs(
        tokenizer, audio_tokenizer, model, [{**request, "id": "golden"}], template,
        guidance_scale=4.0 if template.build_negative_segments is not None else 1.0,
        guidance_scale_ref=None, guidance_scale_ins=None,
    )
    merged = model._merge_input_ids_with_input_values(
        input_ids=inputs["input_ids"], input_values=inputs.get("input_values"),
        text_ids_mask=inputs["text_ids_mask"], text_ids_len=inputs["text_ids_len"],
        attention_mask=inputs["attention_mask"],
    )
    result: dict[str, Any] = {
        "inputs": request,
        "input_ids": inputs["input_ids"][0].long(),
        "text_mask": inputs["text_ids_mask"][0].bool(),
        "ref_codes": inputs["input_values"][0].long() if inputs.get("input_values") is not None else None,
        "prefill_embeds": merged["inputs_embeds"][0],
    }
    if "cfg_negative_prompt_ids" in inputs:
        neg_merged = model._merge_input_ids_with_input_values(
            input_ids=inputs["cfg_negative_prompt_ids"],
            input_values=inputs.get("cfg_negative_input_values"),
            text_ids_mask=inputs["cfg_negative_text_ids_mask"],
            text_ids_len=inputs["cfg_negative_text_ids_len"],
            attention_mask=inputs["cfg_negative_prompt_attention_mask"],
        )
        result.update({
            "neg_input_ids": inputs["cfg_negative_prompt_ids"][0].long(),
            "neg_text_mask": inputs["cfg_negative_text_ids_mask"][0].bool(),
            "neg_ref_codes": inputs["cfg_negative_input_values"][0].long() if inputs.get("cfg_negative_input_values") is not None else None,
            "neg_prefill_embeds": neg_merged["inputs_embeds"][0],
        })
    return result


def prefill(model, embeds: torch.Tensor):
    mask = torch.ones((1, embeds.shape[0]), dtype=torch.long, device=embeds.device)
    positions = torch.arange(embeds.shape[0], dtype=torch.long, device=embeds.device).unsqueeze(0)
    out = model.backbone_model(
        inputs_embeds=embeds.unsqueeze(0), attention_mask=mask,
        position_ids=positions, use_cache=True,
    )
    hidden = out.last_hidden_state[:, -1, :]
    logits = model.lm_head(hidden.to(dtype=model.lm_head.weight.dtype)).float()
    return hidden, logits, out.past_key_values, mask


def backbone_step(model, frame: torch.Tensor, cache: Any, mask: torch.Tensor):
    embeds = model.backbone_model.embed_tokens(frame.view(1, 1, -1))
    mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)), dim=-1)
    position = torch.tensor([[mask.shape[-1] - 1]], dtype=torch.long, device=mask.device)
    out = model.backbone_model(
        inputs_embeds=embeds, attention_mask=mask, position_ids=position,
        past_key_values=cache, use_cache=True, cache_position=position[0],
    )
    hidden = out.last_hidden_state[:, -1, :]
    logits = model.lm_head(hidden.to(dtype=model.lm_head.weight.dtype)).float()
    return hidden, logits, mask


def depth_logits(model, hidden: torch.Tensor, codes_so_far: list[torch.Tensor]) -> torch.Tensor:
    # Avoid materializing all 15 large heads for every prefix (important on 12 GB cards).
    ids = torch.stack([torch.zeros_like(codes_so_far[0]), *codes_so_far], dim=1)
    depth_hidden = model.depth_decoder.model(
        input_ids=ids, backbone_last_hidden_state=hidden, use_cache=False
    ).last_hidden_state[:, -1, :]
    head_index = len(codes_so_far) - 1
    weight = model.depth_decoder.codebooks_head.weight[head_index].T
    return torch.nn.functional.linear(depth_hidden, weight).float()


def greedy_rollout(model, hidden: torch.Tensor, logits: torch.Tensor, cache: Any, mask: torch.Tensor, count: int = 8) -> dict[str, torch.Tensor]:
    codebook_size, vocab_size = int(model.config.codec_config.codebook_size), int(model.config.vocab_size)
    blocked = slice(codebook_size, vocab_size)
    frames, logits0, hidden_rows, depth_rows = [], [], [], []
    for _ in range(count):
        raw0 = logits[0].float()
        code0 = raw0.clone()
        code0[blocked] = -torch.inf
        codes = [code0.argmax().view(1)]
        rows = []
        for _q in range(int(model.config.num_codebooks) - 1):
            raw = depth_logits(model, hidden, codes)[0]
            rows.append(raw)
            valid = raw.clone()
            valid[blocked] = -torch.inf
            codes.append(valid.argmax().view(1))
        frame = torch.cat(codes).long()
        frames.append(frame)
        logits0.append(raw0)
        hidden_rows.append(hidden[0].float())
        depth_rows.append(torch.stack(rows))
        hidden, logits, mask = backbone_step(model, frame, cache, mask)
    frames_t, logits_t, depth_t = torch.stack(frames), torch.stack(logits0), torch.stack(depth_rows)
    depth_mask = torch.arange(vocab_size, device=frames_t.device).ge(codebook_size)
    all_logits_ids = torch.arange(logits_t.shape[-1], device=frames_t.device)
    code0_mask = all_logits_ids.ge(codebook_size) & all_logits_ids.lt(vocab_size)
    assert torch.equal(frames_t[:, 0], logits_t.masked_fill(code0_mask, -torch.inf).argmax(-1))
    assert torch.equal(frames_t[:, 1:], depth_t.masked_fill(depth_mask.view(1, 1, -1), -torch.inf).argmax(-1))
    return {
        "frames": frames_t, "code0_logits": logits_t,
        "backbone_hidden": torch.stack(hidden_rows), "depth_logits": depth_t,
    }


def cfg_rollout(model, cond: dict[str, Any], neg: dict[str, Any], scale: float = 4.0, count: int = 8) -> dict[str, torch.Tensor]:
    codebook_size, vocab_size = int(model.config.codec_config.codebook_size), int(model.config.vocab_size)
    blocked = slice(codebook_size, vocab_size)
    ch, cl, cc, cm = cond["hidden"], cond["logits"], cond["cache"], cond["mask"]
    nh, nl, nc, nm = neg["hidden"], neg["logits"], neg["cache"], neg["mask"]
    frames, cond0, neg0, cond_depth, neg_depth = [], [], [], [], []
    for _ in range(count):
        cr, nr = cl[0].float(), nl[0].float()
        guided = (nr + scale * (cr - nr)).clone()
        guided[blocked] = -torch.inf
        codes = [guided.argmax().view(1)]
        crows, nrows = [], []
        for _q in range(int(model.config.num_codebooks) - 1):
            cnext, nnext = depth_logits(model, ch, codes)[0], depth_logits(model, nh, codes)[0]
            crows.append(cnext)
            nrows.append(nnext)
            gd = (nnext + scale * (cnext - nnext)).clone()
            gd[blocked] = -torch.inf
            codes.append(gd.argmax().view(1))
        frame = torch.cat(codes).long()
        frames.append(frame)
        cond0.append(cr)
        neg0.append(nr)
        cond_depth.append(torch.stack(crows))
        neg_depth.append(torch.stack(nrows))
        ch, cl, cm = backbone_step(model, frame, cc, cm)
        nh, nl, nm = backbone_step(model, frame, nc, nm)
    output = {
        "cfg_frames": torch.stack(frames),
        "cfg_code0_logits_cond": torch.stack(cond0),
        "cfg_code0_logits_uncond": torch.stack(neg0),
        "cfg_depth_logits_cond": torch.stack(cond_depth),
        "cfg_depth_logits_uncond": torch.stack(neg_depth),
    }
    g0 = output["cfg_code0_logits_uncond"] + scale * (output["cfg_code0_logits_cond"] - output["cfg_code0_logits_uncond"])
    g0[:, blocked] = -torch.inf
    assert torch.equal(output["cfg_frames"][:, 0], g0.argmax(-1))
    return output


def rope_info(module: torch.nn.Module, fallback_config: Any = None) -> dict[str, Any]:
    rotary = getattr(module, "rotary_emb", None)
    config = getattr(rotary, "config", getattr(module, "config", fallback_config))
    return {
        "theta": getattr(config, "rope_theta", None),
        "scaling": getattr(config, "rope_scaling", None),
        "rope_type": getattr(rotary, "rope_type", None),
        "rotary_class": type(rotary).__name__ if rotary is not None else None,
    }


def decode_wave(audio_tokenizer: Any, frames: torch.Tensor) -> torch.Tensor:
    codes = frames.to(audio_tokenizer.device).unsqueeze(0)
    result = audio_tokenizer.decode({"audio_codes": codes})
    wave = result[0] if isinstance(result, tuple) else result
    if isinstance(wave, dict):
        wave = wave.get("audio", wave.get("waveform", wave.get("audio_values")))
        if wave is None:
            raise KeyError("Unexpected audio tokenizer decode output")
    if isinstance(wave, (tuple, list)):
        wave = wave[0]
    return torch.as_tensor(wave).squeeze().float().cpu()


def save_case(case: str, result: dict[str, Any], audio_tokenizer: Any, meta: dict[str, Any]) -> None:
    result["wave"] = decode_wave(audio_tokenizer, result["frames"])
    result["meta"] = meta
    result = {
        key: (cpu_tensor(value) if isinstance(value, torch.Tensor) else value)
        for key, value in result.items()
    }
    output = OUT_DIR / f"{case}.pt"
    torch.save(result, output)
    sf.write(output.with_suffix(".wav"), result["wave"].numpy(), 24000)
    print(f"{case}: " + json.dumps({k: list(v.shape) for k, v in result.items() if isinstance(v, torch.Tensor)}))


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    if not REF_AUDIO.is_file():
        raise FileNotFoundError(f"Reference WAV does not exist: {REF_AUDIO}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Breeze checkpoint does not exist: {args.checkpoint}")
    sys.path.insert(0, str(REPO))
    from breeze_infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(1234)
    tokenizer, model, audio_tokenizer = load_runtime(args.checkpoint, device="cuda:0", attn_implementation="eager")
    update_generation_config_for_breeze(model, {
        "depth_decoder_do_sample": False, "depth_decoder_temperature": 1.0,
        "depth_decoder_top_p": 1.0, "depth_decoder_top_k": 0,
        "do_sample": False, "top_p": 1.0, "top_k": 0,
        "max_new_tokens": 8, "temperature": 1.0,
    })
    config = model.config
    meta = {
        "compute_dtype": str(next(model.parameters()).dtype),
        "checkpoint": str(args.checkpoint),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "backbone_rope": rope_info(model.backbone_model, getattr(config, "backbone_config", None)),
        "depth_decoder_rope": rope_info(model.depth_decoder.model, model.depth_decoder.config),
        "attention_implementation": "eager",
        "fast_path_discrepancy": "Eager backbone/depth forwards without CUDA graphs; greedy rollout suppresses codec IDs 2048..2050 before argmax and uses full-sequence eager depth forwards instead of the fast path's cached depth graph.",
        "codec_sample_rate": 24000,
    }
    print("ROPE " + json.dumps({"backbone": meta["backbone_rope"], "depth_decoder": meta["depth_decoder_rope"]}, default=str))
    for case, values in CASES.items():
        set_all_seeds(1234)
        prompt = prepare_prompt(tokenizer, audio_tokenizer, model, values)
        hidden, logits, cache, mask = prefill(model, prompt["prefill_embeds"])
        result = {**prompt, "prefill_last_hidden": hidden[0], "code0_logits_prefill": logits[0]}
        result.update(greedy_rollout(model, hidden, logits, cache, mask))
        if "neg_prefill_embeds" in prompt:
            nh, nl, nc, nm = prefill(model, prompt["neg_prefill_embeds"])
            result.update(cfg_rollout(
                model,
                {"hidden": hidden, "logits": logits, "cache": cache, "mask": mask},
                {"hidden": nh, "logits": nl, "cache": nc, "mask": nm},
            ))
        save_case(case, result, audio_tokenizer, meta)


if __name__ == "__main__":
    main()
