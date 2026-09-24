"""Objective evaluation: WER/CER + speaker similarity, baseline vs tuned.

Three separate process invocations (never loaded in the same process): the
card is 12 GB and Breeze bf16 (~6.5 GiB) + Whisper large-v3 fp16 (~3 GiB) +
ECAPA (a few hundred MB) do not fit together. Select one with the `action`
subcommand; each entry point loads its model, writes its artifacts to disk,
and exits -- `del`-ing the model and calling `torch.cuda.empty_cache()`
before exit.

`faster-whisper` is a CTranslate2 runtime, not a torch model: it ships its
own CUDA/cuDNN libraries. On this machine it additionally needs
`LD_LIBRARY_PATH` to include the venv's bundled `nvidia/cudnn/lib` --
without it, ctranslate2's dlopen of libcudnn hangs rather than erroring
cleanly. `main()`'s `score-intelligibility` subcommand sets this
automatically before importing faster_whisper.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

from breeze_train.manifest import load_manifest, to_request

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_SEED = 42

_VOCAL_EVENT_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def _ensure_cudnn_on_ld_library_path() -> None:
    """faster-whisper (ctranslate2) needs the venv's bundled cuDNN (and, in
    a process that has not already imported torch, cuBLAS) on
    LD_LIBRARY_PATH or its CUDA init hangs / fails with a "library not
    found" error. Only matters for the score-intelligibility subcommand,
    and only affects this process (the caller re-execs after setting it --
    see main())."""
    import nvidia.cublas
    import nvidia.cudnn  # noqa: PLC0415

    lib_dirs = [
        str(Path(nvidia.cudnn.__path__[0]) / "lib"),
        str(Path(nvidia.cublas.__path__[0]) / "lib"),
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = current.split(":")
    missing = [d for d in lib_dirs if d not in current_parts]
    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*missing, current])
        os.execv(sys.executable, [sys.executable, *sys.argv])


def generate_for_manifest(ckpt_dir: str, manifest_path: str, out_dir: str, seed: int = DEFAULT_SEED) -> None:
    """Generate one wav per manifest record using the untouched inference path."""
    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer, model, audio_tokenizer = load_runtime(
        Path(ckpt_dir), device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=False,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)

    records = load_manifest(manifest_path)
    for record in records:
        request = to_request(record)
        template_name = select_template_name(request)

        set_all_seeds(seed)
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template(template_name),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

        import soundfile as sf

        wav_path = out_path / f"{record.id}.wav"
        with sf.SoundFile(
            wav_path, mode="w", samplerate=runtime.sample_rate, channels=1, subtype="PCM_16"
        ) as output_file:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=record.id, seed=seed):
                output_file.write(chunk.audio)

    peak_vram_gib = torch.cuda.max_memory_reserved() / 2**30
    print(f"peak_vram_gib={peak_vram_gib}")
    print(f"generated {len(records)} wavs -> {out_path}")

    del model, audio_tokenizer, runtime
    gc.collect()
    torch.cuda.empty_cache()


def _strip_vocal_events(text: str) -> str:
    return _VOCAL_EVENT_RE.sub(" ", text)


def _normalize(text: str) -> str:
    text = _strip_vocal_events(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_intelligibility(wav_dir: str, manifest_path: str, language: str = "vi") -> dict[str, Any]:
    """WER for space-delimited languages, CER otherwise (Vietnamese is
    space-delimited, so WER is the correct metric here -- 'en_zh' split
    from the original phase-10 plan was for English/Chinese; extended here
    to also route Vietnamese to WER)."""
    _ensure_cudnn_on_ld_library_path()

    import jiwer
    from faster_whisper import WhisperModel

    model_name = "large-v3"
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    records = load_manifest(manifest_path)
    refs, hyps = [], []
    for record in records:
        wav_path = Path(wav_dir) / f"{record.id}.wav"
        segments, _info = model.transcribe(str(wav_path), language=language, beam_size=5)
        hyp_text = " ".join(seg.text for seg in segments)
        refs.append(_normalize(record.text))
        hyps.append(_normalize(hyp_text))

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    result = {"wer": wer, "cer": cer, "n": len(records), "asr_model": model_name, "language": language}

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def score_speaker_similarity(wav_dir: str, manifest_path: str) -> dict[str, Any]:
    import soundfile as sf
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda:0"},
        savedir="/tmp/ecapa_cache",
    )

    def _embed(path: Path) -> torch.Tensor:
        wav_np, sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = torch.from_numpy(wav_np.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            emb = model.encode_batch(wav.to("cuda"))
        return emb.squeeze().cpu()

    records = load_manifest(manifest_path)
    sims = []
    for record in records:
        gt_emb = _embed(Path(record.audio_path))
        gen_emb = _embed(Path(wav_dir) / f"{record.id}.wav")
        cos = torch.nn.functional.cosine_similarity(gt_emb.unsqueeze(0), gen_emb.unsqueeze(0)).item()
        sims.append(cos)

    mean = sum(sims) / len(sims)
    std = (sum((s - mean) ** 2 for s in sims) / len(sims)) ** 0.5
    result = {"mean": mean, "std": std, "n": len(sims)}

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_report(
    out_path: str,
    intelligibility: dict[str, dict],
    speaker_similarity: dict[str, dict],
    config_summary: dict,
) -> None:
    lines = ["# Breeze TTS 2 Vietnamese fine-tuning: objective evaluation report", ""]
    lines.append("## Fine-tuning configuration (for reproducibility)")
    for k, v in config_summary.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Checkpoint | WER | CER | Speaker similarity (mean) | Speaker similarity (std) | N |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("baseline", "tuned"):
        wer = intelligibility[name]["wer"]
        cer = intelligibility[name]["cer"]
        sim_mean = speaker_similarity[name]["mean"]
        sim_std = speaker_similarity[name]["std"]
        n = intelligibility[name]["n"]
        lines.append(f"| {name} | {wer:.4f} | {cer:.4f} | {sim_mean:.4f} | {sim_std:.4f} | {n} |")
    lines.append("")
    if intelligibility["baseline"]["wer"] > 1.0 or intelligibility["tuned"]["wer"] > 1.0:
        lines.append(
            "*Note: WER = (substitutions + deletions + insertions) / reference_word_count "
            "has no upper bound of 1 -- it exceeds 1.0 whenever the ASR hypothesis is "
            "longer than the reference, which is exactly what happens when Whisper "
            "hallucinates an unrelated phrase for unintelligible audio (observed here for "
            "the baseline checkpoint: manual inspection showed generic hallucinated "
            "phrases like \"Hãy subscribe cho kênh...\" substituted for the actual "
            "reference sentence).*"
        )
        lines.append("")

    wer_base, wer_tuned = intelligibility["baseline"]["wer"], intelligibility["tuned"]["wer"]
    cer_base, cer_tuned = intelligibility["baseline"]["cer"], intelligibility["tuned"]["cer"]
    sim_base, sim_tuned = speaker_similarity["baseline"]["mean"], speaker_similarity["tuned"]["mean"]

    wer_regressed = wer_tuned > wer_base * 1.10
    cer_regressed = cer_tuned > cer_base * 1.10
    sim_regressed = sim_tuned < sim_base - 0.02

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "Pass threshold: tuned must not be worse than baseline by more than 10% relative "
        "on WER/CER, and mean speaker similarity must not be lower by more than 0.02 absolute."
    )
    lines.append("")
    if wer_regressed or cer_regressed or sim_regressed:
        lines.append("**Verdict: REGRESSION.** The tuned checkpoint did not meet the pass thresholds:")
        if wer_regressed:
            lines.append(f"- WER regressed: {wer_base:.4f} -> {wer_tuned:.4f}")
        if cer_regressed:
            lines.append(f"- CER regressed: {cer_base:.4f} -> {cer_tuned:.4f}")
        if sim_regressed:
            lines.append(f"- Speaker similarity regressed: {sim_base:.4f} -> {sim_tuned:.4f}")
    else:
        lines.append(
            f"**Verdict: PASS.** WER {wer_base:.4f} -> {wer_tuned:.4f}, "
            f"CER {cer_base:.4f} -> {cer_tuned:.4f}, "
            f"speaker similarity {sim_base:.4f} -> {sim_tuned:.4f}. "
            "No regression beyond the stated thresholds."
        )

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Breeze TTS 2 objective eval harness")
    sub = parser.add_subparsers(dest="action", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("ckpt_dir")
    p_gen.add_argument("manifest")
    p_gen.add_argument("out_dir")
    p_gen.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_wer = sub.add_parser("score-intelligibility")
    p_wer.add_argument("wav_dir")
    p_wer.add_argument("manifest")
    p_wer.add_argument("--out", default="outputs/eval/intelligibility.json")
    p_wer.add_argument("--name", required=True)
    p_wer.add_argument("--language", default="vi")

    p_spk = sub.add_parser("score-speaker-similarity")
    p_spk.add_argument("wav_dir")
    p_spk.add_argument("manifest")
    p_spk.add_argument("--out", default="outputs/eval/speaker_similarity.json")
    p_spk.add_argument("--name", required=True)

    p_report = sub.add_parser("report")
    p_report.add_argument("--out", default="outputs/eval/report.md")

    args = parser.parse_args()

    if args.action == "generate":
        generate_for_manifest(args.ckpt_dir, args.manifest, args.out_dir, args.seed)
    elif args.action == "score-intelligibility":
        result = score_intelligibility(args.wav_dir, args.manifest, args.language)
        out_path = Path(args.out)
        data = json.loads(out_path.read_text()) if out_path.exists() else {}
        data[args.name] = result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        print(f"{args.name}: wer={result['wer']:.4f} cer={result['cer']:.4f}")
    elif args.action == "score-speaker-similarity":
        result = score_speaker_similarity(args.wav_dir, args.manifest)
        out_path = Path(args.out)
        data = json.loads(out_path.read_text()) if out_path.exists() else {}
        data[args.name] = result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        print(f"{args.name}: mean={result['mean']:.4f} std={result['std']:.4f}")
    elif args.action == "report":
        intelligibility = json.loads(Path("outputs/eval/intelligibility.json").read_text())
        speaker_similarity = json.loads(Path("outputs/eval/speaker_similarity.json").read_text())
        config_summary = {
            "policy": "p1",
            "lora_rank": 32,
            "lora_alpha": 64,
            "base_steps": 3000,
            "warm_restart_steps": 1200,
            "total_effective_steps": 4200,
            "learning_rate": "1e-4 (cosine, two cycles)",
            "dataset": "capleaf/viVoice subset, 30,000 train records",
        }
        write_report(args.out, intelligibility, speaker_similarity, config_summary)


if __name__ == "__main__":
    main()
