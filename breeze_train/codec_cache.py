"""Offline codec tokenization cache.

Uses the same audio tokenizer inference reaches (`Qwen3TTSTokenizer`, 16
codebooks) and the same encode helper (`breeze_infer.audio.encode_prompt_audio`)
so training never risks routing through the wrong codec
(`model.codec_model`, Mimi, 32 quantizers) — see plan.md L10.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breeze_infer.audio import encode_prompt_audio
from breeze_train.manifest import load_manifest

_CACHE_VERSION = "qwen3tts-v1"


def load_audio_tokenizer(ckpt_dir: str | Path, device: str) -> Any:
    """Load the bundled Qwen3TTSTokenizer, mirroring breeze_infer/runtime.py:97-108."""
    from qwen_tts import Qwen3TTSTokenizer

    ckpt_dir = Path(ckpt_dir)
    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    return Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )


def cache_key(audio_path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(audio_path).read_bytes())
    digest.update(_CACHE_VERSION.encode("utf-8"))
    return digest.hexdigest()


def encode_to_cache(
    audio_tokenizer: Any, audio_path: str | Path, cache_dir: str | Path
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{cache_key(audio_path)}.npy"
    if out_path.exists():
        return out_path

    codes = encode_prompt_audio(audio_tokenizer, audio_path)
    np.save(out_path, codes.numpy())
    return out_path


def load_codes(cache_dir: str | Path, audio_path: str | Path) -> torch.Tensor:
    cache_dir = Path(cache_dir)
    arr = np.load(cache_dir / f"{cache_key(audio_path)}.npy")
    return torch.from_numpy(arr).to(torch.int16)


def decode_codes(audio_tokenizer: Any, codes: torch.Tensor) -> np.ndarray:
    """Decode `(frames, num_codebooks)` codes back to a waveform.

    Mirrors the call shape at models/generation_breeze.py:1324-1328.
    """
    from models.generation_breeze import _extract_decoded_audio_tensor

    # Cache stores int16; the codec's embedding lookup requires Long indices.
    # `Qwen3TTSTokenizer.decode`'s internal `_to_tensor` only casts non-tensor
    # inputs (see qwen_tts/inference/qwen3_tts_tokenizer.py:288-295) so an
    # already-tensor int16 array rides through unchanged and fails at
    # F.embedding. Cast explicitly here; never touch the on-disk int16 format.
    decoded = audio_tokenizer.decode({"audio_codes": [codes.to(torch.long)]})
    audio = _extract_decoded_audio_tensor(decoded)
    return audio.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode manifest audio into the offline codec cache"
    )
    parser.add_argument("ckpt_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    audio_tokenizer = load_audio_tokenizer(args.ckpt_dir, args.device)
    records = load_manifest(args.manifest)

    seen: set[str] = set()
    for record in records:
        for audio_path in (record.audio_path, record.ref_audio_path):
            if audio_path is None or audio_path in seen:
                continue
            seen.add(audio_path)
            out_path = encode_to_cache(audio_tokenizer, audio_path, args.cache_dir)
            print(f"encoded {audio_path} -> {out_path}")

    print(f"codec cache built: {len(seen)} distinct clips")


if __name__ == "__main__":
    main()
