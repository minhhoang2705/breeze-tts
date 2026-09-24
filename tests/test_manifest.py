from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from breeze_train.manifest import load_manifest, to_request


def _write_wav(path, seconds: float = 0.5, sr: int = 24000) -> str:
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return str(path)


def _write_manifest(path, lines: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(path)


def test_load_manifest_happy_path(tmp_path) -> None:
    wav_a = _write_wav(tmp_path / "a.wav")
    wav_b = _write_wav(tmp_path / "b.wav")
    wav_c = _write_wav(tmp_path / "c.wav")
    manifest_path = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"id": "r0", "audio_path": wav_a, "text": "hello there"},
            {
                "id": "r1",
                "audio_path": wav_b,
                "text": "with instruction",
                "instruction": "speak softly",
            },
            {
                "id": "r2",
                "audio_path": wav_c,
                "text": "with reference",
                "ref_audio_path": wav_a,
                "ref_text": "reference text",
            },
        ],
    )

    records = load_manifest(manifest_path)
    assert len(records) == 3

    request = to_request(records[1])
    assert request["instruction"] == "speak softly"


def test_load_manifest_missing_audio_raises(tmp_path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.jsonl",
        [{"id": "r0", "audio_path": str(tmp_path / "missing.wav"), "text": "hi"}],
    )
    with pytest.raises(ValueError, match="r0"):
        load_manifest(manifest_path)


def test_load_manifest_empty_text_raises(tmp_path) -> None:
    wav_a = _write_wav(tmp_path / "a.wav")
    manifest_path = _write_manifest(
        tmp_path / "manifest.jsonl",
        [{"id": "r0", "audio_path": wav_a, "text": "   "}],
    )
    with pytest.raises(ValueError, match="r0"):
        load_manifest(manifest_path)


def test_load_manifest_ref_audio_without_ref_text_raises(tmp_path) -> None:
    wav_a = _write_wav(tmp_path / "a.wav")
    manifest_path = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {
                "id": "r0",
                "audio_path": wav_a,
                "text": "hi",
                "ref_audio_path": wav_a,
            }
        ],
    )
    with pytest.raises(ValueError, match="r0"):
        load_manifest(manifest_path)


def test_load_manifest_duplicate_id_raises(tmp_path) -> None:
    wav_a = _write_wav(tmp_path / "a.wav")
    wav_b = _write_wav(tmp_path / "b.wav")
    manifest_path = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"id": "dup", "audio_path": wav_a, "text": "hi"},
            {"id": "dup", "audio_path": wav_b, "text": "hi again"},
        ],
    )
    with pytest.raises(ValueError, match="dup"):
        load_manifest(manifest_path)
