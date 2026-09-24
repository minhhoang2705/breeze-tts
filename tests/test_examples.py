from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from breeze_infer.templates import AUDIO_EOS, AUDIO_TAG
from breeze_train.examples import _assert_example_invariants, build_example
from breeze_train.manifest import ManifestRecord

_SPECIAL_TOKEN_RE = re.compile(r"(<\|AUDIO\|>|<\|audio_eos\|>)")
_SPECIAL_IDS = {AUDIO_TAG: 262144, AUDIO_EOS: 262145}


class _FakeTokenizer:
    """Model-free fake tokenizer, special-token aware.

    `tests/test_templates.py`'s `_FakeTokenizer` maps every character to its
    own id and has never been driven through `_prepare_one`; fed
    `<|AUDIO|>`/`<|audio_eos|>` text it would emit per-character ids for
    those substrings too, none equal to `262144`/`262145`, which would make
    this module's id-based audio-position detection find nothing. This fake
    instead splits on the two special strings and emits the real special ids
    for them, per-character ids for everything else -- matching what the
    real tokenizer actually does (see phase-03 oracle test below).
    """

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, list[int] | torch.Tensor]:
        del add_special_tokens
        ids: list[int] = []
        for piece in _SPECIAL_TOKEN_RE.split(text):
            if not piece:
                continue
            if piece in _SPECIAL_IDS:
                ids.append(_SPECIAL_IDS[piece])
            else:
                ids.extend(range(2, 2 + len(piece)))
        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, input_ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        chars = []
        for token_id in input_ids:
            token_id = int(token_id)
            if token_id == _SPECIAL_IDS[AUDIO_TAG]:
                chars.append(AUDIO_TAG)
            elif token_id == _SPECIAL_IDS[AUDIO_EOS]:
                chars.append(AUDIO_EOS)
            else:
                chars.append("x")
        return "".join(chars)


class _FakeModelConfig:
    audio_token_id = 262144
    audio_eos_token_id = 262145
    num_codebooks = 16


def _write_wav(path, seconds: float = 0.3, sr: int = 24000) -> str:
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return str(path)


def test_literal_label_vector_ref_clone(tmp_path) -> None:
    """Explicit expected vector: mislabelling reference frames would fail this."""
    ref_wav = _write_wav(tmp_path / "ref.wav")
    target_wav = _write_wav(tmp_path / "target.wav")
    record = ManifestRecord(
        id="r0",
        audio_path=target_wav,
        text="target speech",
        ref_audio_path=ref_wav,
        ref_text="reference text",
    )
    ref_codes = torch.zeros((3, 16), dtype=torch.int16)
    target_codes = torch.ones((4, 16), dtype=torch.int16)

    example = build_example(_FakeTokenizer(), _FakeModelConfig(), record, target_codes, ref_codes)

    l1 = len("[S0]reference text")
    l2 = len("[S0]target speech")
    expected = (
        [-100] * l1
        + [-101] * 3
        + [-100]
        + [-100] * l2
        + [262144] * 4
        + [-100]
    )
    assert example["labels"].tolist() == expected


def test_input_values_ordering_reference_then_target(tmp_path) -> None:
    ref_wav = _write_wav(tmp_path / "ref.wav")
    target_wav = _write_wav(tmp_path / "target.wav")
    record = ManifestRecord(
        id="r0",
        audio_path=target_wav,
        text="target speech",
        ref_audio_path=ref_wav,
        ref_text="reference text",
    )
    ref_codes = torch.zeros((3, 16), dtype=torch.int16)
    target_codes = torch.ones((4, 16), dtype=torch.int16)

    example = build_example(_FakeTokenizer(), _FakeModelConfig(), record, target_codes, ref_codes)

    input_values = example["input_values"]
    assert input_values.shape == (7, 16)
    assert torch.all(input_values[:3] == 0)
    assert torch.all(input_values[3:] == 1)


@pytest.mark.parametrize(
    "record_kwargs",
    [
        pytest.param({"instruction": None, "ref_audio_path": None, "ref_text": None}, id="tts_plain"),
        pytest.param({"instruction": "speak softly", "ref_audio_path": None, "ref_text": None}, id="tts_instruction"),
        pytest.param({"instruction": None, "ref_audio_path": "REF", "ref_text": "reference text"}, id="ref_clone_tata"),
        pytest.param(
            {"instruction": "speak softly", "ref_audio_path": "REF", "ref_text": "reference text"},
            id="ref_edit_tata",
        ),
    ],
)
def test_text_ids_mask_len_invariant_all_templates(tmp_path, record_kwargs) -> None:
    ref_wav = _write_wav(tmp_path / "ref.wav")
    target_wav = _write_wav(tmp_path / "target.wav")
    kwargs = dict(record_kwargs)
    ref_audio_path = kwargs.pop("ref_audio_path")
    if ref_audio_path == "REF":
        kwargs["ref_audio_path"] = ref_wav
    record = ManifestRecord(id="r0", audio_path=target_wav, text="target speech", **kwargs)

    ref_codes = torch.zeros((3, 16), dtype=torch.int16) if record.ref_audio_path else None
    target_codes = torch.ones((4, 16), dtype=torch.int16)

    example = build_example(_FakeTokenizer(), _FakeModelConfig(), record, target_codes, ref_codes)
    assert int(example["text_ids_mask"].sum()) == int(example["text_ids_len"].sum())


def test_assert_example_invariants_catches_mask_len_mismatch() -> None:
    example = {
        "input_ids": torch.tensor([2, 3, 262144, 262145], dtype=torch.long),
        "text_ids_mask": torch.tensor([True, True, False, False]),
        "text_ids_len": torch.tensor([1], dtype=torch.long),  # wrong: mask has 2 True
        "input_values": torch.ones((1, 16), dtype=torch.long),
    }
    with pytest.raises(ValueError, match="text_ids_mask"):
        _assert_example_invariants(example, _FakeModelConfig(), ref_frames=0, target_frames=1)


def test_assert_example_invariants_catches_wrong_codebook_count() -> None:
    example = {
        "input_ids": torch.tensor([2, 3, 262144, 262145], dtype=torch.long),
        "text_ids_mask": torch.tensor([True, True, False, False]),
        "text_ids_len": torch.tensor([2], dtype=torch.long),
        "input_values": torch.ones((1, 32), dtype=torch.long),  # wrong: 32 columns, not 16
    }
    with pytest.raises(ValueError, match="num_codebooks"):
        _assert_example_invariants(example, _FakeModelConfig(), ref_frames=0, target_frames=1)


# --- Oracle test: prefix must be byte-identical to the real inference builder ---

_CKPT_DIR = Path("../breeze-tts-2")
_CACHE_DIR = Path("data/codec_cache")
_MANIFEST_PATH = Path("data/dev_manifest.jsonl")


def _real_fixtures_available() -> bool:
    if not (_CKPT_DIR.exists() and _CACHE_DIR.exists() and _MANIFEST_PATH.exists()):
        return False
    from breeze_train.codec_cache import cache_key
    from breeze_train.manifest import load_manifest

    for record in load_manifest(_MANIFEST_PATH):
        for audio_path in (record.audio_path, record.ref_audio_path):
            if audio_path is None:
                continue
            if not (_CACHE_DIR / f"{cache_key(audio_path)}.npy").exists():
                return False
    return True


@pytest.mark.skipif(not _real_fixtures_available(), reason="checkpoint or codec cache not present")
@pytest.mark.parametrize("record_id", ["r0", "r1"])
def test_oracle_matches_inference_prefix(record_id: str) -> None:
    from transformers import AutoTokenizer

    import models.breeze  # noqa: F401  -- registers the "breeze" AutoConfig architecture
    from breeze_infer.templates import (
        _prepare_segment_batches,
        get_template,
        select_template_name,
    )
    from breeze_train.codec_cache import load_audio_tokenizer, load_codes
    from breeze_train.manifest import load_manifest, to_request
    from models.breeze_base_config import BreezeConfig

    records = {r.id: r for r in load_manifest(_MANIFEST_PATH)}
    record = records[record_id]

    tokenizer = AutoTokenizer.from_pretrained(str(_CKPT_DIR), fix_mistral_regex=False)
    model_config = BreezeConfig.from_pretrained(str(_CKPT_DIR))

    target_codes = load_codes(_CACHE_DIR, record.audio_path)
    ref_codes = load_codes(_CACHE_DIR, record.ref_audio_path) if record.ref_audio_path else None

    example = build_example(tokenizer, model_config, record, target_codes, ref_codes)

    audio_tokenizer = load_audio_tokenizer(_CKPT_DIR, "cuda")
    request = to_request(record)
    segs = get_template(select_template_name(request)).build_segments(request)
    prepared = _prepare_segment_batches(tokenizer, audio_tokenizer, model_config, "cpu", [segs])

    p_len = int(prepared["input_ids"].shape[1])
    assert torch.equal(example["input_ids"][:p_len], prepared["input_ids"][0])
    assert torch.equal(example["text_ids_mask"][:p_len], prepared["text_ids_mask"][0])
    assert torch.equal(example["text_ids_len"], prepared["text_ids_len"])

    n_target = int(target_codes.shape[0])
    expected_tail = torch.tensor([262144] * n_target + [262145], dtype=torch.long)
    assert torch.equal(example["input_ids"][p_len:], expected_tail)

    if prepared["input_values"] is not None:
        p_frames = int(prepared["input_values"].shape[1])
        assert torch.equal(
            example["input_values"][:p_frames].cpu(),
            prepared["input_values"][0].cpu().to(torch.long),
        )
    else:
        assert example["prompt_frames"] == 0
        assert torch.equal(example["input_values"], target_codes.to(torch.long))
