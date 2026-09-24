from __future__ import annotations

import pytest
import torch

from breeze_train.collator import BreezeDataCollator


class _FakeConfig:
    audio_token_id = 262144
    audio_eos_token_id = 262145


def _feature(seq_len: int, n_frames: int, n_text_segments: int) -> dict:
    return {
        "input_ids": torch.arange(2, 2 + seq_len, dtype=torch.long),
        "labels": torch.full((seq_len,), -100, dtype=torch.long),
        "text_ids_mask": torch.tensor([True] * seq_len, dtype=torch.bool),
        "text_ids_len": torch.tensor([seq_len // max(n_text_segments, 1)] * n_text_segments, dtype=torch.long),
        "input_values": torch.zeros((n_frames, 16), dtype=torch.int16),
    }


def test_right_pads_shorter_sample() -> None:
    collator = BreezeDataCollator(pad_token_id=0, config=_FakeConfig())
    short = _feature(seq_len=3, n_frames=1, n_text_segments=1)
    long = _feature(seq_len=6, n_frames=2, n_text_segments=1)

    batch = collator([short, long])

    # sample 0 (short) is right-padded: real content at the front, padding at the end.
    assert batch["attention_mask"][0, :3].all()
    assert not batch["attention_mask"][0, 3:].any()
    assert torch.equal(batch["input_ids"][0, 0], short["input_ids"][0])

    # sample 1 (long) has no padding at all.
    assert batch["attention_mask"][1, :6].all()


def test_padded_labels_and_text_mask() -> None:
    collator = BreezeDataCollator(pad_token_id=0, config=_FakeConfig())
    short = _feature(seq_len=3, n_frames=1, n_text_segments=1)
    long = _feature(seq_len=6, n_frames=2, n_text_segments=1)

    batch = collator([short, long])

    assert torch.all(batch["labels"][0, 3:] == -100)
    assert not batch["text_ids_mask"][0, 3:].any()


def test_text_ids_len_stays_flat_and_matches_mask_sum() -> None:
    collator = BreezeDataCollator(pad_token_id=0, config=_FakeConfig())
    a = _feature(seq_len=4, n_frames=1, n_text_segments=2)
    b = _feature(seq_len=6, n_frames=2, n_text_segments=3)

    batch = collator([a, b])

    assert batch["text_ids_len"].ndim == 1
    assert int(batch["text_ids_len"].shape[0]) == 5  # 2 + 3 segments, never stacked/padded
    assert int(batch["text_ids_len"].sum()) == int(batch["text_ids_mask"].sum())


def test_input_values_concatenation_order_and_dtype() -> None:
    collator = BreezeDataCollator(pad_token_id=0, config=_FakeConfig())
    a = _feature(seq_len=3, n_frames=2, n_text_segments=1)
    a["input_values"] = torch.zeros((2, 16), dtype=torch.int16)
    b = _feature(seq_len=3, n_frames=3, n_text_segments=1)
    b["input_values"] = torch.ones((3, 16), dtype=torch.int16)

    batch = collator([a, b])

    assert batch["input_values"].shape == (1, 5, 16)
    assert batch["input_values"].dtype == torch.long
    # row `frames_a` (index 2) is the first row of sample B.
    assert torch.all(batch["input_values"][0, 2] == 1)
    assert torch.all(batch["input_values"][0, :2] == 0)


def test_inconsistent_feature_raises() -> None:
    collator = BreezeDataCollator(pad_token_id=0, config=_FakeConfig())
    bad = _feature(seq_len=3, n_frames=1, n_text_segments=1)
    bad["text_ids_len"] = torch.tensor([1, 1], dtype=torch.long)  # mismatched vs mask sum
    with pytest.raises(AssertionError):
        collator([bad])


def test_post_init_rejects_pad_equal_audio_token_id() -> None:
    with pytest.raises(ValueError, match="audio_token_id"):
        BreezeDataCollator(pad_token_id=262144, config=_FakeConfig())


def test_post_init_rejects_pad_equal_audio_eos_token_id() -> None:
    with pytest.raises(ValueError, match="audio_eos_token_id"):
        BreezeDataCollator(pad_token_id=262145, config=_FakeConfig())
