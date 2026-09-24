import json

import pytest

from breeze_train.release import (
    DERIVED_STATEMENT,
    NOTICE_TEXT,
    validate_release_name,
    write_release_files,
)


@pytest.mark.parametrize(
    "name", ["Breeze-TTS-Vietnamese", "my-breezeblue-lora", "BREEZE_tts", "b-r-e-e-z-e"]
)
def test_rejects_breezeblue_marks_as_name(name):
    with pytest.raises(ValueError):
        validate_release_name(name)


@pytest.mark.parametrize("name", ["vi-tts-lora", "breathe-tts"])
def test_accepts_unrelated_names(name):
    validate_release_name(name)


@pytest.fixture
def base_dir(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "LICENSE").write_text("BREEZEBLUE LICENSE\n")
    return base


@pytest.fixture
def adapter_dir(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 32}))
    (adapter / "README.md").write_text("generic PEFT stub\n")
    return adapter


def test_writes_required_distribution_files(base_dir, adapter_dir):
    write_release_files(
        adapter_dir, base_dir, name="vi-tts-lora", merged=False, adapter_dir=adapter_dir,
        dataset="capleaf/viVoice",
    )
    assert (adapter_dir / "LICENSE").read_text() == "BREEZEBLUE LICENSE\n"
    assert NOTICE_TEXT in (adapter_dir / "NOTICE").read_text()
    card = (adapter_dir / "README.md").read_text()
    assert "generic PEFT stub" not in card
    assert DERIVED_STATEMENT in card
    assert "capleaf/viVoice" in card
    assert "LoRA rank: 32" in card


def test_notice_keeps_base_notice_first(base_dir, adapter_dir):
    (base_dir / "NOTICE").write_text("Upstream notice.\n")
    write_release_files(adapter_dir, base_dir, name="vi-tts", merged=True, adapter_dir=adapter_dir)
    notice = (adapter_dir / "NOTICE").read_text()
    assert notice.startswith("Upstream notice.")
    assert NOTICE_TEXT in notice


def test_missing_base_license_fails(tmp_path, adapter_dir):
    with pytest.raises(FileNotFoundError):
        write_release_files(adapter_dir, tmp_path, name="vi-tts", merged=False, adapter_dir=adapter_dir)


def test_rejected_name_writes_nothing(base_dir, adapter_dir):
    with pytest.raises(ValueError):
        write_release_files(
            adapter_dir, base_dir, name="Breeze-VI", merged=False, adapter_dir=adapter_dir
        )
    assert not (adapter_dir / "NOTICE").exists()
    assert (adapter_dir / "README.md").read_text() == "generic PEFT stub\n"
