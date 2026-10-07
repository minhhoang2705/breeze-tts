from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from breeze_omni.components.prompt import build_prompts


CHECKPOINT = Path("/home/minhtranh/works/SLP/TTS/research/breeze-tts-2")
GOLDENS = Path(__file__).resolve().parents[1] / "goldens"


def _load_tokenizer():
    return AutoTokenizer.from_pretrained(CHECKPOINT, tokenizer_type="gemma", fix_mistral_regex=False)

@pytest.mark.parametrize("case", ["plain_vi", "plain_en", "instruction_en", "ref_clone", "ref_edit"])
def test_prompt_matches_golden(case):
    golden = torch.load(GOLDENS / f"{case}.pt", map_location="cpu", weights_only=False)
    inputs = golden["inputs"]
    cfg = 4.0 if case in {"instruction_en", "ref_edit"} else 1.0
    main, negative, mode = build_prompts(
        _load_tokenizer(),
        text=inputs["text"],
        instruction=inputs.get("instruction"),
        ref_text=inputs.get("ref_text"),
        ref_codes=golden.get("ref_codes"),
        speaker=inputs.get("speaker", "S0"),
        cfg_scale=cfg,
    )
    assert main.input_ids == golden["input_ids"].tolist()
    assert main.text_mask == golden["text_mask"].tolist()
    if "neg_input_ids" in golden:
        assert negative is not None
        assert negative.input_ids == golden["neg_input_ids"].tolist()
        assert negative.text_mask == golden["neg_text_mask"].tolist()
        assert mode == "cfg"
    else:
        assert negative is None
        assert mode == "cond"


def test_cfg_zero_selects_negative_only():
    tokenizer = _load_tokenizer()
    main, negative, mode = build_prompts(tokenizer, text="Hello.", instruction="Warmly", cfg_scale=0.0)
    assert mode == "neg_only"
    assert negative is None
    cond, _, _ = build_prompts(tokenizer, text="Hello.", instruction="Warmly")
    assert main.input_ids != cond.input_ids


def test_clone_does_not_define_negative_prompt():
    with pytest.raises(ValueError, match="does not define a negative prompt"):
        build_prompts(_load_tokenizer(), text="Hello.", ref_text="Hi.", ref_codes=torch.zeros(2, 16, dtype=torch.long), cfg_scale=4.0)
