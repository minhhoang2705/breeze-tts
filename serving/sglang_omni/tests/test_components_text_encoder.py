import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from breeze_omni.components.prompt import build_prompts
from breeze_omni.components.text_encoder import BreezeTextEncoder

CHECKPOINT = Path("/home/minhtranh/works/SLP/TTS/research/breeze-tts-2")
GOLDENS = Path(__file__).resolve().parents[1] / "goldens"
CASES = ["plain_vi", "plain_en", "instruction_en", "ref_clone", "ref_edit"]


@pytest.fixture(scope="module")
def prefill_components():
    if not torch.cuda.is_available():
        pytest.skip("prefill parity requires CUDA")
    device = torch.device("cuda")
    encoder = BreezeTextEncoder.from_checkpoint(CHECKPOINT, device)
    with (CHECKPOINT / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    key = "depth_decoder.model.embed_tokens.weight"
    with safe_open(CHECKPOINT / index[key], framework="pt", device="cpu") as f:
        table = f.get_tensor(key).to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, tokenizer_type="gemma", fix_mistral_regex=False)
    def embed_frames(codes):
        q = codes.shape[-1]
        offsets = torch.arange(q, device=device) * 2051
        return table[codes.to(device=device, dtype=torch.long) + offsets].sum(dim=-2)
    return encoder, tokenizer, embed_frames


@pytest.mark.parametrize("case", CASES)
def test_prefill_embeddings_match_golden(case, prefill_components):
    encoder, tokenizer, embed_frames = prefill_components
    golden = torch.load(GOLDENS / f"{case}.pt", map_location="cpu", weights_only=False)
    inputs = golden["inputs"]
    main, negative, _ = build_prompts(
        tokenizer,
        text=inputs["text"],
        instruction=inputs.get("instruction"),
        ref_text=inputs.get("ref_text"),
        ref_codes=golden.get("ref_codes"),
        speaker=inputs.get("speaker", "S0"),
        cfg_scale=4.0 if case in {"instruction_en", "ref_edit"} else 1.0,
    )
    actual = encoder.build_prefill_embeds(main, embed_frames).float().cpu()
    expected = golden["prefill_embeds"].float()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.999
    if negative is not None:
        actual_negative = encoder.build_prefill_embeds(negative, embed_frames).float().cpu()
        expected_negative = golden["neg_prefill_embeds"].float()
        assert torch.nn.functional.cosine_similarity(actual_negative.flatten(), expected_negative.flatten(), dim=0).item() > 0.999
