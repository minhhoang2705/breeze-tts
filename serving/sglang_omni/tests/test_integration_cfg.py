"""CFG guidance on paired SGLang lanes (BreezeTTSModelRunner._guide_cfg_pairs)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from breeze_omni.model_runner import BreezeTTSModelRunner

GOLDENS = Path(__file__).resolve().parents[1] / "goldens"


def _lane(rid: str, *, scale: float = 1.0, has_twin: bool = False, uncond: bool = False):
    data = SimpleNamespace(
        is_cfg_uncond=uncond,
        cfg_uncond=object() if has_twin else None,
        cfg_scale=scale,
    )
    return SimpleNamespace(request_id=rid, data=data)


def test_mixed_batch_guides_only_paired_rows() -> None:
    # Twin rows are deliberately not adjacent to their cond rows.
    requests = [
        _lane("a", scale=4.0, has_twin=True),
        _lane("plain"),
        _lane("b", scale=2.0, has_twin=True),
        _lane("b_uncond", uncond=True),
        _lane("a_uncond", uncond=True),
    ]
    logits = torch.randn(len(requests), 2052)
    original = logits.clone()
    BreezeTTSModelRunner._guide_cfg_pairs(logits, requests)

    guided_a = original[4] + 4.0 * (original[0] - original[4])
    guided_b = original[3] + 2.0 * (original[2] - original[3])
    torch.testing.assert_close(logits[0], guided_a)
    torch.testing.assert_close(logits[4], guided_a)
    torch.testing.assert_close(logits[2], guided_b)
    torch.testing.assert_close(logits[3], guided_b)
    torch.testing.assert_close(logits[1], original[1])


def test_cond_lane_without_its_twin_is_an_error() -> None:
    requests = [_lane("a", scale=4.0, has_twin=True), _lane("plain")]
    with pytest.raises(RuntimeError, match="without its uncond twin"):
        BreezeTTSModelRunner._guide_cfg_pairs(torch.zeros(2, 2052), requests)


@pytest.mark.parametrize("case", ("instruction_en", "ref_edit"))
def test_guided_code0_matches_breeze_cfg_rollout(case: str) -> None:
    path = GOLDENS / f"{case}.pt"
    if not path.is_file():
        pytest.skip(f"golden {case} not generated (tools/make_goldens.py)")
    golden = torch.load(path, map_location="cpu", weights_only=False)
    cond = golden["cfg_code0_logits_cond"].float()
    uncond = golden["cfg_code0_logits_uncond"].float()
    steps = cond.shape[0]
    requests = []
    for step in range(steps):
        requests += [_lane(f"r{step}", scale=4.0, has_twin=True), _lane(f"r{step}_uncond", uncond=True)]
    logits = torch.stack((cond, uncond), dim=1).reshape(2 * steps, -1)
    BreezeTTSModelRunner._guide_cfg_pairs(logits, requests)
    # Breeze order: guide raw logits first, then suppress the reserved codec ids.
    logits[:, 2048:2051] = float("-inf")
    assert torch.equal(logits[0::2].argmax(dim=-1), golden["cfg_frames"][:, 0])
