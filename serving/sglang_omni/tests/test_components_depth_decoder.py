import json
from pathlib import Path
import statistics

import pytest
import torch
from safetensors import safe_open

from breeze_omni.components.depth_decoder import BreezeDepthDecoder

CHECKPOINT = Path("/home/minhtranh/works/SLP/TTS/research/breeze-tts-2")
GOLDENS = Path(__file__).resolve().parents[1] / "goldens"


@pytest.fixture(scope="module")
def decoder_and_golden():
    if not torch.cuda.is_available():
        pytest.skip("depth parity requires CUDA")
    with (CHECKPOINT / "config.json").open() as f:
        cfg = json.load(f)["depth_decoder_config"]
    with (CHECKPOINT / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    decoder = BreezeDepthDecoder(cfg, max_rows=16).cuda().to(torch.bfloat16).eval()
    selected = {name: shard for name, shard in index.items() if name.startswith("depth_decoder.")}
    weights = []
    for shard in sorted(set(selected.values())):
        with safe_open(CHECKPOINT / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in selected:
                    weights.append((name.removeprefix("depth_decoder."), f.get_tensor(name)))
    decoder.load_weights(weights)
    golden = torch.load(GOLDENS / "plain_en.pt", map_location="cpu", weights_only=False)
    return decoder, golden


def test_teacher_forced_depth_codes_match_golden(decoder_and_golden):
    """Each depth step, fed Breeze's own previous codes, predicts Breeze's next code."""
    decoder, golden = decoder_and_golden
    frames = golden["frames"].to(device="cuda", dtype=torch.long)
    steps = frames.shape[0]
    hidden = golden["backbone_hidden"].to(device="cuda", dtype=torch.bfloat16)
    greedy = (torch.zeros(steps, device="cuda"), torch.zeros(steps, device="cuda", dtype=torch.long), torch.ones(steps, device="cuda"))
    lanes = decoder._lane_inputs(hidden, frames[:, 0], *greedy, None, None, None)
    forced = torch.zeros((decoder.max_rows, frames.shape[1] - 1), dtype=torch.long, device="cuda")
    forced[:steps] = frames[:, 1:]
    with torch.inference_mode():
        predicted = decoder._run_lanes(*lanes, forced=forced)[:steps]
    matches = (predicted == frames[:, 1:]).float().mean().item()
    print(f"depth decoder teacher-forced top-1 agreement: {matches:.1%}")
    assert matches >= 0.95


def test_mixed_cfg_batch_matches_rowwise_greedy(decoder_and_golden):
    decoder, golden = decoder_and_golden
    hidden = golden["backbone_hidden"][:4].to(device="cuda", dtype=torch.bfloat16)
    code0 = golden["frames"][:4, 0].to(device="cuda", dtype=torch.long)
    uncond = hidden[[1, 3]] * 0.875
    rows = torch.tensor([1, 3], device="cuda")
    scale = torch.tensor([4.0, 2.0], device="cuda")
    temperature = torch.zeros(4, device="cuda")
    top_k = torch.zeros(4, dtype=torch.long, device="cuda")
    top_p = torch.ones(4, device="cuda")
    batched = decoder.generate(hidden, code0, temperature, top_k, top_p, cfg_uncond_hidden=uncond, cfg_rows=rows, cfg_scale=scale)
    rowwise = []
    cfg_index = {1: 0, 3: 1}
    for row in range(4):
        if row in cfg_index:
            idx = cfg_index[row]
            result = decoder.generate(hidden[row:row + 1], code0[row:row + 1], temperature[row:row + 1], top_k[row:row + 1], top_p[row:row + 1], cfg_uncond_hidden=uncond[idx:idx + 1], cfg_rows=torch.tensor([0], device="cuda"), cfg_scale=scale[idx:idx + 1])
        else:
            result = decoder.generate(hidden[row:row + 1], code0[row:row + 1], temperature[row:row + 1], top_k[row:row + 1], top_p[row:row + 1])
        rowwise.append(result)
    assert torch.equal(batched, torch.cat(rowwise, dim=0))

@pytest.mark.parametrize("batch", [1, 4, 8])
def test_cuda_graph_greedy_parity_and_step_latency(batch, decoder_and_golden):
    decoder, golden = decoder_and_golden
    source_hidden = golden["backbone_hidden"].to(device="cuda", dtype=torch.bfloat16)
    source_code = golden["frames"][:, 0].to(device="cuda", dtype=torch.long)
    hidden = source_hidden.repeat((batch + len(source_hidden) - 1) // len(source_hidden), 1)[:batch].contiguous()
    code0 = source_code.repeat((batch + len(source_code) - 1) // len(source_code))[:batch].contiguous()
    params = (torch.zeros(batch, device="cuda"), torch.zeros(batch, device="cuda", dtype=torch.long), torch.ones(batch, device="cuda"))
    eager = decoder.generate(hidden, code0, *params)
    graph = decoder.generate(hidden, code0, *params, use_cuda_graph=True)
    assert torch.equal(eager, graph)
    for _ in range(3):
        decoder.generate(hidden, code0, *params)
        decoder.generate(hidden, code0, *params, use_cuda_graph=True)
    eager_times, graph_times = [], []
    for _ in range(10):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        decoder.generate(hidden, code0, *params)
        end.record(); end.synchronize()
        eager_times.append(start.elapsed_time(end))
        start.record()
        decoder.generate(hidden, code0, *params, use_cuda_graph=True)
        end.record(); end.synchronize()
        graph_times.append(start.elapsed_time(end))
    print(f"depth decoder B={batch}: eager={statistics.median(eager_times):.3f}ms graph={statistics.median(graph_times):.3f}ms")
