"""Micro-benchmark of the Breeze depth-decoder loop (15 forwards per frame).

Two subcommands:

  capture  Runs one eager synthesis through the full model and saves the real
           (backbone hidden state, code0) pairs that reach the depth decoder,
           so the benchmark measures numerics on realistic inputs:
             python scripts/bench_depth_loop.py capture ../breeze-tts-2 \
               --out outputs/bench_depth/inputs.pt

  bench    Loads only the depth-decoder weights and times the whole 15-step
           loop (CUDA graph replay) for several implementations and lane
           counts, then compares each variant's teacher-forced logits with the
           serving implementation:
             python scripts/bench_depth_loop.py bench ../breeze-tts-2 \
               --inputs outputs/bench_depth/inputs.pt

Variants
  serving   serving/sglang_omni/breeze_omni/components/depth_decoder.py as is
  ref       this file's static-shape port of `serving` (separate q/k/v,
            gate/up), used as the numerical reference for the others
  fused     q|k|v and gate|up concatenated into one GEMM each
  compiled  fused + torch.compile of the per-layer step
  int8      compiled, layer linears int8 weight-only (per-output-channel scale)
  fp8       compiled, layer linears fp8 e4m3 via torch._scaled_mm with unit
            tensor-wise scales; the dynamic per-row activation scale and the
            per-channel weight scale are applied to the fp32 product
  fp8_mlp   fp8 for gate|up and down only (about 90% of the layer bytes),
            attention projections stay bf16
  fp8_rowwise  same math through _scaled_mm's native row-wise scaling. On an
            RTX 4070 Ti (sm_89) with torch 2.9.1 a CUDA graph replaying it
            intermittently hangs in cuda.synchronize (seen at 8 and 12 lanes),
            so it is excluded from the default variant list.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = REPO / "serving" / "sglang_omni" / "breeze_omni" / "components"
VOCAB = 2051
CODEBOOK = 2048
NUM_CB = 16


def _load_serving_components():
    """Import depth_decoder.py and sampling.py without the package __init__,
    which pulls in Transformers-5-only serving modules."""
    package = types.ModuleType("bz_components")
    package.__path__ = [str(COMPONENTS)]
    sys.modules["bz_components"] = package
    modules = {}
    for name in ("sampling", "depth_decoder"):
        spec = importlib.util.spec_from_file_location(f"bz_components.{name}", COMPONENTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["depth_decoder"], modules["sampling"]


def load_serving_depth(ckpt: Path, lanes: int, device: str):
    depth_module, sampling_module = _load_serving_components()
    cfg = json.loads((ckpt / "config.json").read_text())["depth_decoder_config"]
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "depth_decoder."
    files = sorted({f for k, f in index.items() if k.startswith(prefix)})
    weights = []
    for file in files:
        with safe_open(str(ckpt / file), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 (safe_open is not iterable)
                if key.startswith(prefix):
                    weights.append((key[len(prefix):], handle.get_tensor(key)))
    depth = depth_module.BreezeDepthDecoder(cfg, max_rows=lanes).to(device=device, dtype=torch.bfloat16)
    depth.load_weights(weights)
    return depth.eval(), sampling_module.sample


# --------------------------------------------------------------------------- linears


class Linear:
    """One linear layer in bf16, int8 weight-only, or fp8 (_scaled_mm)."""

    def __init__(self, weight: torch.Tensor, mode: str):
        self.mode = mode
        self.out_features = weight.shape[0]
        if mode == "bf16":
            self.w = weight.contiguous()
        elif mode == "int8":
            w = weight.float()
            scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            self.w = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
            self.scale = scale.to(torch.bfloat16)
        elif mode in ("fp8", "fp8_rowwise"):
            w = weight.float()
            scale = w.abs().amax(dim=1).clamp_min(1e-8) / 448.0
            # _scaled_mm wants mat2 column-major: store [out,in] row-major and pass .t().
            self.w = (w / scale[:, None]).to(torch.float8_e4m3fn).contiguous()
            self.scale = scale[None, :].contiguous()  # [1, N] fp32
            self.one = torch.ones((), device=weight.device)
        else:
            raise ValueError(mode)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bf16":
            return F.linear(x, self.w)
        if self.mode == "int8":
            return F.linear(x, self.w.to(x.dtype)) * self.scale
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        a_scale = x2.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448.0
        xq = (x2 / a_scale).to(torch.float8_e4m3fn)
        if self.mode == "fp8_rowwise":
            # Native row-wise scaling. On sm_89 / torch 2.9.1 a CUDA graph that
            # replays this kernel intermittently never completes (see module doc).
            out = torch._scaled_mm(
                xq, self.w.t(), scale_a=a_scale, scale_b=self.scale, out_dtype=torch.bfloat16
            )
        else:
            # Tensor-wise unit scales (the cuBLASLt path, graph-safe here), then
            # the same per-row x per-column scaling applied to the fp32 product.
            out = torch._scaled_mm(
                xq, self.w.t(), scale_a=self.one, scale_b=self.one, out_dtype=torch.float32
            )
            out = (out * a_scale * self.scale).to(torch.bfloat16)
        return out.reshape(*shape[:-1], self.out_features)


# --------------------------------------------------------------------------- static-shape loop


class DepthLoop:
    """Static-shape depth loop (same math as the serving component).

    The KV cache always has 16 slots and attention masks the unwritten ones, so
    every step has identical shapes: torch.compile builds one kernel set per
    lane count and the whole loop captures into one CUDA graph.
    """

    def __init__(self, serving, sample_fn, *, fused: bool, compiled: bool, quant: str, quant_attn: str | None = None):
        m = serving.model
        self.sample = sample_fn
        self.n_heads = serving.cfg["num_attention_heads"]
        self.n_kv = serving.n_kv_heads
        self.head_dim = serving.head_dim
        self.hidden = serving.hidden_size
        self.eps = serving.cfg.get("rms_norm_eps", 1e-6)
        self.fused = fused
        self.embed = m.embed_tokens.weight
        self.in_proj = m.inputs_embeds_projector.weight
        self.norm_w = m.norm.weight
        self.head = serving.codebooks_head.weight  # [15, 1024, 2051]
        self.cos = serving._rope_cos.to(torch.bfloat16)  # [16, D]
        self.sin = serving._rope_sin.to(torch.bfloat16)
        self.layers = []
        for layer in m.layers:
            a, p = layer.self_attn, layer.mlp
            entry = {
                "ln1": layer.input_layernorm.weight,
                "ln2": layer.post_attention_layernorm.weight,
            }
            qa = quant_attn or quant
            if fused:
                entry["qkv"] = Linear(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]), qa)
                entry["gate_up"] = Linear(torch.cat([p.gate_proj.weight, p.up_proj.weight]), quant)
            else:
                for name in ("q_proj", "k_proj", "v_proj"):
                    entry[name] = Linear(getattr(a, name).weight, qa)
                entry["gate"] = Linear(p.gate_proj.weight, quant)
                entry["up"] = Linear(p.up_proj.weight, quant)
            entry["o"] = Linear(a.o_proj.weight, qa)
            entry["down"] = Linear(p.down_proj.weight, quant)
            self.layers.append(entry)
        # One compiled callable per layer: its weights are closure constants, and
        # the cache holds one entry per (lanes, S in {1, 2}) shape.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 1024)
        torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 8192)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 1024)
        self.steps = []
        for lw in self.layers:
            step = functools.partial(self._layer_step, lw)
            self.steps.append(torch.compile(step, fullgraph=True, dynamic=False) if compiled else step)

    def _rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * w.float()).to(x.dtype)

    @staticmethod
    def _rotate(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _layer_step(self, lw, x, kc, vc, pos, cos, sin, mask):
        """x [L,S,H]; kc/vc [L,KV,16,D]; pos [S] slots to write; mask [S,16]."""
        L, S, _ = x.shape
        H, KV, D = self.n_heads, self.n_kv, self.head_dim
        residual = x
        h = self._rms(x, lw["ln1"])
        if self.fused:
            qkv = lw["qkv"](h)
            q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
        else:
            q, k, v = lw["q_proj"](h), lw["k_proj"](h), lw["v_proj"](h)
        q = q.view(L, S, H, D)
        k = k.view(L, S, KV, D)
        v = v.view(L, S, KV, D)
        c, s = cos[:, None, :], sin[:, None, :]  # [S,1,D]
        q = q * c + self._rotate(q) * s
        k = k * c + self._rotate(k) * s
        kc.index_copy_(2, pos, k.transpose(1, 2))
        vc.index_copy_(2, pos, v.transpose(1, 2))
        G = H // KV
        qg = q.view(L, S, KV, G, D).permute(0, 2, 3, 1, 4)  # [L,KV,G,S,D]
        scores = torch.einsum("lkgsd,lktd->lkgst", qg.float(), kc.float()) / math.sqrt(D)
        probs = torch.softmax(scores + mask, dim=-1).to(x.dtype)
        attn = torch.einsum("lkgst,lktd->lkgsd", probs, vc)  # [L,KV,G,S,D]
        attn = attn.permute(0, 3, 1, 2, 4).reshape(L, S, H * D)
        x = residual + lw["o"](attn)
        h = self._rms(x, lw["ln2"])
        if self.fused:
            gate, up = lw["gate_up"](h).chunk(2, dim=-1)
        else:
            gate, up = lw["gate"](h), lw["up"](h)
        return x + lw["down"](F.silu(gate) * up)

    def _forward(self, x, kcs, vcs, pos, mask):
        cos, sin = self.cos[pos], self.sin[pos]
        for step, kc, vc in zip(self.steps, kcs, vcs):
            x = step(x, kc, vc, pos, cos, sin, mask)
        return self._rms(x, self.norm_w)

    def make_state(self, lanes, device):
        kcs = [torch.zeros(lanes, self.n_kv, NUM_CB, self.head_dim, device=device, dtype=torch.bfloat16) for _ in self.layers]
        vcs = [torch.zeros_like(kcs[0]) for _ in self.layers]
        slots = torch.arange(NUM_CB, device=device)
        masks = [
            torch.where(slots[None, :] <= torch.arange(2, device=device)[:, None], 0.0, float("-inf")),
        ] + [
            torch.where(slots[None, :] <= p, 0.0, float("-inf")) for p in range(2, NUM_CB)
        ]
        positions = [torch.tensor([0, 1], device=device)] + [torch.tensor([p], device=device) for p in range(2, NUM_CB)]
        return kcs, vcs, masks, positions

    def run(self, state, hidden, code0, temperature, top_k, top_p, forced=None, logits_out=None):
        kcs, vcs, masks, positions = state
        lanes = hidden.shape[0]
        first = F.linear(hidden, self.in_proj)
        c0 = F.linear(self.embed[code0], self.in_proj)
        x = torch.stack((first, c0), dim=1)
        h = self._forward(x, kcs, vcs, positions[0], masks[0])[:, 1]
        index = torch.arange(lanes, device=hidden.device)
        outputs = torch.empty((lanes, NUM_CB - 1), dtype=torch.long, device=hidden.device)
        for q in range(1, NUM_CB):
            logits = (h @ self.head[q - 1]).float()
            # Same CFG arithmetic as serving (identity partner for this benchmark).
            uncond = logits.index_select(0, index)
            guided = uncond + (logits - uncond)
            guided[:, CODEBOOK:VOCAB] = float("-inf")
            if logits_out is not None:
                logits_out[:, q - 1] = guided
            tokens = self.sample(guided, temperature, top_k, top_p).index_select(0, index)
            outputs[:, q - 1] = tokens
            if q < NUM_CB - 1:
                feed = tokens if forced is None else forced[:, q - 1]
                e = F.linear(self.embed[feed + q * VOCAB], self.in_proj)
                h = self._forward(e[:, None], kcs, vcs, positions[q], masks[q])[:, 0]
        return outputs


# --------------------------------------------------------------------------- capture


def cmd_capture(args):
    sys.path.insert(0, str(REPO))
    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import (
        get_template,
        prepare_inputs,
        select_template_name,
    )
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    tokenizer, model, audio_tokenizer = load_runtime(args.model, device=resolve_device(), attn_implementation="eager")
    update_generation_config_for_breeze(model)
    hiddens, codes0 = [], []

    from models.cudagraph.depth_decoder_graph import DepthDecoderGraph

    original_run = DepthDecoderGraph.run

    def recording_run(self, backbone_hidden, first_cb_token, *rest, **kwargs):
        hiddens.append(backbone_hidden.detach().to("cpu", torch.bfloat16).reshape(-1, backbone_hidden.shape[-1]))
        codes0.append(first_cb_token.detach().to("cpu").reshape(-1))
        return original_run(self, backbone_hidden, first_cb_token, *rest, **kwargs)

    DepthDecoderGraph.run = recording_run
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer,
        FastStreamingConfig(max_new_tokens=args.max_frames, max_seq_len=2048, repetition_penalty=1.1),
        tokenizer=tokenizer,
    )
    for i, text in enumerate(args.text):
        request = {"id": f"cap{i}", "text": text, "speaker": "S0"}
        set_all_seeds(args.seed + i)
        inputs = prepare_inputs(
            tokenizer, audio_tokenizer, model, [request], get_template(select_template_name(request)),
            guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None,
        )
        for _ in runtime.iter_audio_chunks(inputs, request_id=f"cap{i}", seed=args.seed + i):
            pass
    if not hiddens:
        raise RuntimeError("no depth-decoder calls with backbone_last_hidden_state were observed")
    hidden = torch.cat(hiddens)
    code0 = torch.cat(codes0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": hidden, "code0": code0, "texts": args.text}, args.out)
    print(f"captured {hidden.shape[0]} frames -> {args.out}")


# --------------------------------------------------------------------------- bench


def time_graph(fn, iters: int, warmup: int = 3) -> tuple[float, float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)]


def cmd_bench(args):
    device = "cuda"
    torch.manual_seed(0)
    data = torch.load(args.inputs)
    real_hidden = data["hidden"].to(device)
    real_code0 = data["code0"].to(device)
    lanes_list = [int(x) for x in args.lanes.split(",")]
    variants = args.variants.split(",")
    max_lanes = max(lanes_list)
    serving, sample = load_serving_depth(args.model, max_lanes, device)
    layer_params = sum(p.numel() for p in serving.model.layers.parameters())
    mlp_params = sum(p.numel() for layer in serving.model.layers for p in layer.mlp.parameters())
    head_step = serving.codebooks_head.weight[0].numel()
    proj = serving.model.inputs_embeds_projector.weight.numel()
    print(f"GPU {torch.cuda.get_device_name()} · torch {torch.__version__}")
    print(f"depth layer params {layer_params/1e6:.1f}M · frames captured {real_hidden.shape[0]}")

    loops = {}
    specs = {
        "ref": {"fused": False, "compiled": False, "quant": "bf16"},
        "fused": {"fused": True, "compiled": False, "quant": "bf16"},
        "compiled": {"fused": True, "compiled": True, "quant": "bf16"},
        "int8": {"fused": True, "compiled": True, "quant": "int8"},
        "fp8": {"fused": True, "compiled": True, "quant": "fp8"},
        "fp8_mlp": {"fused": True, "compiled": True, "quant": "fp8", "quant_attn": "bf16"},
        "fp8_rowwise": {"fused": True, "compiled": True, "quant": "fp8_rowwise"},
    }
    weight_bytes = {"bf16": 2, "int8": 1, "fp8": 1, "fp8_rowwise": 1}
    if args.sampler == "argmax":
        sample = lambda logits, *_: logits.argmax(dim=-1)
    results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": [], "accuracy": {}}

    def inputs_for(lanes):
        idx = torch.arange(lanes, device=device) % real_hidden.shape[0]
        return (
            real_hidden[idx].contiguous(),
            real_code0[idx].contiguous(),
            torch.full((lanes,), args.temperature, device=device),
            torch.full((lanes,), args.top_k, device=device, dtype=torch.long),
            torch.ones(lanes, device=device),
        )

    for variant in variants:
        if variant != "serving":
            loops[variant] = DepthLoop(serving, sample, **specs[variant])
        for lanes in lanes_list:
            hidden, code0, temp, top_k, top_p = inputs_for(lanes)
            if variant == "serving":
                index = torch.arange(lanes, device=device)
                lane_inputs = (hidden, code0, index, index.clone(), torch.ones(lanes, device=device), temp, top_k, top_p)
                fn = functools.partial(serving._run_lanes, *lane_inputs)
                quant = "bf16"
            else:
                loop = loops[variant]
                state = loop.make_state(lanes, device)
                fn = functools.partial(loop.run, state, hidden, code0, temp, top_k, top_p)
                quant = specs[variant]["quant"]
            quant_attn = specs.get(variant, {}).get("quant_attn") or quant
            t0 = time.perf_counter()
            with torch.inference_mode():
                median, p90 = time_graph(fn, args.iters)
            setup_s = time.perf_counter() - t0
            layer_bytes = mlp_params * weight_bytes[quant] + (layer_params - mlp_params) * weight_bytes[quant_attn]
            bytes_per_frame = 15 * (layer_bytes + head_step * 2 + proj * 2)
            row = {
                "variant": variant, "lanes": lanes, "ms_median": median, "ms_p90": p90,
                "gbps": bytes_per_frame / (median / 1e3) / 1e9, "setup_s": setup_s,
            }
            results["rows"].append(row)
            print(f"{variant:9s} lanes={lanes:3d}  {median:7.2f} ms (p90 {p90:6.2f})  "
                  f"weights {row['gbps']:6.1f} GB/s  setup {setup_s:5.1f}s", flush=True)
            torch.cuda.empty_cache()

    # Accuracy: teacher-force every variant with the serving-greedy codes and
    # compare per-step logits with `ref` on the captured frames.
    n = min(args.accuracy_frames, real_hidden.shape[0])
    hidden, code0 = real_hidden[:n].contiguous(), real_code0[:n].contiguous()
    greedy = (torch.zeros(n, device=device), torch.zeros(n, device=device, dtype=torch.long), torch.ones(n, device=device))
    with torch.inference_mode():
        index = torch.arange(n, device=device)
        serving_codes = serving._run_lanes(hidden, code0, index, index.clone(), torch.ones(n, device=device), *greedy)
        ref_loop = loops.get("ref") or DepthLoop(serving, sample, **specs["ref"])
        ref_logits = torch.empty(n, 15, VOCAB, device=device)
        ref_codes = ref_loop.run(ref_loop.make_state(n, device), hidden, code0, *greedy, forced=serving_codes, logits_out=ref_logits)
        results["accuracy"]["ref_vs_serving_greedy_match"] = float((ref_codes == serving_codes).float().mean())
        print(f"ref vs serving greedy code match: {results['accuracy']['ref_vs_serving_greedy_match']*100:.2f}%")
        valid = slice(0, CODEBOOK)
        p_ref = torch.softmax(ref_logits[..., valid] / 0.9, dim=-1)
        for variant, loop in loops.items():
            if variant == "ref":
                continue
            logits = torch.empty_like(ref_logits)
            loop.run(loop.make_state(n, device), hidden, code0, *greedy, forced=serving_codes, logits_out=logits)
            logp = torch.log_softmax(logits[..., valid] / 0.9, dim=-1)
            kl = (p_ref * (torch.log(p_ref.clamp_min(1e-30)) - logp)).sum(-1)  # [n,15]
            top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float()
            acc = {
                "top1_agree": float(top1.mean()),
                "top1_agree_per_cb": [round(float(x), 4) for x in top1.mean(0)],
                "kl_mean": float(kl.mean()),
                "kl_per_cb": [round(float(x), 5) for x in kl.mean(0)],
                "max_abs_logit_diff": float((logits[..., valid] - ref_logits[..., valid]).abs().max()),
            }
            results["accuracy"][variant] = acc
            print(f"{variant:9s} top1 agree {acc['top1_agree']*100:6.2f}%  KL(ref||var) {acc['kl_mean']:.5f}  "
                  f"max|Δlogit| {acc['max_abs_logit_diff']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    _log_to_wandb(args, results, variants, lanes_list)


def _log_to_wandb(args, results: dict, variants: list[str], lanes_list: list[int]) -> None:
    sys.path.insert(0, str(REPO))
    from breeze_train.tracking import init_run

    config = {k: v for k, v in vars(args).items() if not k.startswith("wandb_") and k != "func"}
    run = init_run(args, job_type="benchmark", config={"bench": "depth_loop", **{k: str(v) if isinstance(v, Path) else v for k, v in config.items()}})
    if run is None:
        return
    import wandb

    rows = results["rows"]
    timing = wandb.Table(columns=["variant", "lanes", "ms_median", "ms_p90", "weight_gbps"])
    for r in rows:
        timing.add_data(r["variant"], r["lanes"], r["ms_median"], r["ms_p90"], r["gbps"])
        run.summary[f"{r['variant']}/ms_lanes{r['lanes']}"] = r["ms_median"]
    by_variant = {v: {r["lanes"]: r["ms_median"] for r in rows if r["variant"] == v} for v in variants}
    present = [v for v in variants if by_variant[v]]
    chart = wandb.plot.line_series(
        xs=lanes_list,
        ys=[[by_variant[v].get(l, float("nan")) for l in lanes_list] for v in present],
        keys=present,
        title="Depth loop ms/frame vs lanes",
        xname="lanes",
    )
    accuracy = wandb.Table(columns=["variant", "top1_agree", "kl_mean", "max_abs_logit_diff"])
    for name, acc in results["accuracy"].items():
        if isinstance(acc, dict):
            accuracy.add_data(name, acc["top1_agree"], acc["kl_mean"], acc["max_abs_logit_diff"])
            run.summary[f"{name}/kl_mean"] = acc["kl_mean"]
            run.summary[f"{name}/top1_agree"] = acc["top1_agree"]
    run.log({"timing": timing, "ms_vs_lanes": chart, "accuracy": accuracy})
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("model", type=Path)
    cap.add_argument("--out", type=Path, default=REPO / "outputs/bench_depth/inputs.pt")
    cap.add_argument("--text", nargs="+", default=[
        "The quick brown fox jumps over the lazy dog, and then it runs far away into the forest.",
        "Hôm nay trời đẹp, chúng ta cùng nhau đi dạo quanh hồ Gươm nhé.",
    ])
    cap.add_argument("--max-frames", type=int, default=200)
    cap.add_argument("--seed", type=int, default=0)
    cap.set_defaults(func=cmd_capture)
    b = sub.add_parser("bench")
    b.add_argument("model", type=Path)
    b.add_argument("--inputs", type=Path, default=REPO / "outputs/bench_depth/inputs.pt")
    b.add_argument("--out", type=Path, default=REPO / "outputs/bench_depth/results.json")
    b.add_argument("--variants", default="serving,ref,fused,compiled,int8,fp8")
    b.add_argument("--lanes", default="1,2,4,8,12,16,32")
    b.add_argument("--iters", type=int, default=50)
    b.add_argument("--temperature", type=float, default=0.9)
    b.add_argument("--top-k", type=int, default=50)
    b.add_argument("--sampler", choices=("serving", "argmax"), default="serving",
                   help="argmax isolates the cost of serving's top-k/top-p sampler (variant 'serving' always uses its own)")
    b.add_argument("--accuracy-frames", type=int, default=128)
    b.set_defaults(func=cmd_bench)
    sys.path.insert(0, str(REPO))
    from breeze_train.tracking import add_wandb_args

    add_wandb_args(b)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
