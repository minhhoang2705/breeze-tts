"""Batched incremental Breeze depth decoder, ported from models/breeze.py:463-628
and models/cudagraph/depth_decoder_graph.py. The transformer layer tensor names
and equations mirror BreezeAttention/DecoderLayer at models/breeze.py:289-425.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .sampling import sample


def _get(cfg: Any, name: str, default=None):
    return cfg.get(name, default) if isinstance(cfg, dict) else getattr(cfg, name, default)


class _RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class _MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, m = _get(cfg, "hidden_size"), _get(cfg, "intermediate_size")
        bias = _get(cfg, "mlp_bias", False)
        self.gate_proj = nn.Linear(h, m, bias=bias)
        self.up_proj = nn.Linear(h, m, bias=bias)
        self.down_proj = nn.Linear(m, h, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = _get(cfg, "hidden_size")
        self.head_dim = _get(cfg, "head_dim", h // _get(cfg, "num_attention_heads"))
        self.n_heads = _get(cfg, "num_attention_heads")
        self.n_kv_heads = _get(cfg, "num_key_value_heads")
        bias = _get(cfg, "attention_bias", False)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=bias)
        self.self_attn.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=bias)
        self.self_attn.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=bias)
        self.self_attn.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=bias)
        self.input_layernorm = _RMSNorm(h, _get(cfg, "rms_norm_eps", 1e-6))
        self.post_attention_layernorm = _RMSNorm(h, _get(cfg, "rms_norm_eps", 1e-6))
        self.mlp = _MLP(cfg)

    def forward_one(self, x, pos, key_cache, value_cache, cos, sin):
        residual = x
        x = self.input_layernorm(x)
        b = x.shape[0]
        q = self.self_attn.q_proj(x).view(b, self.n_heads, self.head_dim)
        k = self.self_attn.k_proj(x).view(b, self.n_kv_heads, self.head_dim)
        v = self.self_attn.v_proj(x).view(b, self.n_kv_heads, self.head_dim)
        q, k = _rotary(q, k, cos, sin)
        key_cache[:, :, pos, :] = k
        value_cache[:, :, pos, :] = v
        keys = key_cache[:, :, : pos + 1, :]
        values = value_cache[:, :, : pos + 1, :]
        attn = F.scaled_dot_product_attention(q.unsqueeze(2), keys, values, is_causal=False, enable_gqa=True)
        attn = attn.squeeze(2).reshape(b, self.n_heads * self.head_dim)
        x = residual + self.self_attn.o_proj(attn)
        return x + self.mlp(self.post_attention_layernorm(x))



    def forward_prefill(self, x, key_cache, value_cache, cos, sin):
        residual = x
        x = self.input_layernorm(x)
        batch, seq_len = x.shape[:2]
        q = self.self_attn.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _rotary(q, k, cos, sin)
        key_cache[:, :, :seq_len, :] = k
        value_cache[:, :, :seq_len, :] = v
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.n_heads * self.head_dim)
        x = residual + self.self_attn.o_proj(attn)
        return x + self.mlp(self.post_attention_layernorm(x))
def _rotary(q, k, cos, sin):
    # q [B,H,D], k [B,KV,D], rotary last dimension uses split-half convention.
    def rotate(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class BreezeDepthDecoder(nn.Module):
    def __init__(self, depth_cfg: Any, max_rows: int):
        super().__init__()
        self.cfg = depth_cfg
        self.max_rows = max_rows
        self.vocab_size = int(_get(depth_cfg, "vocab_size", 2051))
        self.codebook_size = 2048
        self.num_codebooks = int(_get(depth_cfg, "num_codebooks", 16))
        self.hidden_size = int(_get(depth_cfg, "hidden_size"))
        self.audio_embed_size = int(_get(depth_cfg, "audio_embed_size", _get(depth_cfg, "backbone_hidden_size", self.hidden_size)))
        self.head_dim = int(_get(depth_cfg, "head_dim", self.hidden_size // _get(depth_cfg, "num_attention_heads")))
        self.n_kv_heads = int(_get(depth_cfg, "num_key_value_heads"))
        self.num_layers = int(_get(depth_cfg, "num_hidden_layers"))
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(self.num_codebooks * self.vocab_size, self.audio_embed_size)
        self.model.inputs_embeds_projector = nn.Linear(self.audio_embed_size, self.hidden_size, bias=False)
        backbone_h = int(_get(depth_cfg, "backbone_hidden_size", self.audio_embed_size))
        self.model.backbone_hidden_state_projector = nn.Linear(backbone_h, self.audio_embed_size, bias=False) if backbone_h != self.audio_embed_size else None
        self.model.layers = nn.ModuleList([_Layer(depth_cfg) for _ in range(self.num_layers)])
        self.model.norm = _RMSNorm(self.hidden_size, _get(depth_cfg, "rms_norm_eps", 1e-6))
        self.codebooks_head = nn.Module()
        self.codebooks_head.weight = nn.Parameter(torch.empty(self.num_codebooks - 1, self.hidden_size, self.vocab_size))
        self.rope_theta = float(_get(depth_cfg, "rope_theta", 10000.0))
        self.rope_scaling = _get(depth_cfg, "rope_scaling", None) or {}
        inv = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        scale = self.rope_scaling
        if scale.get("rope_type", scale.get("type")) == "llama3":
            factor = float(scale["factor"])
            low, high = float(scale["low_freq_factor"]), float(scale["high_freq_factor"])
            old = float(scale["original_max_position_embeddings"])
            wavelength = 2 * math.pi / inv
            low_wavelen, high_wavelen = old / low, old / high
            smooth = (old / wavelength - low) / (high - low)
            scaled = inv / factor
            blend = (1 - smooth) * scaled + smooth * inv
            inv = torch.where(
                wavelength > low_wavelen,
                scaled,
                torch.where(wavelength < high_wavelen, inv, blend),
            )
        phase = torch.arange(self.num_codebooks, dtype=torch.float32)[:, None] * inv[None]
        phase = torch.cat((phase, phase), dim=-1)
        self.register_buffer("_rope_cos", phase.cos(), persistent=False)
        self.register_buffer("_rope_sin", phase.sin(), persistent=False)
        # One CUDA graph over a fixed max_rows-lane layout (see capture_graph).
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_inputs: tuple[torch.Tensor, ...] | None = None
        self._graph_output: torch.Tensor | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded = set()
        with torch.no_grad():
            for name, tensor in weights:
                if name not in params:
                    continue
                param = params[name]
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ValueError(f"Weight shape mismatch for {name}: expected {tuple(param.shape)}, got {tuple(tensor.shape)}")
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(f"Missing depth decoder weights: {missing[:12]}")

    def embed_frames(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"codes must have shape [N,{self.num_codebooks}]")
        offsets = torch.arange(self.num_codebooks, device=codes.device) * self.vocab_size
        return self.model.embed_tokens(codes.long() + offsets).sum(dim=1)

    def _rope(self, pos: int, dtype: torch.dtype, device: torch.device):
        return (
            self._rope_cos[pos].to(device=device, dtype=dtype).view(1, 1, self.head_dim),
            self._rope_sin[pos].to(device=device, dtype=dtype).view(1, 1, self.head_dim),
        )

    def _forward_token(self, x: torch.Tensor, pos: int, keys, values):
        cos, sin = self._rope(pos, x.dtype, x.device)
        for i, layer in enumerate(self.model.layers):
            x = layer.forward_one(x, pos, keys[i], values[i], cos, sin)
        return self.model.norm(x)


    def _forward_prefill(self, x: torch.Tensor, keys, values):
        cos = self._rope_cos[:2].to(device=x.device, dtype=x.dtype).view(1, 1, 2, self.head_dim)
        sin = self._rope_sin[:2].to(device=x.device, dtype=x.dtype).view(1, 1, 2, self.head_dim)
        for i, layer in enumerate(self.model.layers):
            x = layer.forward_prefill(x, keys[i], values[i], cos, sin)
        return self.model.norm(x)
    @torch.inference_mode()
    def generate(
        self,
        hidden: torch.Tensor,
        code0: torch.LongTensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        *,
        cfg_uncond_hidden: torch.Tensor | None = None,
        cfg_rows: torch.LongTensor | None = None,
        cfg_scale: torch.Tensor | None = None,
        use_cuda_graph: bool = False,
    ) -> torch.LongTensor:
        """Codebooks 1..15 for B frames; CFG rows guide every depth step.

        cfg_rows[j] is the conditional row that cfg_uncond_hidden[j] (its uncond
        lane's backbone hidden state) guides with cfg_scale[j].
        """
        batch = hidden.shape[0]
        if batch == 0:
            return torch.empty((0, self.num_codebooks - 1), dtype=torch.long, device=hidden.device)
        lanes = self._lane_inputs(
            hidden, code0, temperature, top_k, top_p, cfg_uncond_hidden, cfg_rows, cfg_scale
        )
        if use_cuda_graph and hidden.device.type == "cuda":
            if self._graph is None:
                self.capture_graph(hidden.device, hidden.dtype)
            assert self._graph_inputs is not None and self._graph_output is not None
            for target, source in zip(self._graph_inputs, lanes):
                target.copy_(source)
            self._graph.replay()
            codes = self._graph_output[:batch].clone()
        else:
            codes = self._run_lanes(*lanes)[:batch]
        if bool((codes >= self.codebook_size).any()):
            raise RuntimeError("depth decoder emitted a code outside codec cardinality 2048")
        return codes

    @torch.inference_mode()
    def capture_graph(self, device: torch.device | str, dtype: torch.dtype) -> None:
        """Capture the depth loop once, over all max_rows lanes.

        Every decode step uses the same lane layout (conditional rows, then CFG
        uncond lanes, then padding), so one graph serves any batch/CFG mix. The
        loop is weight-bandwidth bound, so idle padding lanes cost almost nothing,
        while per-shape graphs cost ~65 MB of device memory each. Capturing at
        startup also keeps live decode steps from stalling on a capture.
        """
        lanes = self.max_rows
        width = int(_get(self.cfg, "backbone_hidden_size", self.audio_embed_size))
        index = torch.arange(lanes, device=device)
        inputs = (
            torch.zeros((lanes, width), device=device, dtype=dtype),
            torch.zeros(lanes, device=device, dtype=torch.long),
            index.clone(),
            index.clone(),
            torch.ones(lanes, device=device),
            torch.full((lanes,), 0.9, device=device),
            torch.full((lanes,), 50, device=device, dtype=torch.long),
            torch.ones(lanes, device=device),
        )
        self._run_lanes(*inputs)
        graph = torch.cuda.CUDAGraph()
        # thread_local: the vocoder stage shares this process and GPU and keeps
        # launching work from its own thread while this graph is captured.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            output = self._run_lanes(*inputs)
        self._graph, self._graph_inputs, self._graph_output = graph, inputs, output

    def _lane_inputs(
        self,
        hidden: torch.Tensor,
        code0: torch.LongTensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        cfg_uncond_hidden: torch.Tensor | None,
        cfg_rows: torch.LongTensor | None,
        cfg_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        batch, width = hidden.shape
        device = hidden.device
        rows = (
            torch.empty(0, dtype=torch.long, device=device)
            if cfg_rows is None
            else cfg_rows.to(device=device, dtype=torch.long).reshape(-1)
        )
        n_cfg = rows.numel()
        if (cfg_uncond_hidden is None) != (n_cfg == 0):
            raise ValueError("cfg_rows and cfg_uncond_hidden must be given together")
        if cfg_uncond_hidden is not None and cfg_uncond_hidden.shape != (n_cfg, width):
            raise ValueError("CFG hidden states must be [C,H] and cfg_rows [C]")
        if width != int(_get(self.cfg, "backbone_hidden_size", self.audio_embed_size)):
            raise ValueError("hidden width does not match the backbone")
        if code0.shape != (batch,):
            raise ValueError("code0 must have shape [B]")
        if batch + n_cfg > self.max_rows:
            raise ValueError(f"depth decoder needs {batch + n_cfg} lanes, max_rows={self.max_rows}")

        def per_row(value: Any, count: int, dtype: torch.dtype) -> torch.Tensor:
            tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
            if tensor.numel() == 1:
                return tensor.expand(count)
            if tensor.numel() != count:
                raise ValueError("sampling and CFG parameters must be scalar or one per row")
            return tensor

        lanes = self.max_rows
        index = torch.arange(lanes, device=device)
        hidden_l = torch.zeros((lanes, width), dtype=hidden.dtype, device=device)
        hidden_l[:batch] = hidden
        code0_l = torch.zeros(lanes, dtype=torch.long, device=device)
        code0_l[:batch] = code0
        partner = index.clone()  # lane whose logits are the "uncond" side
        owner = index.clone()  # lane whose sampled token this lane feeds back
        scale = torch.ones(lanes, device=device)
        # Padding and uncond lanes decode greedily; their samples are discarded.
        temps = torch.zeros(lanes, device=device)
        ks = torch.zeros(lanes, dtype=torch.long, device=device)
        ps = torch.ones(lanes, device=device)
        temps[:batch] = per_row(temperature, batch, torch.float32)
        ks[:batch] = per_row(top_k, batch, torch.long)
        ps[:batch] = per_row(top_p, batch, torch.float32)
        if n_cfg:
            uncond = batch + torch.arange(n_cfg, device=device)
            hidden_l[batch:batch + n_cfg] = cfg_uncond_hidden.to(dtype=hidden.dtype)
            code0_l[uncond] = code0[rows]
            partner[rows] = uncond
            owner[uncond] = rows
            scale[rows] = per_row(cfg_scale, n_cfg, torch.float32)
        return hidden_l, code0_l, partner, owner, scale, temps, ks, ps

    def _run_lanes(
        self,
        hidden: torch.Tensor,
        code0: torch.LongTensor,
        partner: torch.LongTensor,
        owner: torch.LongTensor,
        scale: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.LongTensor,
        top_p: torch.Tensor,
        forced: torch.LongTensor | None = None,
    ) -> torch.LongTensor:
        """Static-shape depth loop over all lanes; safe to capture in a CUDA graph.

        With ``forced`` [lanes, 15], each step still returns its own prediction but
        feeds the forced code to the next step (teacher forcing, for parity tests).
        """
        lanes = hidden.shape[0]
        if self.model.backbone_hidden_state_projector is not None:
            first = self.model.backbone_hidden_state_projector(hidden)
        else:
            first = hidden
        first = self.model.inputs_embeds_projector(first)
        code0_emb = self.model.inputs_embeds_projector(self.model.embed_tokens(code0))
        keys = torch.empty(
            (self.num_layers, lanes, self.n_kv_heads, self.num_codebooks, self.head_dim),
            dtype=hidden.dtype, device=hidden.device,
        )
        values = torch.empty_like(keys)
        h = self._forward_prefill(torch.stack((first, code0_emb), dim=1), keys, values)[:, 1]
        outputs = torch.empty((lanes, self.num_codebooks - 1), dtype=torch.long, device=hidden.device)
        for q in range(1, self.num_codebooks):
            logits = torch.einsum("bh,hv->bv", h, self.codebooks_head.weight[q - 1]).float()
            # CFG on raw logits (uncond + s * (cond - uncond)); unpaired lanes have
            # partner == self and s == 1, which leaves their logits unchanged.
            uncond = logits.index_select(0, partner)
            guided = uncond + scale[:, None] * (logits - uncond)
            guided[:, self.codebook_size:self.vocab_size] = float("-inf")
            tokens = sample(guided, temperature, top_k, top_p).index_select(0, owner)
            outputs[:, q - 1] = tokens
            if q < self.num_codebooks - 1:
                feed = tokens if forced is None else forced[:, q - 1]
                embedded = self.model.inputs_embeds_projector(
                    self.model.embed_tokens(feed + q * self.vocab_size)
                )
                h = self._forward_token(embedded, q + 1, keys, values)
        return outputs
