"""SGLang Omni runner hooks for Breeze frame feedback and depth decoding."""

from __future__ import annotations

from typing import Any

import torch
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import OmniPrefillInputs, attach_omni_prefill_inputs

from .request_builders import cfg_uncond_rid, is_cfg_uncond_rid


class BreezeTTSModelRunner(ModelRunner):
    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del schedule_batch
        rows = []
        for sched_req in requests:
            data = sched_req.data
            embeds = data.prefill_input_embeds
            if embeds is None:
                raise RuntimeError(f"Breeze prefill embeddings missing for {sched_req.request_id}")
            rows.append(embeds.to(device=self.model.frame_embedding.weight.device))
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if len(requests) > self.model.decode_feedback_buffer.shape[0]:
            raise RuntimeError("Breeze decode batch exceeds feedback buffer capacity")
        embeddings = []
        for sched_req in requests:
            data = sched_req.data
            if not data.generated_frames:
                raise RuntimeError(f"Breeze feedback frame missing for {sched_req.request_id}")
            frame = data.generated_frames[-1].reshape(1, 16).to(
                device=self.model.frame_embedding.weight.device, dtype=torch.long
            )
            embeddings.append(self.model.embed_frames(frame)[0])
        if embeddings:
            self.model.decode_feedback_buffer[:len(embeddings)].copy_(torch.stack(embeddings))

    def post_prefill(self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del schedule_batch
        self._decode_depth(result, forward_batch, requests)

    def post_decode(self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list) -> None:
        del schedule_batch
        self._decode_depth(result, forward_batch, requests)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, requests
        return not schedule_batch.is_prefill_only

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, requests
        return not schedule_batch.is_prefill_only

    def _sample_next_token_ids(
        self, logits_output: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        # Breeze order (fast_streaming.py:832-842 / backbone_graph.py:280, then
        # sampling.py:54-60): CFG on raw logits, then repetition penalty, then
        # codec-reserved suppression, then sampling.
        self._guide_cfg_pairs(logits_output.next_token_logits, requests)
        return super()._sample_next_token_ids(
            logits_output, forward_batch, schedule_batch, requests
        )

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        # SGLang's penalizer (SamplingParams.repetition_penalty) already applies the
        # HF-style penalty inside ModelRunner.sample; applying it here too would
        # penalize twice.
        del logits_output, requests

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        del requests
        logits = logits_output.next_token_logits
        logits[:, self.model.codec_codebook_size:self.model.audio_vocab_size] = float("-inf")

    @staticmethod
    def _guide_cfg_pairs(logits: torch.Tensor, requests: list) -> None:
        row_by_rid = {str(req.request_id): row for row, req in enumerate(requests)}
        cond_rows: list[int] = []
        uncond_rows: list[int] = []
        scales: list[float] = []
        for row, sched_req in enumerate(requests):
            data = sched_req.data
            if data.is_cfg_uncond or data.cfg_uncond is None:
                continue
            twin = row_by_rid.get(cfg_uncond_rid(str(sched_req.request_id)))
            if twin is None:
                raise RuntimeError(
                    f"Breeze CFG lane {sched_req.request_id} scheduled without its uncond twin"
                )
            cond_rows.append(row)
            uncond_rows.append(twin)
            scales.append(float(data.cfg_scale))
        if not cond_rows:
            return
        device = logits.device
        cond = torch.tensor(cond_rows, dtype=torch.long, device=device)
        uncond = torch.tensor(uncond_rows, dtype=torch.long, device=device)
        scale = torch.tensor(scales, dtype=logits.dtype, device=device)[:, None]
        uncond_logits = logits.index_select(0, uncond)
        guided = uncond_logits + scale * (logits.index_select(0, cond) - uncond_logits)
        logits[cond] = guided
        logits[uncond] = guided

    def _decode_depth(self, result: Any, forward_batch: Any, requests: list) -> None:
        if not requests or result.next_token_ids is None:
            return
        hidden = result.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("Breeze backbone did not return hidden states")
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        if hidden.ndim != 2 or hidden.shape[0] < len(requests):
            raise RuntimeError(f"invalid Breeze hidden state shape {tuple(hidden.shape)}")
        token_ids = result.next_token_ids.reshape(-1)
        if token_ids.numel() < len(requests):
            raise RuntimeError("Breeze sampler returned fewer code0 ids than active requests")
        row_by_rid = {str(req.request_id): index for index, req in enumerate(requests)}
        cond_rows: list[int] = []
        cond_requests: list[Any] = []
        for index, sched_req in enumerate(requests):
            if is_cfg_uncond_rid(str(sched_req.request_id)):
                continue
            cond_rows.append(index)
            cond_requests.append(sched_req)
        if not cond_rows:
            return
        device = hidden.device
        cond_idx = torch.tensor(cond_rows, dtype=torch.long, device=device)
        cond_hidden = hidden.index_select(0, cond_idx)
        code0 = token_ids.to(device=device, dtype=torch.long).index_select(0, cond_idx)
        cfg_pair_rows: list[int] = []
        cfg_hidden_rows: list[torch.Tensor] = []
        scales: list[float] = []
        for cond_pos, cond_request in enumerate(cond_requests):
            twin_row = row_by_rid.get(f"{cond_request.request_id}_uncond")
            if twin_row is not None:
                cfg_pair_rows.append(cond_pos)
                cfg_hidden_rows.append(hidden[twin_row])
                cfg_data = requests[twin_row].data
                scales.append(float(cfg_data.cfg_scale))
        state_params = [req.data.state.sampling for req in cond_requests]
        temperature = torch.tensor([float(p["depth_temperature"]) for p in state_params], device=device)
        top_k = torch.tensor([int(p["depth_top_k"]) for p in state_params], device=device)
        top_p = torch.tensor([float(p["depth_top_p"]) for p in state_params], device=device)
        codes = self.model.depth_decoder.generate(
            cond_hidden,
            code0,
            temperature,
            top_k,
            top_p,
            cfg_uncond_hidden=(torch.stack(cfg_hidden_rows) if cfg_hidden_rows else None),
            cfg_rows=(torch.tensor(cfg_pair_rows, dtype=torch.long, device=device) if cfg_pair_rows else None),
            cfg_scale=(torch.tensor(scales, dtype=torch.float32, device=device) if scales else None),
            use_cuda_graph=True,
        )
        eos = code0.eq(self.model.eos_token_id)
        frames = torch.cat((code0[:, None], codes), dim=1)
        frames = frames.masked_fill(eos[:, None], self.model.audio_vocab_size - 1)
        eos_host = eos.tolist()  # one device sync for the whole batch
        for cond_pos, sched_req in enumerate(cond_requests):
            twin_row = row_by_rid.get(f"{sched_req.request_id}_uncond")
            if twin_row is not None:
                token_ids[twin_row] = token_ids[cond_rows[cond_pos]]
            if eos_host[cond_pos]:
                continue
            frame = frames[cond_pos].detach()
            sched_req.data.generated_frames.append(frame)
            if twin_row is not None:
                twin = requests[twin_row]
                twin.data.generated_frames.append(frame)
        result.next_token_ids = token_ids.reshape_as(result.next_token_ids)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        if req_data.is_cfg_uncond:
            return
        state = req_data.state
        frames = req_data.generated_frames or []
        state.generated_codes = (
            torch.stack(frames).detach().cpu()
            if frames
            else torch.empty((0, 16), dtype=torch.long)
        )
