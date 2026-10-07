"""OmniScheduler specialization that admits Breeze CFG pairs atomically."""

from __future__ import annotations

from typing import Any

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .request_builders import cfg_uncond_rid, is_cfg_uncond_rid


class BreezeScheduler(OmniScheduler):
    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        # The uncond twin must sit right behind its cond request in waiting_queue.
        # Request builders run on several threads, so hold the (reentrant)
        # admission lock across both appends; otherwise another request can land
        # between them, the twin is never queued and the cond request stalls.
        with self._request_admission_lock:
            super()._enqueue_built_request(
                payload, pending_stream_done, req_data,
                request_admission_lock_held=True,
            )
            twin_data = getattr(req_data, "cfg_uncond", None)
            if twin_data is not None:
                self._enqueue_cfg_uncond(req_data, twin_data)

    def _enqueue_cfg_uncond(self, cond_data: Any, twin_data: Any) -> None:
        cond_req = cond_data.req
        if not self.waiting_queue or self.waiting_queue[-1] is not cond_req:
            return  # cond was rejected or aborted before it was queued
        twin_req = twin_data.req
        self._normalize_req_token_arrays(twin_req)
        twin_req._coalesce_enqueue_t = cond_req._coalesce_enqueue_t
        twin_req._omni_terminal_claimed = False
        twin_req._omni_data = twin_data
        self.waiting_queue.append(twin_req)

    def get_new_batch_prefill(self, running_batch: Any) -> Any:
        queue = self.waiting_queue
        limit = self._pair_admission_limit(queue, running_batch)
        if limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            # SGLang rebinds self.waiting_queue to a new list when it admits
            # requests; re-queue onto the live attribute, not the local alias,
            # or the deferred requests are silently dropped.
            self.waiting_queue.extend(deferred)

    def _pair_admission_limit(self, queue: list, running_batch: Any) -> int:
        allocatable = max(0, int(self.get_num_allocatable_reqs(len(running_batch.reqs))))
        budget = int(self.max_prefill_tokens)
        accepted = 0
        tokens = 0
        index = 0
        while index < len(queue):
            data = getattr(queue[index], "_omni_data", None)
            is_twin = bool(getattr(data, "is_cfg_uncond", False))
            group = 1
            if not is_twin and getattr(data, "cfg_uncond", None) is not None:
                group = 2
            if accepted + group > allocatable or index + group > len(queue):
                break
            group_tokens = sum(len(req.origin_input_ids) for req in queue[index:index + group])
            if index and tokens + group_tokens > budget:
                break
            tokens += group_tokens
            accepted += group
            index += group
        return index

    def stream_output(self, reqs: Any, return_logprob: bool = False, skip_req: Any = None) -> None:
        visible = []
        for req in reqs:
            if not self._is_cfg_uncond(req):
                visible.append(req)
            elif req.finished():
                self._close_completed_request(req)
        super().stream_output(visible, return_logprob, skip_req)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if is_cfg_uncond_rid(request_id):
            return
        super().abort(
            cfg_uncond_rid(request_id), defer_running_cleanup=defer_running_cleanup
        )

    @staticmethod
    def _is_cfg_uncond(req: Any) -> bool:
        data = getattr(req, "_omni_data", None)
        return data is not None and bool(getattr(data, "is_cfg_uncond", False))
