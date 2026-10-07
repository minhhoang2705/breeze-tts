"""CFG pair admission in BreezeScheduler (no GPU, SGLang admission faked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from breeze_omni.scheduler import BreezeScheduler


def _req(rid: str, *, twin: bool = False, uncond: bool = False, tokens: int = 10):
    data = SimpleNamespace(is_cfg_uncond=uncond, cfg_uncond=object() if twin else None)
    return SimpleNamespace(rid=rid, _omni_data=data, origin_input_ids=[0] * tokens)


@pytest.fixture
def scheduler(monkeypatch):
    sched = object.__new__(BreezeScheduler)
    sched.max_prefill_tokens = 8192
    sched.admitted = []

    def fake_prefill(self, running_batch):
        # Mirrors SGLang: admit what it is shown, then *rebind* waiting_queue.
        self.admitted.extend(r.rid for r in self.waiting_queue)
        self.waiting_queue = []

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", fake_prefill)
    return sched


def test_deferred_requests_survive_admission(scheduler) -> None:
    # Three free lanes: "a" plus the "b" pair fit, the "c" pair must wait.
    scheduler.get_num_allocatable_reqs = lambda running: 3
    scheduler.waiting_queue = [
        _req("a"),
        _req("b", twin=True), _req("b_uncond", uncond=True),
        _req("c", twin=True), _req("c_uncond", uncond=True),
    ]
    scheduler.get_new_batch_prefill(SimpleNamespace(reqs=[]))
    assert scheduler.admitted == ["a", "b", "b_uncond"]
    assert [r.rid for r in scheduler.waiting_queue] == ["c", "c_uncond"]


def test_pair_is_never_split_across_admissions(scheduler) -> None:
    scheduler.get_num_allocatable_reqs = lambda running: 2
    scheduler.waiting_queue = [_req("a"), _req("b", twin=True), _req("b_uncond", uncond=True)]
    scheduler.get_new_batch_prefill(SimpleNamespace(reqs=[]))
    assert scheduler.admitted == ["a"]
    assert [r.rid for r in scheduler.waiting_queue] == ["b", "b_uncond"]
