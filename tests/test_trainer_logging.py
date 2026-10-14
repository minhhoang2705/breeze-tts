from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments

from breeze_train.trainer import BreezeTrainer, grad_norm_by_module


class _FixedLossModel(torch.nn.Module):
    """Returns the loss components passed in the batch, so tests control every value."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, backbone, depth):
        return SimpleNamespace(
            loss=self.weight.sum() + backbone + depth,
            backbone_loss=torch.tensor(backbone),
            depth_decoder_loss=torch.tensor(depth),
        )


@pytest.fixture
def trainer(tmp_path) -> BreezeTrainer:
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    return BreezeTrainer(model=_FixedLossModel(), args=args)


def _forward(trainer: BreezeTrainer, training: bool, backbone: float, depth: float) -> None:
    trainer.model.train(training)
    trainer.compute_loss(trainer.model, {"backbone": backbone, "depth": depth})


def test_eval_inside_a_logging_window_keeps_train_and_eval_components_apart(trainer) -> None:
    _forward(trainer, training=True, backbone=1.0, depth=10.0)
    _forward(trainer, training=True, backbone=3.0, depth=30.0)
    # An evaluation lands in the middle of the training logging window.
    _forward(trainer, training=False, backbone=100.0, depth=1000.0)
    trainer.log({"eval_val_loss": 1100.0, "eval_val_runtime": 0.5})
    _forward(trainer, training=True, backbone=5.0, depth=50.0)
    trainer.log({"loss": 33.0})

    eval_log, train_log = trainer.state.log_history[-2:]
    assert eval_log["eval_val_backbone_loss"] == pytest.approx(100.0)
    assert eval_log["eval_val_depth_decoder_loss"] == pytest.approx(1000.0)
    assert "backbone_loss" not in eval_log
    assert train_log["backbone_loss"] == pytest.approx(3.0)
    assert train_log["depth_decoder_loss"] == pytest.approx(30.0)


def test_each_eval_set_reports_only_its_own_batches(trainer) -> None:
    _forward(trainer, training=False, backbone=2.0, depth=20.0)
    trainer.log({"eval_sentences_loss": 22.0})
    _forward(trainer, training=False, backbone=4.0, depth=40.0)
    _forward(trainer, training=False, backbone=6.0, depth=60.0)
    trainer.log({"eval_voices_loss": 55.0})

    sentences, voices = trainer.state.log_history[-2:]
    assert sentences["eval_sentences_backbone_loss"] == pytest.approx(2.0)
    assert voices["eval_voices_backbone_loss"] == pytest.approx(5.0)
    assert voices["eval_voices_depth_decoder_loss"] == pytest.approx(50.0)


def test_grad_norm_by_module_groups_peft_names_and_skips_frozen() -> None:
    def param(grad: list[float], trainable: bool = True) -> torch.nn.Parameter:
        p = torch.nn.Parameter(torch.zeros(len(grad)), requires_grad=trainable)
        p.grad = torch.tensor(grad)
        return p

    named = [
        ("base_model.model.backbone_model.layers.0.q_proj.lora_A.default.weight", param([3.0])),
        ("base_model.model.backbone_model.layers.1.v_proj.lora_B.default.weight", param([4.0])),
        ("base_model.model.depth_decoder.layers.0.o_proj.lora_A.default.weight", param([1.0, 2.0, 2.0])),
        ("base_model.model.text_encoder.embed.weight", param([9.0], trainable=False)),
    ]

    norms = grad_norm_by_module(named)

    assert norms == {"backbone_model": pytest.approx(5.0), "depth_decoder": pytest.approx(3.0)}
