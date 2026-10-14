"""HF Trainer subclass that surfaces the model's two loss components.

`compute_loss` returns exactly `outputs.loss` (the model already applies
`depth_header_loss_weight` at models/breeze.py:1858-1862 -- never
recompute/reweight it here). `log` merges the mean `backbone_loss` /
`depth_decoder_loss` into the logs dict and emits it through the
`breeze_train` logger, because `Trainer` prints its `{'loss': ...}` lines via
`print`/`tqdm.write`, not the `logging` module -- a bare `FileHandler` would
otherwise produce a `train.log` with no loss records at all, breaking every
gate that greps it (phases 6.4, 8.2, 9.2).

Training and evaluation forwards are accounted separately (`model.training`
decides which ledger a forward lands in): an evaluation in the middle of a
logging window must neither consume the training sums nor mix eval batches
into them. Training components are attached only to training-loss logs
(`'loss'` key); eval components are attached to the matching evaluation log
as `<prefix>_backbone_loss` / `<prefix>_depth_decoder_loss`, next to the
`<prefix>_loss` the Trainer computes (e.g. `eval_val_voices_loss`).

Under multi-GPU DDP the component sums are all-reduced before logging so the
logged values are the mean over every rank's micro-batches, matching how
`Trainer` gathers `loss`. Which ledger is reduced depends only on the logs
keys, which are identical on every rank, so the collectives stay in lockstep.

Training-loss logs also carry throughput (`step_time`, `samples_per_second`,
`eta_seconds`) measured between consecutive log calls, minus the time spent
in evaluations and audio sampling inside the window; the first window is
skipped because it includes dataloader start-up and CUDA warm-up. They also
carry `grad_norm/<module>` (e.g. `grad_norm/backbone_model`,
`grad_norm/depth_decoder`), the L2 norm of each top-level module's trainable
gradients on the optimizer step that ends the logging window. HF Trainer clips
before the `on_pre_optimizer_step` hook runs, so these are post-clip values:
identical to the raw norms whenever the logged total `grad_norm` is below
`max_grad_norm`, all scaled by the same factor otherwise (ratios between
modules are unaffected). Not computed under DeepSpeed/FSDP, where each rank
holds only a shard of the gradients.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Iterable

import torch
import transformers

from breeze_train.policy import top_module

logger = logging.getLogger("breeze_train")


def grad_norm_by_module(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    """L2 norm of the gradients of trainable parameters, grouped by top-level module."""
    squares: dict[str, float] = {}
    for name, param in named_parameters:
        if not param.requires_grad or param.grad is None:
            continue
        module = top_module(name)
        squares[module] = squares.get(module, 0.0) + param.grad.detach().float().pow(2).sum().item()
    return {module: math.sqrt(total) for module, total in squares.items()}


def eval_metric_prefix(logs: dict) -> str | None:
    """`{'eval_val_voices_loss': ..., ...}` -> `'eval_val_voices'`; None for non-eval logs."""
    for key in logs:
        if key.startswith("eval") and key.endswith("_loss") and not key.endswith(
            ("_backbone_loss", "_depth_decoder_loss")
        ):
            return key[: -len("_loss")]
    return None


class LossComponents:
    """Running sums of the model's backbone / depth-decoder losses for one ledger."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.backbone_sum = 0.0
        self.depth_decoder_sum = 0.0
        self.steps = 0

    def add(self, backbone_loss: float, depth_decoder_loss: float) -> None:
        self.backbone_sum += backbone_loss
        self.depth_decoder_sum += depth_decoder_loss
        self.steps += 1

    def pop_means(self, reduce: Callable[[list[float]], list[float]] = lambda x: x) -> dict[str, float]:
        """Means since the last pop (after `reduce`, e.g. a cross-rank sum), then reset."""
        backbone_sum, depth_decoder_sum, steps = reduce(
            [self.backbone_sum, self.depth_decoder_sum, float(self.steps)]
        )
        self.reset()
        if steps <= 0:
            return {}
        return {"backbone_loss": backbone_sum / steps, "depth_decoder_loss": depth_decoder_sum / steps}


class _GradNormProbe(transformers.TrainerCallback):
    """Measures per-module gradient norms on the optimizer step that closes a logging window."""

    def __init__(self, trainer: "BreezeTrainer") -> None:
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled:
            return
        # global_step is incremented right after this optimizer step; the
        # Trainer logs when the incremented value is a multiple of logging_steps.
        if args.logging_steps and (state.global_step + 1) % int(args.logging_steps) == 0:
            trainer._pending_grad_norms = grad_norm_by_module(trainer.model.named_parameters())


class BreezeTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_components = LossComponents()
        self._eval_components = LossComponents()
        self._pending_grad_norms: dict[str, float] = {}
        self._last_log_time: float | None = None
        self._last_log_step = 0
        self._excluded_seconds = 0.0
        self.add_callback(_GradNormProbe(self))

    def exclude_from_throughput(self, seconds: float) -> None:
        """Wall time spent outside training steps (eval, audio sampling) in the current window."""
        self._excluded_seconds += seconds

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        outputs = model(**inputs)

        backbone_loss = outputs.backbone_loss
        depth_decoder_loss = outputs.depth_decoder_loss
        if backbone_loss is not None and depth_decoder_loss is not None:
            ledger = self._train_components if model.training else self._eval_components
            ledger.add(backbone_loss.detach().float().item(), depth_decoder_loss.detach().float().item())
        # else: a batch with no target frames yields depth_decoder_loss is
        # None; skip accumulation rather than crashing.

        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

    def _reduce_sum(self, values: list[float]) -> list[float]:
        totals = torch.tensor(values, dtype=torch.float64, device=self.args.device)
        return self.accelerator.reduce(totals, reduction="sum").tolist()

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs:
            logs.update(self._train_components.pop_means(self._reduce_sum))
            for module, norm in self._pending_grad_norms.items():
                logs[f"grad_norm/{module}"] = norm
            self._pending_grad_norms = {}

            now = time.perf_counter()
            step = self.state.global_step
            if self._last_log_time is not None and step > self._last_log_step:
                train_seconds = now - self._last_log_time - self._excluded_seconds
                step_time = train_seconds / (step - self._last_log_step)
                global_batch = (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                    * self.args.world_size
                )
                logs["step_time"] = step_time
                logs["samples_per_second"] = global_batch / step_time
                logs["eta_seconds"] = max(self.state.max_steps - step, 0) * step_time
            self._last_log_time = now
            self._last_log_step = step
            self._excluded_seconds = 0.0

        prefix = eval_metric_prefix(logs)
        if prefix is not None:
            for key, value in self._eval_components.pop_means(self._reduce_sum).items():
                logs[f"{prefix}_{key}"] = value
            self.exclude_from_throughput(logs.get(f"{prefix}_runtime", 0.0))

        if self.is_world_process_zero():
            # `step` goes into the train.log record only (Trainer.log adds its own to
            # log_history), so readers never have to infer it from `epoch`.
            logger.info(str({**logs, "step": self.state.global_step}))
        super().log(logs, *args, **kwargs)
