"""HF Trainer subclass that surfaces the model's two loss components.

`compute_loss` returns exactly `outputs.loss` (the model already applies
`depth_header_loss_weight` at models/breeze.py:1858-1862 -- never
recompute/reweight it here). `log` merges the mean `backbone_loss` /
`depth_decoder_loss` since the previous log call into the logs dict and
emits it through the `breeze_train` logger, because `Trainer` prints its
`{'loss': ...}` lines via `print`/`tqdm.write`, not the `logging` module --
a bare `FileHandler` would otherwise produce a `train.log` with no loss
records at all, breaking every gate that greps it (phases 6.4, 8.2, 9.2).

Under multi-GPU DDP the component sums are all-reduced before logging so the
logged values are the mean over every rank's micro-batches, matching how
`Trainer` gathers `loss`. `Trainer.log` runs on every rank, so the reduce is
issued unconditionally to keep the collective in lockstep.

Training-loss logs also carry throughput (`step_time`, `samples_per_second`,
`eta_seconds`) measured between consecutive log calls; the first window is
skipped because it includes dataloader start-up and CUDA warm-up.
"""

from __future__ import annotations

import logging
import time

import torch
import transformers

logger = logging.getLogger("breeze_train")


class BreezeTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._backbone_loss_sum = 0.0
        self._depth_decoder_loss_sum = 0.0
        self._loss_component_steps = 0
        self._last_log_time: float | None = None
        self._last_log_step = 0

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        outputs = model(**inputs)

        backbone_loss = outputs.backbone_loss
        depth_decoder_loss = outputs.depth_decoder_loss
        if backbone_loss is not None and depth_decoder_loss is not None:
            self._backbone_loss_sum += backbone_loss.detach().float().item()
            self._depth_decoder_loss_sum += depth_decoder_loss.detach().float().item()
            self._loss_component_steps += 1
        # else: a batch with no target frames yields depth_decoder_loss is
        # None (e.g. during an eval batch that happens to carry no labels);
        # skip accumulation rather than crashing.

        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        totals = torch.tensor(
            [self._backbone_loss_sum, self._depth_decoder_loss_sum, self._loss_component_steps],
            dtype=torch.float64,
            device=self.args.device,
        )
        totals = self.accelerator.reduce(totals, reduction="sum")
        backbone_sum, depth_decoder_sum, steps = totals.tolist()
        if steps > 0:
            logs["backbone_loss"] = backbone_sum / steps
            logs["depth_decoder_loss"] = depth_decoder_sum / steps
        self._backbone_loss_sum = 0.0
        self._depth_decoder_loss_sum = 0.0
        self._loss_component_steps = 0

        if "loss" in logs:
            now = time.perf_counter()
            step = self.state.global_step
            if self._last_log_time is not None and step > self._last_log_step:
                step_time = (now - self._last_log_time) / (step - self._last_log_step)
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
        if self.is_world_process_zero():
            logger.info(str(logs))
        super().log(logs, *args, **kwargs)
