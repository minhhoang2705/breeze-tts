"""HF Trainer subclass that surfaces the model's two loss components.

`compute_loss` returns exactly `outputs.loss` (the model already applies
`depth_header_loss_weight` at models/breeze.py:1858-1862 -- never
recompute/reweight it here). `log` merges the mean `backbone_loss` /
`depth_decoder_loss` since the previous log call into the logs dict and
emits it through the `breeze_train` logger, because `Trainer` prints its
`{'loss': ...}` lines via `print`/`tqdm.write`, not the `logging` module --
a bare `FileHandler` would otherwise produce a `train.log` with no loss
records at all, breaking every gate that greps it (phases 6.4, 8.2, 9.2).
"""

from __future__ import annotations

import logging

import transformers

logger = logging.getLogger("breeze_train")


class BreezeTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._backbone_loss_sum = 0.0
        self._depth_decoder_loss_sum = 0.0
        self._loss_component_steps = 0

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
        if self._loss_component_steps > 0:
            logs["backbone_loss"] = self._backbone_loss_sum / self._loss_component_steps
            logs["depth_decoder_loss"] = (
                self._depth_decoder_loss_sum / self._loss_component_steps
            )
            self._backbone_loss_sum = 0.0
            self._depth_decoder_loss_sum = 0.0
            self._loss_component_steps = 0

        logger.info(str(logs))
        super().log(logs, *args, **kwargs)
