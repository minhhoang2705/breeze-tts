"""Training-time batch collator.

Right-pads (never left-pads, per L9 -- see phase-04) per-sample examples from
`breeze_train.examples.build_example` into the exact tensor shapes the
model's forward expects, including the non-per-sample flat `text_ids_len`
contract (L4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class BreezeDataCollator:
    pad_token_id: int
    config: Any
    label_pad_id: int = -100

    def __post_init__(self) -> None:
        if self.pad_token_id == self.config.audio_token_id:
            raise ValueError(
                "pad_token_id "
                f"({self.pad_token_id}) equals config.audio_token_id "
                f"({self.config.audio_token_id}) -- right-padding would make every "
                "pad position look like an audio frame"
            )
        if self.pad_token_id == self.config.audio_eos_token_id:
            raise ValueError(
                "pad_token_id "
                f"({self.pad_token_id}) equals config.audio_eos_token_id "
                f"({self.config.audio_eos_token_id}) -- right-padding would silently "
                "overwrite pad labels with a real backbone-EOS target"
            )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].shape[0]) for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        text_ids_mask_list = []
        for f in features:
            seq_len = int(f["input_ids"].shape[0])
            pad_len = max_len - seq_len

            input_ids = F.pad(f["input_ids"], (0, pad_len), value=self.pad_token_id)
            attention_mask = F.pad(
                torch.ones(seq_len, dtype=torch.long), (0, pad_len), value=0
            )
            labels = F.pad(f["labels"], (0, pad_len), value=self.label_pad_id)
            text_ids_mask = F.pad(f["text_ids_mask"], (0, pad_len), value=False)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            text_ids_mask_list.append(text_ids_mask)

        text_ids_len = torch.cat([f["text_ids_len"] for f in features], dim=0)
        input_values = (
            torch.cat([f["input_values"] for f in features], dim=0)
            .unsqueeze(0)
            .to(torch.long)
        )

        batch = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
            "text_ids_mask": torch.stack(text_ids_mask_list, dim=0),
            "text_ids_len": text_ids_len,
            "input_values": input_values,
        }

        mask_sum = int(batch["text_ids_mask"].sum())
        len_sum = int(batch["text_ids_len"].sum())
        assert mask_sum == len_sum, (
            f"text_ids_mask.sum() ({mask_sum}) != text_ids_len.sum() ({len_sum}) "
            "-- the collator produced an inconsistent batch"
        )

        return batch
