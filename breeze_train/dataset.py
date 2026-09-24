"""Training dataset: manifest records + phase-2 codec cache -> phase-3 examples.

Never encodes audio lazily. `__getitem__` requires the codec cache to already
exist; a missing entry raises `FileNotFoundError` naming the record and the
expected cache path, instructing the caller to run
`python -m breeze_train.codec_cache` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.utils.data

from breeze_train import manifest as manifest_module
from breeze_train.codec_cache import cache_key, load_codes
from breeze_train.examples import build_example


class BreezeTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        tokenizer: Any,
        model_config: Any,
    ) -> None:
        self.records = manifest_module.load_manifest(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __len__(self) -> int:
        return len(self.records)

    def _load_cached_codes(self, record_id: str, audio_path: str) -> torch.Tensor:
        cache_path = self.cache_dir / f"{cache_key(audio_path)}.npy"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Manifest record {record_id!r}: no codec cache entry for "
                f"{audio_path!r} (expected {cache_path}). Run "
                "`python -m breeze_train.codec_cache <ckpt_dir> <manifest> "
                "<cache_dir>` first."
            )
        return load_codes(self.cache_dir, audio_path)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        target_codes = self._load_cached_codes(record.id, record.audio_path)
        ref_codes = (
            self._load_cached_codes(record.id, record.ref_audio_path)
            if record.ref_audio_path
            else None
        )
        return build_example(
            self.tokenizer, self.model_config, record, target_codes, ref_codes
        )
