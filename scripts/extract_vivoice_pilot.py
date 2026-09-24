"""One-off data extraction from capleaf/viVoice into breeze_train manifests.

Reads one or more already-downloaded parquet shards, decodes the embedded
audio bytes to real wav files on disk, and writes train/val JSONL manifests
in the breeze_train.manifest schema. Throwaway data-prep tooling for the
Vietnamese pilot, not part of the reusable breeze_train package.

Usage: extract_vivoice_pilot.py <out_subdir> <n_train> <n_val> <shard_path> [<shard_path> ...]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

OUT_SUBDIR = sys.argv[1]
N_TRAIN = int(sys.argv[2])
N_VAL = int(sys.argv[3])
SHARD_PATHS = [Path(p) for p in sys.argv[4:]]

OUT_AUDIO_DIR = Path(f"data/{OUT_SUBDIR}/audio")
TRAIN_MANIFEST = Path(f"data/{OUT_SUBDIR}/train.jsonl")
VAL_MANIFEST = Path(f"data/{OUT_SUBDIR}/val.jsonl")

OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

all_channels: list[str] = []
all_texts: list[str] = []
all_audio: list[dict] = []
shard_of: list[int] = []

for shard_idx, shard_path in enumerate(SHARD_PATHS):
    table = pq.read_table(shard_path)
    n = table.num_rows
    all_channels.extend(table.column("channel").to_pylist())
    all_texts.extend(table.column("text").to_pylist())
    all_audio.extend(table.column("audio").to_pylist())
    shard_of.extend([shard_idx] * n)

seen_channels_order: list[str] = []
for ch in all_channels:
    if ch not in seen_channels_order:
        seen_channels_order.append(ch)

n_val_channels = max(1, round(len(seen_channels_order) * 0.15))
val_channels = set(seen_channels_order[-n_val_channels:])

train_records = []
val_records = []

for i, (ch, text, audio, shard_idx) in enumerate(
    zip(all_channels, all_texts, all_audio, shard_of)
):
    if not text or not text.strip():
        continue
    audio_bytes = audio["bytes"]
    if audio_bytes is None:
        continue

    target_list = val_records if ch in val_channels else train_records
    limit = N_VAL if ch in val_channels else N_TRAIN
    if len(target_list) >= limit:
        continue

    record_id = f"vivoice_s{shard_idx}_{i:07d}"
    wav_path = OUT_AUDIO_DIR / f"{record_id}.wav"
    if not wav_path.exists():
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        sf.write(wav_path, wav, sr)

    target_list.append(
        {
            "id": record_id,
            "audio_path": str(wav_path),
            "text": text.strip(),
            "speaker": ch,
        }
    )

    if len(train_records) >= N_TRAIN and len(val_records) >= N_VAL:
        break

TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with TRAIN_MANIFEST.open("w") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with VAL_MANIFEST.open("w") as f:
    for r in val_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

train_channels = {r["speaker"] for r in train_records}
val_channels_used = {r["speaker"] for r in val_records}
overlap = train_channels & val_channels_used

print(f"train: {len(train_records)} records, {len(train_channels)} channels")
print(f"val: {len(val_records)} records, {len(val_channels_used)} channels")
print(f"channel overlap train/val: {len(overlap)}")
