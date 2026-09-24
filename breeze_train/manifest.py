"""JSONL manifest schema and loader for training samples.

A manifest record describes one training example: the target audio/text, plus
optional instruction and reference-audio fields for the conditioning
templates in ``breeze_infer.templates``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    audio_path: str
    text: str
    instruction: str | None = None
    ref_audio_path: str | None = None
    ref_text: str | None = None
    speaker: str = "S0"


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """Load and validate a JSONL manifest.

    Raises ``ValueError`` naming the offending record ``id`` when a
    validation rule is violated.
    """
    path = Path(path)
    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        record = ManifestRecord(
            id=data["id"],
            audio_path=data["audio_path"],
            text=data["text"],
            instruction=data.get("instruction"),
            ref_audio_path=data.get("ref_audio_path"),
            ref_text=data.get("ref_text"),
            speaker=data.get("speaker", "S0"),
        )

        if record.id in seen_ids:
            raise ValueError(f"Duplicate manifest record id: {record.id!r}")
        seen_ids.add(record.id)

        if not Path(record.audio_path).exists():
            raise ValueError(
                f"Manifest record {record.id!r}: audio_path does not exist: "
                f"{record.audio_path!r}"
            )

        if not record.text.strip():
            raise ValueError(f"Manifest record {record.id!r}: text is empty")

        has_ref_audio = record.ref_audio_path is not None
        has_ref_text = record.ref_text is not None
        if has_ref_audio != has_ref_text:
            raise ValueError(
                f"Manifest record {record.id!r}: ref_audio_path and ref_text "
                "must be both present or both absent"
            )

        records.append(record)

    return records


def to_request(record: ManifestRecord) -> dict:
    """Return the request dict shape ``breeze_infer.templates`` expects."""
    request: dict = {
        "id": record.id,
        "text": record.text,
        "speaker": record.speaker,
    }
    if record.instruction is not None:
        request["instruction"] = record.instruction
    if record.ref_audio_path is not None:
        request["ref_audio_path"] = record.ref_audio_path
    if record.ref_text is not None:
        request["ref_text"] = record.ref_text
    return request
