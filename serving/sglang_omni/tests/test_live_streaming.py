"""Live: streamed PCM must equal the buffered WAV for the same greedy request.

Opt-in, runs only when a Breeze server answers at BREEZE_OMNI_URL (default
http://127.0.0.1:8000). Greedy sampling makes both requests produce the same codes,
so this checks the chunked engine->vocoder stream (4 then 12 frames, flushed at
request end) and the incremental codec against the one-shot buffered decode.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("BREEZE_OMNI_URL", "http://127.0.0.1:8000")
REF_WAV = REPO / "data/vivoice_scale3/audio/vivoice_s0_0001491.wav"
REF_TEXT = "Rồi rồi, tới công chuyện rồi."
TEXT = "Please remember to bring your umbrella, because the forecast says it will rain."


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no Breeze server at {BASE_URL}")


def _case(kind: str) -> dict:
    body: dict = {"input": TEXT, "seed": 0}
    # Greedy decoding tends to loop to max_new_tokens; 150 frames (12 s) still
    # spans the 4-frame first chunk, many 12-frame chunks and the final flush.
    engine = {"temperature": 0.0, "depth_temperature": 0.0, "max_new_tokens": 150}
    if kind in ("design_cfg", "edit_cfg"):
        body["instructions"] = "Speak slowly with a calm, serious tone."
        engine["cfg_scale"] = 4.0
    if kind in ("clone", "edit_cfg"):
        if not REF_WAV.is_file():
            pytest.skip(f"reference wav {REF_WAV} not available")
        body["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(REF_WAV.read_bytes()).decode()
        body["ref_text"] = REF_TEXT
    body["stage_params"] = {"tts_engine": engine}
    return body


def _post(body: dict) -> bytes:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        assert response.status == 200
        return response.read()


@pytest.mark.parametrize("kind", ["plain", "design_cfg", "clone", "edit_cfg"])
def test_streamed_pcm_equals_buffered_wav(kind: str) -> None:
    body = _case(kind)
    with wave.open(io.BytesIO(_post({**body, "response_format": "wav"}))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24_000, 1, 2)
        buffered = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    pcm = _post({**body, "response_format": "pcm", "stream": True})
    streamed = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)

    # Whole 80 ms frames; streaming must not drop the tail chunk flushed at the end.
    assert buffered.size > 0 and buffered.size % 1920 == 0
    assert streamed.size == buffered.size
    corr = float(np.corrcoef(streamed.astype(np.float64), buffered.astype(np.float64))[0, 1])
    assert corr > 0.99, f"{kind}: streamed vs buffered correlation {corr:.4f}"
