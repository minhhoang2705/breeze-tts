#!/usr/bin/env python3
"""Generate <id>.wav for every row of a tts-eval evalset through the live Breeze server.

Each row is a zero-shot clone: synthesize `text` with `ref_audio` + `ref_text` as the
voice prompt (seed 42, like tts-eval's eval_breeze.sh). Resumes: existing wavs are kept.
Writes timing.jsonl (latency and audio duration per item) next to the wavs.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import time
import urllib.request
import wave
from pathlib import Path


def one(url: str, row: dict, out: Path) -> dict:
    ref = Path(row["ref_audio"]).read_bytes()
    body = {
        "input": row["text"],
        "ref_audio": f"data:audio/wav;base64,{base64.b64encode(ref).decode()}",
        "ref_text": row["ref_text"],
        "seed": 42,
        "response_format": "wav",
    }
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    data = urllib.request.urlopen(req, timeout=300).read()
    elapsed = time.perf_counter() - t0
    with wave.open(io.BytesIO(data)) as w:
        duration = w.getnframes() / w.getframerate()
    tmp = out / f".{row['id']}.wav.part"
    tmp.write_bytes(data)
    tmp.rename(out / f"{row['id']}.wav")
    return {"id": row["id"], "elapsed": elapsed, "duration": duration}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.evalset.read_text().splitlines() if line.strip()]
    todo = [r for r in rows if not (args.out / f"{r['id']}.wav").is_file()]
    print(f"{len(rows)} rows, {len(todo)} to generate", flush=True)
    errors = 0
    t0 = time.perf_counter()
    with (args.out / "timing.jsonl").open("a") as log, \
            concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(one, args.url, r, args.out): r["id"] for r in todo}
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                log.write(json.dumps(fut.result()) + "\n")
            except Exception as exc:  # keep going; a missing wav scores as WER 1
                errors += 1
                print(f"ERROR {futs[fut]}: {exc!r}", flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(todo)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"finished: {len(todo) - errors} ok, {errors} errors, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
