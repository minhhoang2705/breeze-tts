#!/usr/bin/env python3
"""Concurrent streaming benchmark for the Breeze SGLang-Omni speech API.

The workload cycles through the five request kinds the server supports: plain
English, plain Vietnamese, voice design with CFG, reference clone, and reference
edit with CFG. Requests run to their natural end (no max_new_tokens cap), every
response is saved as a WAV, and the table reports throughput, real-time factor
(wall time / audio duration) and time to first audio.

With --wandb-project the per-concurrency rows, a per-request table and the WAVs
of the lowest concurrency are logged to Weights & Biases (breeze_train/tracking.py:
the API key comes from the repository-root .env). The server's own settings are
not visible from here, so describe the variant with --wandb-run-name / --wandb-tags
(e.g. `--wandb-tags depth-compile-on`). This script is stdlib-only otherwise, so it
also runs with the repository-root venv, which has wandb installed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from breeze_train.tracking import add_wandb_args, init_run

SAMPLE_RATE = 24_000

TEXT_EN = [
    "The library opens at nine, but the coffee shop next door is busy long before that.",
    "Please remember to bring your umbrella, because the forecast says it will rain this afternoon.",
    "We finally fixed the bug that kept the build from passing on the night shift.",
]
TEXT_VI = [
    "Xin chào, hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé.",
    "Cảm ơn bạn đã đến dự buổi họp sáng nay, chúng ta bắt đầu thôi.",
]
INSTRUCTION = "A warm, thoughtful young woman with a clear voice and a calm delivery."
EDIT_INSTRUCTION = "Speak slowly with a restrained, serious tone."
KINDS = ("plain_en", "plain_vi", "design_cfg", "clone", "edit_cfg")


def build_case(index: int, reference: tuple[str, str] | None) -> tuple[str, dict[str, Any]]:
    kind = KINDS[index % len(KINDS)]
    if kind in ("clone", "edit_cfg") and reference is None:
        kind = "plain_en"
    body: dict[str, Any] = {
        "input": TEXT_VI[index % len(TEXT_VI)] if kind == "plain_vi" else TEXT_EN[index % len(TEXT_EN)],
        "response_format": "pcm",
        "stream": True,
        "seed": 1000 + index,
    }
    if kind == "design_cfg":
        body["instructions"] = INSTRUCTION
        body["stage_params"] = {"tts_engine": {"cfg_scale": 4.0}}
    if kind in ("clone", "edit_cfg"):
        assert reference is not None
        body["ref_audio"], body["ref_text"] = reference
    if kind == "edit_cfg":
        body["instructions"] = EDIT_INSTRUCTION
        body["stage_params"] = {"tts_engine": {"cfg_scale": 4.0}}
    return kind, body


def run_one(base_url: str, kind: str, body: dict[str, Any], path: Path, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    chunks: list[bytes] = []
    ttfa: float | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while chunk := response.read1(65536):
            if ttfa is None:
                ttfa = time.perf_counter() - started
            chunks.append(chunk)
    elapsed = time.perf_counter() - started
    pcm = b"".join(chunks)
    if not pcm:
        raise RuntimeError(f"{path.name}: empty audio")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm[: len(pcm) - len(pcm) % 2])
    duration = len(pcm) / (2 * SAMPLE_RATE)
    return {
        "kind": kind, "elapsed": elapsed, "duration": duration, "ttfa": ttfa or elapsed,
        "text": body["input"], "wav": str(path),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("../../outputs/sglang_omni/bench"))
    parser.add_argument("--reference-audio", type=Path, help="WAV used for clone/edit requests")
    parser.add_argument("--reference-text", help="exact transcript of --reference-audio")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request socket timeout (s)")
    add_wandb_args(parser)
    args = parser.parse_args()
    if args.requests < 1 or any(value < 1 for value in args.concurrency):
        parser.error("requests and concurrency values must be positive")
    if bool(args.reference_audio) != bool(args.reference_text):
        parser.error("--reference-audio and --reference-text go together")
    reference = None
    if args.reference_audio:
        encoded = base64.b64encode(args.reference_audio.read_bytes()).decode("ascii")
        reference = (f"data:audio/wav;base64,{encoded}", args.reference_text)

    run = init_run(
        args,
        job_type="benchmark",
        config={"bench": "serving", "url": args.url, "requests": args.requests,
                "concurrency": args.concurrency, "kinds": list(KINDS), "with_reference": reference is not None},
    )
    if run is not None:
        import wandb

        run.define_metric("concurrency")
        run.define_metric("serving/*", step_metric="concurrency")
        summary_table = wandb.Table(columns=["concurrency", "requests", "errors", "req_per_s", "audio_s_per_s", "rtf_mean", "ttfa_p50", "ttfa_p95"])
        request_table = wandb.Table(columns=["concurrency", "kind", "text", "duration_s", "elapsed_s", "rtf", "ttfa_s", "audio"])

    print("| concurrency | requests | errors | req/s | audio s/s | RTF mean | TTFA p50 (s) | TTFA p95 (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for concurrency in args.concurrency:
        run_dir = args.output_dir / f"c{concurrency}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cases = [build_case(i, reference) for i in range(args.requests)]
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(run_one, args.url, kind, body, run_dir / f"{i:02d}-{kind}.wav", args.timeout): i
                for i, (kind, body) in enumerate(cases)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # report every failure, keep measuring the rest
                    errors.append(f"request {futures[future]}: {exc!r}")
        wall = time.perf_counter() - started
        (run_dir / "results.json").write_text(json.dumps(
            {"concurrency": concurrency, "wall_s": wall, "results": results, "errors": errors},
            indent=1,
        ))
        if not results:
            print(f"| {concurrency} | {args.requests} | {len(errors)} | - | - | - | - | - |")
        else:
            ttfa = [r["ttfa"] for r in results]
            if run is not None:
                row = {
                    "req_per_s": len(results) / wall,
                    "audio_s_per_s": sum(r["duration"] for r in results) / wall,
                    "rtf_mean": statistics.mean(r["elapsed"] / r["duration"] for r in results),
                    "ttfa_p50": percentile(ttfa, 0.5),
                    "ttfa_p95": percentile(ttfa, 0.95),
                    "errors": len(errors),
                }
                run.log({"concurrency": concurrency, **{f"serving/{k}": v for k, v in row.items()}})
                summary_table.add_data(concurrency, args.requests, len(errors), row["req_per_s"], row["audio_s_per_s"],
                                       row["rtf_mean"], row["ttfa_p50"], row["ttfa_p95"])
                with_audio = concurrency == min(args.concurrency)
                for r in results:
                    request_table.add_data(
                        concurrency, r["kind"], r["text"], r["duration"], r["elapsed"], r["elapsed"] / r["duration"],
                        r["ttfa"], wandb.Audio(r["wav"]) if with_audio else None,
                    )
            print(
                f"| {concurrency} | {args.requests} | {len(errors)} | {len(results) / wall:.2f} "
                f"| {sum(r['duration'] for r in results) / wall:.2f} "
                f"| {statistics.mean(r['elapsed'] / r['duration'] for r in results):.3f} "
                f"| {percentile(ttfa, 0.5):.3f} | {percentile(ttfa, 0.95):.3f} |"
            )
        for error in errors:
            print(f"  ERROR {error}")
    if run is not None:
        run.log({"serving_summary": summary_table, "serving_requests": request_table})
        run.finish()


if __name__ == "__main__":
    main()
