#!/usr/bin/env python3
"""Latency matrix for the Breeze SGLang-Omni server: kind x mode x concurrency.

For each config it reports per-request latency, TTFA, audio duration, RTF, plus
average GPU utilization and the busiest server threads (CPU seconds per wall
second), to tell GPU-bound from CPU/GIL-bound stages apart.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
import wave
from pathlib import Path

SR = 24_000
TEXT = "Cảm ơn bạn đã đến dự buổi họp sáng nay, hôm nay chúng ta sẽ cùng nhau xem lại kết quả của tuần trước."
INSTRUCTION = "Giọng nữ trẻ, ấm áp, đọc rõ ràng và bình tĩnh."
CLK = os.sysconf("SC_CLK_TCK")


def body_for(kind: str, mode: str, seed: int, ref) -> dict:
    body: dict = {"input": TEXT, "seed": seed}
    if mode == "stream":
        body.update(stream=True, response_format="pcm")
    else:
        body.update(response_format="wav")
    if kind == "design_cfg":
        body["instructions"] = INSTRUCTION
        body["stage_params"] = {"tts_engine": {"cfg_scale": 4.0}}
    if kind == "clone":
        body["ref_audio"], body["ref_text"] = ref()
    return body


def ref_factory(ref_audio: Path, ref_text: str, unique: bool):
    """Return a callable giving (data URI, transcript) for each clone request.

    With ``unique``, every call appends a different number of silent samples, so
    the server sees new reference bytes (a cache miss) while the audio stays the same.
    """
    raw = ref_audio.read_bytes()
    fixed = (f"data:audio/wav;base64,{base64.b64encode(raw).decode()}", ref_text)
    if not unique:
        return lambda: fixed
    with wave.open(io.BytesIO(raw)) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    salt = iter(range(1, 1_000_000))

    def make():
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + b"\0" * params.sampwidth * params.nchannels * next(salt))
        return f"data:audio/wav;base64,{base64.b64encode(out.getvalue()).decode()}", ref_text

    return make


def run_one(url: str, mode: str, body: dict) -> dict:
    req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttfa = None
    chunks = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        while chunk := resp.read1(65536):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(chunk)
    elapsed = time.perf_counter() - t0
    data = b"".join(chunks)
    if mode == "stream":
        duration = len(data) / (2 * SR)
    else:
        with wave.open(io.BytesIO(data)) as w:
            duration = w.getnframes() / w.getframerate()
    return {"elapsed": elapsed, "ttfa": ttfa or elapsed, "duration": duration}


def server_pids() -> list[int]:
    parents: dict[int, int] = {}
    cmd: dict[int, str] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            parents[int(p.name)] = int(stat.rsplit(")", 1)[1].split()[1])
            cmd[int(p.name)] = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
    roots = {pid for pid, c in cmd.items() if "sgl-omni serve" in c}
    tree = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sorted(tree)


def thread_cpu(pids: list[int]) -> dict[tuple[int, int, str], float]:
    out = {}
    for pid in pids:
        try:
            for t in Path(f"/proc/{pid}/task").iterdir():
                try:
                    fields = (t / "stat").read_text().rsplit(")", 1)[1].split()
                    name = (t / "comm").read_text().strip()
                    out[(pid, int(t.name), name)] = (int(fields[11]) + int(fields[12])) / CLK
                except OSError:
                    continue
        except OSError:
            continue
    return out


class GpuSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[int] = []
        self.running = True

    def run(self) -> None:
        while self.running:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                self.samples.append(int(r.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            time.sleep(0.25)


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--kinds", nargs="+", default=["plain", "clone", "design_cfg"])
    ap.add_argument("--modes", nargs="+", default=["wav", "stream"])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--requests-per-level", type=int, default=None,
                    help="requests per kind/mode/concurrency cell; default 2x the highest --concurrency (min 16)")
    ap.add_argument("--unique-refs", action="store_true", help="new reference bytes per clone request (cache miss)")
    args = ap.parse_args()
    ref = ref_factory(args.ref_audio, args.ref_text, args.unique_refs)
    pids = server_pids()
    print(f"server pids: {pids}")
    rows = []
    print("| mode | kind | conc | n | err | req/s | latency p50 | latency max | TTFA p50 | audio s (min-max) | RTF mean | GPU util avg | busiest threads (cpu s / wall s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n_per_level = args.requests_per_level or max(16, 2 * max(args.concurrency))
    for mode in args.modes:
        for kind in args.kinds:
            run_one(args.url, mode, body_for(kind, mode, 99, ref))  # warm this kind+mode once, untimed
    for mode in args.modes:
        for kind in args.kinds:
            # Same fixed request set (same seeds -> same text/ref) at every concurrency
            # level, so C1 and C4 do identical work and latency/duration are comparable.
            # Rebuilt per level so --unique-refs never reuses a reference across levels.
            for conc in args.concurrency:
                bodies = [body_for(kind, mode, 100 + i, ref) for i in range(n_per_level)]
                n = n_per_level
                cpu0 = thread_cpu(pids)
                gpu = GpuSampler()
                gpu.start()
                t0 = time.perf_counter()
                results, errors = [], []
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    futs = {pool.submit(run_one, args.url, mode, b): i for i, b in enumerate(bodies)}
                    for fut in concurrent.futures.as_completed(futs):
                        try:
                            r = fut.result()
                            r["idx"] = futs[fut]
                            results.append(r)
                        except Exception as exc:  # keep measuring the rest
                            errors.append(repr(exc))
                wall = time.perf_counter() - t0
                gpu.running = False
                gpu.join()
                cpu1 = thread_cpu(pids)
                busy = sorted(((cpu1[k] - cpu0.get(k, 0.0)) / wall, k[2]) for k in cpu1)[-3:][::-1]
                busy_s = ", ".join(f"{name}={v:.2f}" for v, name in busy)
                gpu_avg = statistics.mean(gpu.samples) if gpu.samples else float("nan")
                if results:
                    lat = [r["elapsed"] for r in results]
                    ttfa = [r["ttfa"] for r in results]
                    durs = [r["duration"] for r in results]
                    print(f"| {mode} | {kind} | {conc} | {n} | {len(errors)} | {len(results) / wall:.2f} "
                          f"| {pct(lat, .5):.2f} | {max(lat):.2f} "
                          f"| {pct(ttfa, .5):.2f} | {min(durs):.2f}-{max(durs):.2f} "
                          f"| {statistics.mean(r['elapsed'] / r['duration'] for r in results):.3f} | {gpu_avg:.0f}% | {busy_s} |", flush=True)
                else:
                    print(f"| {mode} | {kind} | {conc} | {n} | {len(errors)} | - | - | - | - | - | - | {gpu_avg:.0f}% | {busy_s} |", flush=True)
                for e in errors:
                    print(f"  ERROR {e}", flush=True)
                rows.append({"mode": mode, "kind": kind, "conc": conc, "wall": wall, "results": results,
                             "errors": errors, "gpu": gpu.samples, "busy": busy})
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
