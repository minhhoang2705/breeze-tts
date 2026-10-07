#!/usr/bin/env python3
"""Plot bench.py results (one ``c<N>/results.json`` per concurrency) as an SVG.

    python tools/plot_bench.py ../../outputs/sglang_omni/bench docs/assets/bench.svg

Left panel: delivered audio seconds per wall second (throughput) and mean RTF
per request. Right panel: time to first audio p50/p95.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(bench_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(bench_dir.glob("c*/results.json"), key=lambda p: int(p.parent.name[1:])):
        data = json.loads(path.read_text())
        results = data["results"]
        if not results or "wall_s" not in data:
            raise SystemExit(f"{path}: no results or no wall_s (re-run bench.py)")
        ttfa = sorted(r["ttfa"] for r in results)
        rows.append({
            "c": data["concurrency"],
            "audio_per_s": sum(r["duration"] for r in results) / data["wall_s"],
            "rtf": statistics.mean(r["elapsed"] / r["duration"] for r in results),
            "ttfa_p50": statistics.median(ttfa),
            "ttfa_p95": ttfa[min(len(ttfa) - 1, int(0.95 * len(ttfa)))],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = _load(args.bench_dir)
    xs = [r["c"] for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    left.plot(xs, [r["audio_per_s"] for r in rows], "o-", color="tab:blue", label="audio s / wall s")
    left.set_xlabel("concurrent requests")
    left.set_ylabel("audio seconds per second", color="tab:blue")
    left.set_xticks(xs)
    rtf_axis = left.twinx()
    rtf_axis.plot(xs, [r["rtf"] for r in rows], "s--", color="tab:orange", label="RTF (mean)")
    rtf_axis.axhline(1.0, color="tab:orange", lw=0.8, ls=":")
    rtf_axis.set_ylabel("RTF per request (<1 = faster than real time)", color="tab:orange")
    rtf_axis.set_ylim(0, 1.2)
    left.set_title("Throughput and real-time factor")

    right.plot(xs, [r["ttfa_p50"] for r in rows], "o-", label="p50")
    right.plot(xs, [r["ttfa_p95"] for r in rows], "s--", label="p95")
    right.set_xlabel("concurrent requests")
    right.set_ylabel("seconds")
    right.set_xticks(xs)
    right.set_title("Time to first audio (streaming)")
    right.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
