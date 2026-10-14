#!/usr/bin/env python3
"""Plot matrix_sweep.json (bench_matrix.py, concurrency 1/4/8/16): latency, TTFA,
throughput, GPU util and the max_running_requests=12 admission cap vs concurrency.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_COLOR = {"plain": "tab:blue", "clone": "tab:orange", "design_cfg": "tab:green"}
CAP = 12  # engine_builder.py generation_defaults: max_running_requests / cuda_graph_max_bs


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def load(path: Path) -> dict[str, dict[str, list[dict]]]:
    rows = json.loads(path.read_text())
    by_mode: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], {}).setdefault(r["kind"], []).append(r)
    for kinds in by_mode.values():
        for cells in kinds.values():
            cells.sort(key=lambda r: r["conc"])
    return by_mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    by_mode = load(args.sweep)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode, kinds in by_mode.items():
        if mode == "stream":
            fig, ((ax_lat, ax_ttfa), (ax_thr, ax_gpu)) = plt.subplots(2, 2, figsize=(11, 8))
            panels = [(ax_lat, "Latency (solid=p50, dashed=p95)", "seconds"),
                      (ax_ttfa, "Time to first audio (p50)", "seconds"),
                      (ax_thr, "Throughput", "requests / s"),
                      (ax_gpu, "GPU utilization (mean during cell)", "%")]
        else:
            # WAV is buffered, not chunked: "first byte" only arrives once generation
            # finishes, so TTFA == latency here and the panel would be a duplicate.
            fig, (ax_lat, ax_thr, ax_gpu) = plt.subplots(1, 3, figsize=(15, 4.2))
            ax_ttfa = None
            panels = [(ax_lat, "Latency (solid=p50, dashed=p95)", "seconds"),
                      (ax_thr, "Throughput", "requests / s"),
                      (ax_gpu, "GPU utilization (mean during cell)", "%")]
        for kind, cells in kinds.items():
            xs = [c["conc"] for c in cells]
            color = KIND_COLOR.get(kind, None)
            lat_p50 = [pct([r["elapsed"] for r in c["results"]], 0.5) for c in cells]
            lat_p95 = [pct([r["elapsed"] for r in c["results"]], 0.95) for c in cells]
            ax_lat.plot(xs, lat_p50, "o-", color=color, label=kind)
            ax_lat.plot(xs, lat_p95, "o--", color=color, alpha=0.4)

            if ax_ttfa is not None:
                ttfa_p50 = [pct([r["ttfa"] for r in c["results"]], 0.5) for c in cells]
                ax_ttfa.plot(xs, ttfa_p50, "o-", color=color, label=kind)

            thr = [len(c["results"]) / c["wall"] for c in cells]
            ax_thr.plot(xs, thr, "o-", color=color, label=kind)

            gpu = [statistics.mean(c["gpu"]) if c["gpu"] else float("nan") for c in cells]
            ax_gpu.plot(xs, gpu, "o-", color=color, label=kind)

        for ax, title, ylabel in panels:
            ax.set_title(title)
            ax.set_xlabel("concurrent requests")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 8, 16])
            ax.set_xticklabels(["1", "4", "8", "16"])
            ax.axvline(CAP, color="gray", lw=1, ls=":")
            ax.legend(fontsize=8)
        ax_lat.text(CAP, ax_lat.get_ylim()[1], f" max_running_requests={CAP}",
                    fontsize=7, color="gray", va="top")

        fig.suptitle(f"Breeze TTS server — {mode} response, checkpoint-8000, H100 (unique reference per clone request)")
        fig.tight_layout()
        out = args.out_dir / f"concurrency_sweep_{mode}.svg"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
