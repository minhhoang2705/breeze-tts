"""Visualize training and validation metrics for the Vietnamese pilot runs.

Throwaway analysis tooling (not part of breeze_train), producing PNGs under
outputs/plots/. Parses train.log files (the `'key': value` dict-repr lines
BreezeTrainer.log() writes) and combines them with the manual validation
sweeps run against saved checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_NUM = r"[0-9.eE+-]+"


def parse_train_log(path: Path, logging_steps: int) -> dict[str, list[float]]:
    text = path.read_text()
    steps, loss, bb, dd = [], [], [], []
    for i, line in enumerate(text.splitlines()):
        m_loss = re.search(rf"'loss': ({_NUM})", line)
        m_bb = re.search(rf"'backbone_loss': ({_NUM})", line)
        m_dd = re.search(rf"'depth_decoder_loss': ({_NUM})", line)
        if not (m_loss and m_bb and m_dd):
            continue
        steps.append((i + 1) * logging_steps)
        loss.append(float(m_loss.group(1)))
        bb.append(float(m_bb.group(1)))
        dd.append(float(m_dd.group(1)))
    return {"step": steps, "loss": loss, "backbone_loss": bb, "depth_decoder_loss": dd}


def plot_training_curves() -> None:
    runs = [
        ("pilot (120 train)", "outputs/vietnamese_pilot/train.log", 10, "tab:red"),
        ("scale1 (3,000 train)", "outputs/vietnamese_scale1/train.log", 10, "tab:orange"),
        ("scale2 (9,000 train)", "outputs/vietnamese_scale2/train.log", 20, "tab:green"),
        ("scale3 (30,000 train)", "outputs/vietnamese_scale3/train.log", 30, "tab:blue"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=False)
    metrics = [("loss", "Total loss"), ("backbone_loss", "Backbone loss"), ("depth_decoder_loss", "Depth decoder loss")]

    for ax, (key, title) in zip(axes, metrics):
        for label, path, logging_steps, color in runs:
            p = Path(path)
            if not p.exists():
                continue
            data = parse_train_log(p, logging_steps)
            if not data["step"]:
                continue
            ax.plot(data["step"], data[key], label=label, color=color, linewidth=1.5)
        ax.set_xlabel("training step")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Vietnamese LoRA fine-tuning: training loss curves (backbone_model + depth_decoder, policy p1)")
    fig.tight_layout()
    out = OUT_DIR / "training_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_validation_sweep(name: str, baseline: dict, checkpoints: dict[int, dict], out_name: str) -> None:
    steps = sorted(checkpoints.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("mean_loss", "Total loss (val, held-out speakers)"), ("mean_backbone", "Backbone loss (val)"), ("mean_depth", "Depth decoder loss (val)")]

    for ax, (key, title) in zip(axes, metrics):
        vals = [checkpoints[s][key] for s in steps]
        ax.plot(steps, vals, marker="o", color="tab:blue", label=name)
        ax.axhline(baseline[key], color="tab:gray", linestyle="--", label="baseline (untrained)")
        ax.set_xlabel("training step (checkpoint)")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{name}: held-out validation loss vs training step (lower = better generalization)")
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_scale_comparison_summary() -> None:
    """Final-checkpoint backbone_loss across all three runs, on held-out val."""
    labels = ["pilot\n(120, 300 steps)", "scale1\n(3k, 600 steps)", "scale2\n(9k, 1600 steps)", "scale3\n(30k, 3000 steps)"]
    baseline_bb = [2.8333, 2.8583, 2.8222, 2.8215]
    tuned_bb = [4.1307, 1.7587, 1.5923, 1.5339]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline_bb, width, label="baseline (untrained)", color="tab:gray")
    ax.bar([i + width / 2 for i in x], tuned_bb, width, label="tuned (final checkpoint)", color="tab:blue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean backbone_loss on held-out validation")
    ax.set_title(
        "Held-out validation backbone_loss vs training set size:\n"
        "overfitting at 120 samples, genuine improvement at 3k/9k/30k, flattening by 30k"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "scale_comparison_all_runs.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_training_curves()

    scale1_baseline = {"mean_loss": 8.4319, "mean_backbone": 2.8583, "mean_depth": 5.5736}
    scale1_checkpoints = {
        100: {"mean_loss": 7.2609, "mean_backbone": 1.9365, "mean_depth": 5.3244},
        200: {"mean_loss": 7.1115, "mean_backbone": 1.8206, "mean_depth": 5.2910},
        300: {"mean_loss": 7.0566, "mean_backbone": 1.7805, "mean_depth": 5.2760},
        400: {"mean_loss": 7.0405, "mean_backbone": 1.7668, "mean_depth": 5.2737},
        500: {"mean_loss": 7.0300, "mean_backbone": 1.7591, "mean_depth": 5.2709},
        600: {"mean_loss": 7.0290, "mean_backbone": 1.7587, "mean_depth": 5.2704},
    }
    plot_validation_sweep("scale1 (3,000 train samples)", scale1_baseline, scale1_checkpoints, "scale1_validation_sweep.png")

    scale2_baseline = {"mean_loss": 8.3928, "mean_backbone": 2.8222, "mean_depth": 5.5707}
    scale2_checkpoints = {
        200: {"mean_loss": 7.0790, "mean_backbone": 1.7921, "mean_depth": 5.2868},
        400: {"mean_loss": 6.9490, "mean_backbone": 1.6954, "mean_depth": 5.2536},
        600: {"mean_loss": 6.8878, "mean_backbone": 1.6520, "mean_depth": 5.2357},
        800: {"mean_loss": 6.8446, "mean_backbone": 1.6190, "mean_depth": 5.2255},
        1000: {"mean_loss": 6.8205, "mean_backbone": 1.6026, "mean_depth": 5.2179},
        1200: {"mean_loss": 6.8121, "mean_backbone": 1.5972, "mean_depth": 5.2148},
        1400: {"mean_loss": 6.8060, "mean_backbone": 1.5942, "mean_depth": 5.2119},
        1600: {"mean_loss": 6.8032, "mean_backbone": 1.5923, "mean_depth": 5.2109},
    }
    plot_validation_sweep("scale2 (9,000 train samples)", scale2_baseline, scale2_checkpoints, "scale2_validation_sweep.png")

    scale3_baseline = {"mean_loss": 8.3895, "mean_backbone": 2.8215, "mean_depth": 5.5680}
    scale3_checkpoints = {
        300: {"mean_loss": 7.0132, "mean_backbone": 1.7440, "mean_depth": 5.2692},
        600: {"mean_loss": 6.8873, "mean_backbone": 1.6550, "mean_depth": 5.2324},
        900: {"mean_loss": 6.8154, "mean_backbone": 1.6037, "mean_depth": 5.2117},
        1200: {"mean_loss": 6.7735, "mean_backbone": 1.5781, "mean_depth": 5.1954},
        1500: {"mean_loss": 6.7507, "mean_backbone": 1.5644, "mean_depth": 5.1863},
        1800: {"mean_loss": 6.7245, "mean_backbone": 1.5470, "mean_depth": 5.1775},
        2100: {"mean_loss": 6.7148, "mean_backbone": 1.5405, "mean_depth": 5.1743},
        2400: {"mean_loss": 6.7098, "mean_backbone": 1.5381, "mean_depth": 5.1717},
        2700: {"mean_loss": 6.7044, "mean_backbone": 1.5342, "mean_depth": 5.1702},
        3000: {"mean_loss": 6.7043, "mean_backbone": 1.5339, "mean_depth": 5.1703},
    }
    plot_validation_sweep("scale3 (30,000 train samples)", scale3_baseline, scale3_checkpoints, "scale3_validation_sweep.png")


    restart_checkpoints = {
        3200: {"mean_loss": 6.7399, "mean_backbone": 1.5573, "mean_depth": 5.1826},
        3400: {"mean_loss": 6.7271, "mean_backbone": 1.5476, "mean_depth": 5.1795},
        3600: {"mean_loss": 6.7106, "mean_backbone": 1.5370, "mean_depth": 5.1736},
        3800: {"mean_loss": 6.6964, "mean_backbone": 1.5274, "mean_depth": 5.1690},
        4000: {"mean_loss": 6.6904, "mean_backbone": 1.5232, "mean_depth": 5.1671},
        4200: {"mean_loss": 6.6902, "mean_backbone": 1.5232, "mean_depth": 5.1670},
    }
    # Combine scale3's own sweep (steps 300-3000) with the warm-restart
    # continuation (steps 3200-4200, same run continued with a fresh cosine
    # cycle from the checkpoint-3000 weights) into one combined sweep,
    # proving/disproving the schedule-artifact vs capacity-ceiling question.
    combined = {**scale3_checkpoints, **restart_checkpoints}
    plot_validation_sweep(
        "scale3 + warm restart (30,000 train, steps 300-3000 then fresh LR cycle to 4200)",
        scale3_baseline,
        combined,
        "scale3_warm_restart_sweep.png",
    )
    plot_scale_comparison_summary()
