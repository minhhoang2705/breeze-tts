#!/usr/bin/env python3
"""Plot a BreezeTrainer train.log: loss terms (train + held-out), grad norms, LR vs step.

Records are the dict reprs BreezeTrainer.log() writes: training records start
with `{'loss': ...`, evaluation records with `{'eval_<name>_loss': ...`. They are
matched across the whole text, not per line, because in a stdout capture tqdm's
carriage-return bar can splice them mid-line. Current logs carry `step`; older
ones (e.g. nv_lora01) don't, so the step is then recovered from `epoch`,
snapped to the logging grid (--logging-steps required in that case).

Held-out losses (`--eval-manifest` runs) are drawn over the training curves;
`grad_norm/<module>` (post-clip, per top-level module) next to the total.

Usage: plot_train.py TRAIN_LOG OUT_PNG --steps-per-epoch N [--logging-steps K] [--title T]
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r"\{'loss': [^{}]*\}")
EVAL_RE = re.compile(r"\{'eval_[^{}]*\}")
EVAL_LOSS_RE = re.compile(r"^eval_(.+)_loss$")
COMPONENTS = ("loss", "backbone_loss", "depth_decoder_loss")


def with_steps(rows: list[dict], steps_per_epoch: float, logging_steps: int | None) -> list[dict]:
    for row in rows:
        if "step" in row:
            continue
        if logging_steps is None:
            raise SystemExit("log records have no 'step'; pass --logging-steps to derive it from 'epoch'")
        raw = row["epoch"] * steps_per_epoch
        row["step"] = max(logging_steps, round(raw / logging_steps) * logging_steps)
    steps = [r["step"] for r in rows]
    if len(set(steps)) != len(steps):
        raise SystemExit(f"duplicate steps; check --steps-per-epoch ({steps_per_epoch})")
    return rows


def parse(path: Path, steps_per_epoch: float, logging_steps: int | None) -> tuple[list[dict], dict[str, list[dict]]]:
    text = path.read_text()
    train = with_steps([ast.literal_eval(m.group(0)) for m in TRAIN_RE.finditer(text)],
                       steps_per_epoch, logging_steps)
    evals: dict[str, list[dict]] = {}
    for match in EVAL_RE.finditer(text):
        row = ast.literal_eval(match.group(0))
        name = next(m.group(1) for k in row if (m := EVAL_LOSS_RE.match(k))
                    and not k.endswith(("_backbone_loss", "_depth_decoder_loss")))
        evals.setdefault(name, []).append(
            {"step": row["step"], **{c: row.get(f"eval_{name}_{c}") for c in COMPONENTS}})
    return train, evals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_log", type=Path)
    parser.add_argument("out_png", type=Path)
    parser.add_argument("--steps-per-epoch", type=float, required=True)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    train, evals = parse(args.train_log, args.steps_per_epoch, args.logging_steps)
    step = [r["step"] for r in train]
    panels = [("loss", "Total loss"), ("backbone_loss", "Backbone loss"),
              ("depth_decoder_loss", "Depth decoder loss"), ("grad_norm", "Grad norm"),
              ("learning_rate", "Learning rate")]
    module_norms = sorted({k for r in train for k in r if k.startswith("grad_norm/")})
    fig, axes = plt.subplots(1, len(panels), figsize=(22, 4.2))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(step, [r[key] for r in train], color="tab:blue", linewidth=1.2,
                label="total" if key == "grad_norm" else "train")
        if key in COMPONENTS:
            for i, (name, rows) in enumerate(sorted(evals.items())):
                ax.plot([r["step"] for r in rows], [r[key] for r in rows], marker="o", markersize=3,
                        linewidth=1.4, color=f"C{i + 1}", label=f"eval {name}")
        if key == "grad_norm":
            for i, module in enumerate(module_norms):
                ax.plot(step, [r.get(module) for r in train], linewidth=1, color=f"C{i + 4}",
                        label=module.removeprefix("grad_norm/"))
        if len(ax.get_lines()) > 1:
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        for epoch_boundary in range(1, int(step[-1] / args.steps_per_epoch) + 1):
            ax.axvline(epoch_boundary * args.steps_per_epoch, color="gray", linestyle=":", linewidth=1)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=120)
    first, last = train[0], train[-1]
    print(f"{len(train)} train points, steps {first['step']}..{last['step']}; "
          f"loss {first['loss']:.3f} -> {last['loss']:.3f}; eval sets: "
          + (", ".join(f"{n} ({len(r)} points, {r[0]['loss']:.3f} -> {r[-1]['loss']:.3f})"
                       for n, r in sorted(evals.items())) or "none"))


if __name__ == "__main__":
    main()
