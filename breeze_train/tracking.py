"""Weights & Biases helpers shared by training, eval and the benchmarks.

Stdlib-only at import time (wandb is imported lazily), so the serving benchmark
can use it from an environment without torch. Every entry point exposes the same
flags through `add_wandb_args`; tracking is off unless `--wandb-project` is set.

Credentials: put `WANDB_API_KEY=...` in the repository-root `.env` (git-ignored;
see `.env.example`), export it, or run `wandb login`. `.env` never overrides a
variable that is already set. Without any of them, runs fall back to offline mode
with a warning instead of blocking on an interactive login prompt; upload them
later with `wandb sync <dir>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import netrc
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV = REPO_ROOT / ".env"
RUN_FILE = "wandb_run.json"

logger = logging.getLogger(__name__)


def add_wandb_args(parser: argparse.ArgumentParser, *, attach: bool = False) -> None:
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument("--wandb-project", default=None, help="Enables Weights & Biases logging")
    group.add_argument("--wandb-entity", default=None)
    group.add_argument("--wandb-run-name", default=None)
    group.add_argument("--wandb-tags", default=None, help="Comma-separated tags")
    group.add_argument("--wandb-group", default=None, help="Groups related runs (e.g. one experiment)")
    group.add_argument("--wandb-notes", default=None)
    if attach:
        group.add_argument(
            "--wandb-run-id",
            default=None,
            help="Log into this existing run (e.g. the training run of the checkpoint "
            "being evaluated) instead of creating a new one",
        )


def load_dotenv(path: Path = DOTENV) -> list[str]:
    """Export KEY=VALUE lines from `path` that are not already set; returns their names.

    Accepts blank lines, `#` comments, an optional `export ` prefix and single
    or double quotes around the value. Values are never logged.
    """
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").split("://")[-1].split("/")[0]
    try:
        return netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *cmd], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {}


def host_state() -> dict[str, Any]:
    state: dict[str, Any] = {"hostname": platform.node(), "python": platform.python_version()}
    try:
        import torch

        state["torch"] = torch.__version__
        if torch.cuda.is_available():
            state["gpu"] = torch.cuda.get_device_name()
            state["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return state


def init_run(
    args: argparse.Namespace,
    *,
    job_type: str,
    config: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    resume_from: Path | None = None,
):
    """Start (or resume) a run from `add_wandb_args` flags; None when tracking is off.

    `run_dir` stores the run id in `wandb_run.json`, so a later process given the
    same directory as `resume_from` (train.py --resume) continues the same run.
    """
    if not getattr(args, "wandb_project", None):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "--wandb-project needs the wandb package in this environment (pip install wandb)"
        ) from exc

    loaded = load_dotenv()
    if loaded:
        logger.info("wandb: loaded %s from %s", ", ".join(loaded), DOTENV)
    run_id = getattr(args, "wandb_run_id", None)
    resume = "must" if run_id else None
    if run_id is None and resume_from is not None and (resume_from / RUN_FILE).is_file():
        run_id = json.loads((resume_from / RUN_FILE).read_text())["id"]
        resume = "allow"

    mode = os.environ.get("WANDB_MODE")
    if mode is None and not has_credentials():
        logger.warning(
            "wandb: no credentials (WANDB_API_KEY in %s, the environment, or `wandb login`); "
            "logging offline, upload later with `wandb sync`",
            DOTENV,
        )
        mode = "offline"

    context = {"git": git_state(), "host": host_state(), **(config or {})}
    if getattr(args, "wandb_run_id", None):
        # Attaching to someone else's run (e.g. eval into the training run):
        # nest everything so the run's own git/host/config stay as recorded.
        context = {f"{job_type}_context": context}
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        notes=args.wandb_notes,
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None,
        job_type=job_type,
        id=run_id,
        resume=resume,
        mode=mode,
        dir=str(run_dir) if run_dir is not None else None,
        config=context,
    )
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / RUN_FILE).write_text(
            json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": run.url}, indent=2)
        )
    logger.info("wandb run %s (%s)", run.id, run.url or "offline")
    return run
