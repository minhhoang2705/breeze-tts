#!/usr/bin/env python3
"""Held-out vs. train per-sample loss across nv_lora01 checkpoints (post-hoc eval loss).

nv_lora01 trained without --eval-manifest, so this reconstructs the curve it lacked:
scripts/per_sample_loss.py on data/val.jsonl (unseen sentences) and on a fixed 300-clip
sample of data/train.jsonl, for the init model (nv01-8000) and checkpoints 200..800.
Train loss falling while val loss stalls or rises is memorization.

Usage (on the login node; the forwards run inside job $JOB — required, no default, see
nv_lora/env.sh): JOB=22286 python3 heldout_loss.py
Writes eval/psl/<model>__<split>.jsonl and prints one summary line per (model, split).
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

JOB = os.environ.get("JOB") or sys.exit(
    "set JOB=<slurm job id>, e.g. JOB=22286 (check squeue --me for a free one) — "
    "no default here, shared-cluster job ids get repurposed, see nv_lora/env.sh")
REPO = Path(os.environ.get("REPO", "/mnt/data/users/minhth21/breeze-tts"))
D = REPO / "nv_lora"
OUT = D / "eval" / "psl"
PY = os.environ.get("PY_BREEZE", "/mnt/data/users/minhth21/conda_env/miniconda3/envs/breeze_tts/bin/python")
INIT = "/mnt/data/users/anhnct/breeze-tts/outputs/exported/nv01_fft-p2_from-vi33k_nvsyn68k_lr1e-5_8k__checkpoint-8000"
RUN = REPO / "outputs" / "nv_lora01_p1-r32_from-nv01-8k_vieneu-tc4.4k"
MODELS = {"init_8000": INIT, **{f"lora_{s}": str(RUN / f"checkpoint-{s}") for s in (200, 400, 600, 800)}}

OUT.mkdir(parents=True, exist_ok=True)
train = [l for l in (D / "data" / "train.jsonl").read_text().splitlines() if l.strip()]
sample = sorted(train, key=lambda l: hashlib.sha256(json.loads(l)["id"].encode()).hexdigest())[:300]
(OUT / "train_sample300.jsonl").write_text("\n".join(sample) + "\n")
splits = {"val": D / "data" / "val.jsonl", "train300": OUT / "train_sample300.jsonl"}

jobs = " ; ".join(
    f"{PY} scripts/per_sample_loss.py {INIT} {ckpt} {path} {D / 'codec_cache'} {OUT / f'{m}__{s}.jsonl'} --top 0"
    f" > {OUT / f'{m}__{s}.log'} 2>&1"
    for m, ckpt in MODELS.items() for s, path in splits.items()
)
subprocess.run(["srun", "--overlap", f"--jobid={JOB}", "bash", "-c", f"cd {REPO} && {jobs}"], check=True)

print("model split n mean_loss mean_backbone mean_depth")
for m in MODELS:
    for s in splits:
        rows = [json.loads(l) for l in (OUT / f"{m}__{s}.jsonl").read_text().splitlines()]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        print(m, s, len(rows), f"{mean('loss'):.4f} {mean('backbone_loss'):.4f} {mean('depth_decoder_loss'):.4f}")
