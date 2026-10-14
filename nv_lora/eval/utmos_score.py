#!/usr/bin/env python3
"""UTMOSv2 predicted MOS for a directory of generated wavs, mirroring Senvoice's protocol
(anhnct's benchmark_utmosv2/scripts/score_utmosv2.py): 5-fold ensemble (fold{0..4}_s42_
best_model.pth, config=fusion_stage3, predict_dataset=sarulab, seed 42,
remove_silent_section=True -- the Senvoice/sarulab default, kept here for comparability
with Senvoice's own OmniVoice numbers), 1 repetition. Simplified from Senvoice's version:
one GPU, folds run sequentially in-process (no parallel-fold subprocess orchestration --
not needed at this evalset size, 200-300 utterances), no OOM auto-retry.

Run in an env with torch+torchvision+torchaudio+transformers+timm+librosa (breeze312 on
H100 has all five) and PYTHONPATH pointing at the vendored utmosv2 package (see
nv_lora/eval/score_senvoice.sh for the exact invocation); weights are local (no internet
on H100) at --checkpoint-dir, default /mnt/data/users/minhth21/models/utmosv2.

Usage:
  utmos_score.py --audio-dir DIR --output OUT.json [--checkpoint-dir DIR] [--folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 1000


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ci95(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    n = len(values)
    means = [values[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/mnt/data/users/minhth21/models/utmosv2"))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predict-dataset", default="sarulab")
    args = parser.parse_args()

    import torch
    import utmosv2

    files = sorted(args.audio_dir.glob("*.wav"))
    if not files:
        raise SystemExit(f"no wavs in {args.audio_dir}")
    folds = [int(f) for f in args.folds.split(",") if f.strip()]

    scores: dict[str, list[float]] = {f.stem: [] for f in files}
    for fold in folds:
        checkpoint = args.checkpoint_dir / f"fold{fold}_s42_best_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seed_everything(args.seed + fold * 1000)
        print(f"UTMOS fold={fold} load {checkpoint}", flush=True)
        model = utmosv2.create_model(
            pretrained=True, config="fusion_stage3", fold=fold, checkpoint_path=checkpoint,
            seed=args.seed, device=args.device,
        )
        predictions = model.predict(
            input_dir=args.audio_dir, predict_dataset=args.predict_dataset, device=args.device,
            num_workers=args.num_workers, batch_size=args.batch_size, num_repetitions=1,
            remove_silent_section=True, verbose=True,
        )
        for prediction in predictions:
            scores[Path(str(prediction["file_path"])).stem].append(float(prediction["predicted_mos"]))
        del model
        torch.cuda.empty_cache()

    per_utt = [{"id": stem, "utmosv2_mean": round(float(np.mean(v)), 6), "fold_scores": v}
               for stem, v in scores.items() if v]
    missing = [stem for stem, v in scores.items() if not v]
    values = np.array([row["utmosv2_mean"] for row in per_utt])
    result = {
        "mean": round(float(values.mean()), 6), "ci95": ci95(values), "n": len(per_utt),
        "n_missing": len(missing), "folds": folds, "config": "fusion_stage3",
        "predict_dataset": args.predict_dataset, "remove_silent_section": True, "per_utt": per_utt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"UTMOSv2 mean={result['mean']:.4f} ci95={result['ci95']} n={result['n']} missing={result['n_missing']}")


if __name__ == "__main__":
    main()
