#!/usr/bin/env python3
"""DPO signal probe: for each prompt, do same-prompt/different-seed samples differ enough
to form a DPO chosen/rejected pair? Groups nv_lora/eval's dpo_probe_* per-sample scores
(tts-eval per_utt.jsonl for 1-WER/spk_sim/pMOS/duration, compute_nvpa.py's nvpa.jsonl for
tag placement) by prompt across seeds.

Two reports:

1. Spread per evalset — a prompt is "pairable" when at least one axis has a real spread:
  nvpa        any seed differs from any other (binary per seed; tagged evalsets only)
  wer         max-min word_errors across seeds >= 1 (one whole word, not a rounding wobble)
  spk_sim     max-min spk_sim across seeds >= 0.03 (nv_lora01's observed real effect size
              for a paired-CI-significant spk_sim shift was 0.013-0.026; 0.03 is a
              deliberately conservative floor above that)
  runaway     any seed's seconds/word is >= 2x (or <= 1/2x) that prompt's own median
              across its seeds (the documented no-EOS failure mode)
  Prompts with fewer than 2 non-missing seeds are excluded from the pairable rate and
  reported separately, not silently dropped.

2. Tag discrimination per tagged evalset (the 4-tag probe: tc/sn/br/la_clone_100 share
  sentences, voices and positions, so only the tag differs) — what the blind judge heard
  at the tag position, as an expected-tag x heard-label confusion matrix, and per prompt
  across seeds:
  all-correct  every seed realised the required sound — nothing for DPO to fix
  all-wrong    no seed did — DPO has no chosen sample here; needs data/SFT, not RL
  mixed        some seeds right, some wrong — a direct DPO chosen/rejected pair; the
               wrong seeds' labels are exactly what DPO would push the model away from
  plus hit rate by tag position and mean judge naturalness (1-5; the judge's prompt scores
  faint/swallowed sounds 1-2, so it's the closest available proxy for "clear vs light").

Usage: dpo_probe_analyze_spread.py RUNS_DIR NVPA_JSONL [NVPA_JSONL ...]
         [--seeds 42 43 ...] [--evalsets tc_clone_100 ...] [--out summary.md]
  RUNS_DIR    directory containing seed_<seed>__<evalset>/per_utt.jsonl
  NVPA_JSONL  compute_nvpa.py output(s); ids "seed_<seed>__<evalset>__<prompt_id>",
              source "seed_<seed>|<evalset>" (several audit runs may be combined)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_EVALSETS = ("tc_clone_100", "sn_clone_100", "br_clone_100", "la_clone_100", "vi_probe_100")
WER_WORD_THRESHOLD = 1
SPK_SIM_THRESHOLD = 0.03
RUNAWAY_RATIO = 2.0
HEARD_LABELS = ("laughter", "breath", "gasp", "sigh", "cough", "throat_clear", "sniff", "other", "none")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_per_utt(runs_dir: Path, seeds: list[int], evalset: str) -> dict[int, dict[str, dict]]:
    """-> {seed: {prompt_id: per_utt_row}}, missing run dirs reported to stderr, not silently skipped."""
    by_seed = {}
    for seed in seeds:
        path = runs_dir / f"seed_{seed}__{evalset}" / "per_utt.jsonl"
        rows = read_jsonl(path)
        if not rows:
            print(f"WARNING: no per_utt rows for seed {seed} {evalset} ({path})", file=sys.stderr)
        by_seed[seed] = {r["id"]: r for r in rows}
    return by_seed


def load_nvpa(paths: list[Path], seeds: list[int]) -> dict[str, dict[int, dict[str, dict]]]:
    """-> {evalset: {seed: {prompt_id: nvpa_row}}} from merged ids "seed_<s>__<evalset>__<id>"."""
    out: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        if not path.is_file():
            print(f"WARNING: NVPA file not found: {path}", file=sys.stderr)
        for row in read_jsonl(path):
            model, _, evalset = row.get("source", "").partition("|")
            if not model.startswith("seed_") or not evalset:
                continue
            try:
                seed = int(model.removeprefix("seed_"))
            except ValueError:
                continue
            prefix = f"{model}__{evalset}__"
            if seed not in seeds or not row["id"].startswith(prefix):
                continue
            out[evalset][seed][row["id"].removeprefix(prefix)] = row
    return out


def seconds_per_word(row: dict) -> float | None:
    if row.get("missing") or not row.get("ref_words"):
        return None
    return row["duration_sec"] / row["ref_words"]


def analyze_evalset(per_utt: dict[int, dict[str, dict]],
                     nvpa: dict[int, dict[str, dict]] | None) -> tuple[list[dict], dict]:
    prompt_ids: set[str] = set()
    for rows in per_utt.values():
        prompt_ids |= rows.keys()

    results = []
    for pid in sorted(prompt_ids):
        samples = []
        for seed, rows in per_utt.items():
            row = rows.get(pid)
            if row is None or row.get("missing"):
                continue
            nv = nvpa.get(seed, {}).get(pid) if nvpa is not None else None
            samples.append({
                "seed": seed, "word_errors": row["word_errors"], "spk_sim": row["spk_sim"],
                "dnsmos_ovrl": row["dnsmos_ovrl"], "seconds_per_word": seconds_per_word(row),
                "nvpa": nv["nvpa"] if nv else None,
            })
        entry = {"id": pid, "n_seeds": len(samples), "pairable": False, "signals": []}
        if len(samples) >= 2:
            we = [s["word_errors"] for s in samples]
            if max(we) - min(we) >= WER_WORD_THRESHOLD:
                entry["signals"].append("wer")
            sims = [s["spk_sim"] for s in samples]
            if max(sims) - min(sims) >= SPK_SIM_THRESHOLD:
                entry["signals"].append("spk_sim")
            spw_vals = [s["seconds_per_word"] for s in samples if s["seconds_per_word"]]
            if len(spw_vals) >= 2:
                med = statistics.median(spw_vals)
                if med > 0 and any(w >= med * RUNAWAY_RATIO or w <= med / RUNAWAY_RATIO for w in spw_vals):
                    entry["signals"].append("runaway")
            nvpa_vals = [s["nvpa"] for s in samples if s["nvpa"] is not None]
            if len(nvpa_vals) >= 2 and len(set(nvpa_vals)) > 1:
                entry["signals"].append("nvpa")
            entry["pairable"] = bool(entry["signals"])
        entry["samples"] = samples
        results.append(entry)

    scored = [r for r in results if r["n_seeds"] >= 2]
    stats = {
        "n_prompts": len(results),
        "n_scored": len(scored),
        "n_too_few_seeds": len(results) - len(scored),
        "n_pairable": sum(1 for r in scored if r["pairable"]),
        "by_signal": {sig: sum(1 for r in scored if sig in r["signals"])
                      for sig in ("wer", "spk_sim", "runaway", "nvpa")},
    }
    return results, stats


def discrimination(nvpa: dict[int, dict[str, dict]], positions: dict[str, str]) -> dict:
    """Tag-confusion view of one tagged evalset's NVPA rows (single-tag rows)."""
    heard = Counter()
    by_pos: dict[str, list[bool]] = defaultdict(list)
    nat_hit, nat_miss = [], []
    per_prompt: dict[str, list[dict]] = defaultdict(list)
    expected = Counter()
    for seed, rows in nvpa.items():
        for pid, row in rows.items():
            for m in row["markers"]:
                expected[m["expected"]] += 1
                heard[m["heard"]] += 1
                by_pos[positions.get(pid, "?")].append(m["hit"])
                if m.get("naturalness") is not None:
                    (nat_hit if m["hit"] else nat_miss).append(m["naturalness"])
                per_prompt[pid].append(m)
    classes = Counter()
    wrong_in_mixed, wrong_in_all_wrong = Counter(), Counter()
    for marks in per_prompt.values():
        if len(marks) < 2:
            classes["too_few_seeds"] += 1
            continue
        hits = [m["hit"] for m in marks]
        if all(hits):
            classes["all_correct"] += 1
        elif not any(hits):
            classes["all_wrong"] += 1
            wrong_in_all_wrong.update(m["heard"] for m in marks)
        else:
            classes["mixed"] += 1
            wrong_in_mixed.update(m["heard"] for m in marks if not m["hit"])
    n = sum(heard.values())
    return {
        "expected": expected.most_common(1)[0][0] if expected else "?",
        "n": n, "hit_rate": sum(sum(v) for v in by_pos.values()) / n if n else float("nan"),
        "heard": heard, "classes": classes,
        "wrong_in_mixed": wrong_in_mixed, "wrong_in_all_wrong": wrong_in_all_wrong,
        "by_pos": {p: (sum(v) / len(v), len(v)) for p, v in sorted(by_pos.items())},
        "nat_hit": statistics.mean(nat_hit) if nat_hit else None,
        "nat_miss": statistics.mean(nat_miss) if nat_miss else None,
    }


def fmt_counter(c: Counter, top: int = 4) -> str:
    return ", ".join(f"{k} {v}" for k, v in c.most_common(top)) or "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("nvpa_jsonl", type=Path, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--evalsets", nargs="+", default=list(DEFAULT_EVALSETS))
    ap.add_argument("--evalsets-dir", type=Path, default=None,
                    help="where <evalset>.jsonl live (for tag positions); default RUNS_DIR/../evalsets")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    args = ap.parse_args()
    out_md = args.out or (args.runs_dir.parent / "spread_summary.md")
    out_jsonl = args.out_jsonl or (args.runs_dir.parent / "spread_detail.jsonl")
    evalsets_dir = args.evalsets_dir or (args.runs_dir.parent / "evalsets")

    nvpa_all = load_nvpa(args.nvpa_jsonl, args.seeds)
    lines = ["# DPO signal probe — same-prompt / different-seed spread", "",
             f"Seeds: {' '.join(map(str, args.seeds))}", ""]
    disc_rows = []
    all_results = []
    for evalset in args.evalsets:
        per_utt = load_per_utt(args.runs_dir, args.seeds, evalset)
        nvpa = nvpa_all.get(evalset)
        results, stats = analyze_evalset(per_utt, nvpa)
        for r in results:
            r["evalset"] = evalset
        all_results.extend(results)

        lines.append(f"## {evalset}")
        lines.append("")
        if stats["n_scored"]:
            lines.append(f"{stats['n_prompts']} prompts · {stats['n_scored']} with >=2 seeds "
                         f"({stats['n_too_few_seeds']} excluded, too few seeds) · "
                         f"**{stats['n_pairable']}/{stats['n_scored']} pairable** "
                         f"({stats['n_pairable'] / stats['n_scored'] * 100:.1f}%)")
        else:
            lines.append("no scored prompts")
        lines.append("")
        lines.append("| signal | prompts flagged |")
        lines.append("|---|---|")
        for sig, count in stats["by_signal"].items():
            if nvpa is None and sig == "nvpa":
                continue
            lines.append(f"| {sig} | {count} |")
        lines.append("")

        if nvpa:
            positions = {r["id"]: r.get("position", "?") for r in read_jsonl(evalsets_dir / f"{evalset}.jsonl")}
            disc_rows.append((evalset, discrimination(nvpa, positions)))

    if disc_rows:
        lines += ["# Tag discrimination (blind judge, what was heard at the tag position)", ""]
        lines.append("| expected tag | n | hit rate | " + " | ".join(HEARD_LABELS) + " |")
        lines.append("|---|---|---|" + "---|" * len(HEARD_LABELS))
        for evalset, d in disc_rows:
            cells = " | ".join(str(d["heard"].get(lb, 0)) for lb in HEARD_LABELS)
            lines.append(f"| {d['expected']} ({evalset}) | {d['n']} | {d['hit_rate']:.1%} | {cells} |")
        lines.append("")
        lines.append("| expected tag | prompts all-correct | **mixed (DPO pairs)** | all-wrong (needs data) "
                     "| <2 seeds judged | wrong labels in mixed | wrong labels in all-wrong "
                     "| hit rate by position | naturalness hit / miss |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for evalset, d in disc_rows:
            c = d["classes"]
            pos = ", ".join(f"{p} {rate:.0%} (n={n})" for p, (rate, n) in d["by_pos"].items())
            nat = " / ".join(f"{v:.2f}" if v is not None else "—" for v in (d["nat_hit"], d["nat_miss"]))
            lines.append(f"| {d['expected']} | {c['all_correct']} | **{c['mixed']}** | {c['all_wrong']} "
                         f"| {c['too_few_seeds']} | {fmt_counter(d['wrong_in_mixed'])} "
                         f"| {fmt_counter(d['wrong_in_all_wrong'])} | {pos} | {nat} |")
        lines.append("")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_results))
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md} and {out_jsonl}", file=sys.stderr)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
