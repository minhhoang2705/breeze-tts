#!/usr/bin/env python3
"""NV Placement Accuracy (NVPA) from a nv_gemini_audit run, per the metric definition:
"An NV event is considered correctly realized only when the required NV type is present
at the intended position." Range 0-1, higher better.

Computed strictly from results.jsonl's raw per-marker `heard`/`label` (the Gemini judge's
blind listen at the exact <NVk> text position, see gemini_audit.py PROMPT_MARKED) plus
tags.json's `accept` list (the official VLSP label for that tag, not the `review`
data-compromise list) — NOT decide.py's row-level `verdict`, which also folds in the
whole-sentence transcript gate. A reading slip elsewhere in the sentence is a 1-WER
problem, not an NV-placement problem, so it must not zero out NVPA here. (An actual
NV-Leak, where the tag is spoken as text instead of realized as a sound, already shows up
as heard=false at that marker, since the judge is asked only what it hears AT that exact
position, not what the transcript says — so leaks are still correctly caught.)

NVPA(row) = mean over that row's markers of 1[heard and label in accept[expected_tag]].
Rows with zero markers (errored / not audited) are excluded, not scored as 0.

Usage: compute_nvpa.py RUN_DIR [--tags tags.json] [--out RUN_DIR/nvpa.jsonl]
Writes one {id, source, nvpa, n_tags, n_hit, markers} row per audited, tagged row, plus an
aggregate line to stderr (mean NVPA, n rows). `markers` keeps each tag's raw judgement —
{expected, heard ("none" when nothing was heard at that position), hit, naturalness} — so
downstream analysis can build the expected-tag x heard-label confusion matrix (e.g.
[sniff] heard as breath) without re-reading results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--tags", type=Path, default=Path(__file__).parent.parent.parent / "nv_gemini_audit" / "tags.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or (args.run / "nvpa.jsonl")

    policy = json.loads(args.tags.read_text(encoding="utf-8"))
    accept = {t: set(v["accept"]) for t, v in policy.items() if not t.startswith("_")}

    input_rows = {json.loads(l)["id"]: json.loads(l)
                  for l in (args.run / "input.jsonl").open(encoding="utf-8") if l.strip()}

    results: dict[str, dict] = {}
    for line in (args.run / "results.jsonl").open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            results[rec["id"]] = rec  # last write wins — results.jsonl is append-only across
                                       # resumed/--retry-errors reruns, same convention decide.py uses

    rows = []
    for rid, res in results.items():
        row = input_rows.get(rid)
        if row is None or not row.get("tags") or res.get("status") != "ok":
            continue  # untagged baseline row, or not usably audited — excluded, not scored 0
        markers = [m for m in res["markers"] if m.get("expected_tag")]
        if not markers:
            continue
        detail = [{"expected": m["expected_tag"], "heard": m["label"] if m["heard"] else "none",
                   "hit": bool(m["heard"] and m["label"] in accept.get(m["expected_tag"], set())),
                   "naturalness": m.get("naturalness")} for m in markers]
        n_hit = sum(d["hit"] for d in detail)
        rows.append({"id": rid, "source": row.get("source"), "nvpa": n_hit / len(markers),
                      "n_tags": len(markers), "n_hit": n_hit, "markers": detail})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    mean_nvpa = sum(r["nvpa"] for r in rows) / len(rows) if rows else float("nan")
    print(f"NVPA: {mean_nvpa:.4f} over {len(rows)} rows -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
