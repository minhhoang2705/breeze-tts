#!/usr/bin/env python3
"""Senvoice-style metrics for Breeze checkpoints, scored offline on H100.

The metric definitions follow anhnct's "Senvoice Evaluation" suite
(/mnt/data/users/anhnct/Higgs-tts-3-finetune/benchmark_portal_20260912). That suite uses
ElevenLabs Scribe v2 + Gemini for its ASR-based metrics, and H100 has no internet, so
these numbers use offline ASR instead. They are comparable across Breeze checkpoints, but
NOT with the Senvoice portal's numbers:

  VI-WER              sv_vi_200: corpus WER from tts-eval (ZipFormer-vi ASR, its own text
                      normalization), read from runs/<model>__sv_vi_200/summary.json.
  EN-WER              sv_en_200: Whisper large-v3 (language=en) -> corpus WER, using the
                      Senvoice offline EN normalization (NFKC, lowercase, digits -> English
                      words, n't contractions split, punctuation dropped).
  CS-Latin / CS-IPA / CS-Transliteration
                      the same 200 sentences written three ways (sv_cs_latin_200 /
                      sv_cs_ipa_200 / sv_cs_translit_200). Whisper large-v3 transcript vs
                      `raw_text` (the Latin spelling), Senvoice CS scoring: cs_wer = corpus
                      word errors / reference words; en_recall = English reference tokens
                      (from `english_tokens`) matched in the edit alignment / all English
                      tokens. `strict` and `suffix_tolerant` (English token missing only
                      -s/-es/-ed/-ies counts as a match) as in Senvoice.
  UTMOSv2             mean predicted MOS per evalset (utmos_score.py, 5-fold ensemble).

STOI (SQUIM-estimated, Senvoice OmniVoice exp5 mean 0.9958) is saturated for TTS output and
SIM-O (WavLM cosine to the prompt) duplicates tts-eval's spk_sim (ECAPA cosine to
ref_audio), so neither is computed here.

Usage:
  senvoice_metrics.py --eval-dir E --model NAME   # writes runs_senvoice/NAME/metrics.json
  senvoice_metrics.py --eval-dir E --table NAME... # prints a markdown table
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 1000
CS_FORMS = {"CS-Latin": "sv_cs_latin_200", "CS-IPA": "sv_cs_ipa_200", "CS-Transliteration": "sv_cs_translit_200"}
EN_SET = "sv_en_200"
VI_SET = "sv_vi_200"
UTMOS_SETS = ("vi_clone_300", VI_SET, EN_SET, *CS_FORMS.values())
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})

# --- Vietnamese number reading (copied from Senvoice benchmark_vi_wer/scripts/vi_num.py) ---
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = [(10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn")]
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
PERCENT = re.compile(r"(?<!\w)(\d+(?:[.,]\d{1,2})?)\s*%")
CLOCK = re.compile(r"(?<!\w)([01]?\d|2[0-3])\s*(?:[hH]|:)\s*([0-5]\d)(?!\d)")
DECIMAL = re.compile(r"(?<!\d)(\d+)[,.](\d{1,2})(?!\d)")
NUMBER = re.compile(r"\d+")


def _under_thousand(number: int, full: bool = False) -> list[str]:
    words = []
    hundreds, remainder = divmod(number, 100)
    tens, units = divmod(remainder, 10)
    if hundreds or full:
        words += [DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units and (hundreds or full):
            words += ["lẻ", DIGITS[units]]
        elif units:
            words.append(DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(DIGITS[units])
    else:
        words += [DIGITS[tens], "mươi"]
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(DIGITS[units])
    return words


def read_int(number: int) -> str:
    if number == 0:
        return "không"
    words: list[str] = []
    first = True
    for value, name in SCALES:
        if number >= value:
            quotient, number = divmod(number, value)
            words += read_int(quotient).split() if quotient >= 1000 else _under_thousand(quotient, not first)
            words.append(name)
            first = False
    if number:
        words += _under_thousand(number, not first)
    return " ".join(words)


def _read_digits(digits: str) -> str:
    if len(digits) > 1 and digits[0] == "0":
        return " ".join(DIGITS[int(c)] for c in digits)
    return read_int(int(digits))


def _read_decimal(value: str) -> str:
    integer, fraction = re.split(r"[,.]", value, maxsplit=1)
    return f"{_read_digits(integer)} phẩy {_read_digits(fraction)}"


def vi_numbers(text: str, max_digits: int = 12) -> str:
    text = THOUSANDS_SEPARATOR.sub("", text)
    text = PERCENT.sub(lambda m: f"{_read_decimal(m.group(1)) if re.search(r'[,.]', m.group(1)) else _read_digits(m.group(1))} phần trăm", text)
    text = CLOCK.sub(lambda m: f"{read_int(int(m.group(1)))} giờ {_read_digits(m.group(2))}", text)
    text = DECIMAL.sub(lambda m: _read_decimal(m.group(0)), text)

    def replace(m: re.Match) -> str:
        digits = m.group(0)
        if len(digits) > max_digits or (len(digits) > 1 and digits[0] == "0"):
            return " ".join(DIGITS[int(c)] for c in digits)
        return read_int(int(digits))

    return NUMBER.sub(replace, text)


# --- CS tokenization (Senvoice benchmark_cs/scripts/bench_cs.py) ---
CS_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def cs_word(word: str) -> str:
    return unicodedata.normalize("NFKC", word).translate(APOSTROPHES).lower()


def cs_words(text: str) -> list[str]:
    return [cs_word(w) for w in CS_WORD_RE.findall(vi_numbers(text))]


def cs_languages(raw_text: str, english_tokens: list[str]) -> tuple[list[str], list[str]]:
    """Reference words of `raw_text` and a vi/en label per word: each English token is
    matched to the next occurrence in order, as Senvoice's build_independent_label does."""
    ref = cs_words(raw_text)
    languages = ["vi"] * len(ref)
    cursor = 0
    for token in map(cs_word, english_tokens):
        position = ref.index(token, cursor)  # ValueError = broken dataset row
        languages[position] = "en"
        cursor = position + 1
    return ref, languages


def missing_suffix(reference: str, prediction: str) -> bool:
    if len(prediction) < 3:
        return False
    candidates = set()
    if reference.endswith("s"):
        candidates.add(reference[:-1])
        if reference.endswith("ies"):
            candidates.add(reference[:-3] + "y")
        if reference.endswith("es"):
            candidates.add(reference[:-2])
    if reference.endswith("ed"):
        candidates.update((reference[:-2], reference[:-1]))
        if len(reference) >= 5 and reference[-3] == reference[-4]:
            candidates.add(reference[:-3])
        if reference.endswith("ied"):
            candidates.add(reference[:-3] + "y")
    return prediction in candidates


def _equivalent(ref: str, hyp: str, language: str, suffix_tolerant: bool) -> bool:
    return ref == hyp or (suffix_tolerant and language == "en" and missing_suffix(ref, hyp))


def align(ref: list[str], hyp: list[str], languages: list[str], suffix_tolerant: bool) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment with Senvoice's backtrace order (match, substitution,
    deletion, insertion)."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        dist[r][0] = r
    for c in range(cols):
        dist[0][c] = c
    for r in range(1, rows):
        for c in range(1, cols):
            sub = 0 if _equivalent(ref[r - 1], hyp[c - 1], languages[r - 1], suffix_tolerant) else 1
            dist[r][c] = min(dist[r - 1][c] + 1, dist[r][c - 1] + 1, dist[r - 1][c - 1] + sub)
    ops: list[tuple[str, int | None, int | None]] = []
    r, c = len(ref), len(hyp)
    while r or c:
        if r and c and _equivalent(ref[r - 1], hyp[c - 1], languages[r - 1], suffix_tolerant) and dist[r][c] == dist[r - 1][c - 1]:
            ops.append(("match" if ref[r - 1] == hyp[c - 1] else "suffix_match", r - 1, c - 1))
            r, c = r - 1, c - 1
        elif r and c and dist[r][c] == dist[r - 1][c - 1] + 1:
            ops.append(("substitution", r - 1, c - 1))
            r, c = r - 1, c - 1
        elif r and dist[r][c] == dist[r - 1][c] + 1:
            ops.append(("deletion", r - 1, None))
            r -= 1
        else:
            ops.append(("insertion", None, c - 1))
            c -= 1
    ops.reverse()
    return ops


def cs_counts(ref: list[str], hyp: list[str], languages: list[str], suffix_tolerant: bool) -> dict:
    ops = align(ref, hyp, languages, suffix_tolerant)
    errors = sum(op in ("substitution", "deletion", "insertion") for op, _, _ in ops)
    en_hits = sum(1 for op, ri, _ in ops if ri is not None and languages[ri] == "en" and op in ("match", "suffix_match"))
    return {"errors": errors, "ref_words": len(ref), "en_hits": en_hits, "en_tokens": languages.count("en"),
            "en_misses": [ref[ri] for op, ri, _ in ops if ri is not None and languages[ri] == "en" and op not in ("match", "suffix_match")]}


# --- English WER tokenization (Senvoice benchmark_en_wer/scripts/score_en_wer.py) ---
EN_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
NEGATIVE_CONTRACTIONS = {
    "aren": "are", "isn": "is", "wasn": "was", "weren": "were", "don": "do", "doesn": "does",
    "didn": "did", "can": "can", "couldn": "could", "won": "will", "wouldn": "would",
    "shouldn": "should", "hasn": "has", "haven": "have", "hadn": "had", "mustn": "must",
}


def integer_words(value: str | int) -> list[str]:
    value = int(value)
    if value < 10:
        return [ONES[value]]
    if value < 20:
        return [TEENS[value - 10]]
    if value < 100:
        return [TENS[value // 10]] + ([] if value % 10 == 0 else [ONES[value % 10]])
    if value < 1000:
        return [ONES[value // 100], "hundred"] + ([] if value % 100 == 0 else integer_words(value % 100))
    for divisor, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= divisor:
            return integer_words(value // divisor) + [name] + ([] if value % divisor == 0 else integer_words(value % divisor))
    raise ValueError(f"unsupported integer: {value}")


def en_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).translate(APOSTROPHES).lower()
    normalized = re.sub(r"\d+", lambda m: " " + " ".join(integer_words(m.group())) + " ", normalized)
    tokens = EN_WORD_RE.findall(normalized)
    result, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        stem = token[:-2] if token.endswith("'t") else token if i + 1 < len(tokens) and tokens[i + 1] == "t" else None
        if stem in NEGATIVE_CONTRACTIONS:
            result.extend((NEGATIVE_CONTRACTIONS[stem], "not"))
            i += 1 if token.endswith("'t") else 2
        else:
            result.append(token)
            i += 1
    return result


def edit_errors(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for r in range(1, len(ref) + 1):
        cur = [r] + [0] * len(hyp)
        for c in range(1, len(hyp) + 1):
            cur[c] = min(prev[c] + 1, cur[c - 1] + 1, prev[c - 1] + (ref[r - 1] != hyp[c - 1]))
        prev = cur
    return prev[-1]


# --- aggregation ---
def ratio_ci(num: np.ndarray, den: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    n = len(num)
    draws = [num[i].sum() / den[i].sum() for i in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transcripts(path: Path) -> dict[str, str | None]:
    """id -> Whisper text; None for a missing/unusable wav (scored as an empty transcript)."""
    return {row["id"]: row["hyp"] for row in read_jsonl(path)}


def score_en(evalset: list[dict], hyps: dict) -> dict:
    errors, refs, rows = [], [], []
    for item in evalset:
        ref, hyp = en_words(item["text"]), en_words(hyps.get(item["id"]) or "")
        errors.append(edit_errors(ref, hyp))
        refs.append(len(ref))
        rows.append({"id": item["id"], "ref": " ".join(ref), "hyp": " ".join(hyp), "errors": errors[-1]})
    e, n = np.array(errors, float), np.array(refs, float)
    return {"wer": round(float(e.sum() / n.sum()), 6), "ci95": ratio_ci(e, n), "n": len(evalset),
            "n_missing": sum(hyps.get(i["id"]) is None for i in evalset), "per_utt": rows}


def score_cs(evalset: list[dict], hyps: dict) -> dict:
    out: dict = {"n": len(evalset), "n_missing": sum(hyps.get(i["id"]) is None for i in evalset)}
    per_utt = []
    for policy, tolerant in (("strict", False), ("suffix_tolerant", True)):
        counts = []
        for item in evalset:
            ref, languages = cs_languages(item["raw_text"], item["english_tokens"])
            counts.append(cs_counts(ref, cs_words(hyps.get(item["id"]) or ""), languages, tolerant))
            if not tolerant:
                per_utt.append({"id": item["id"], "hyp": hyps.get(item["id"]), "errors": counts[-1]["errors"],
                                "en_misses": counts[-1]["en_misses"]})
        err = np.array([c["errors"] for c in counts], float)
        ref_n = np.array([c["ref_words"] for c in counts], float)
        hit = np.array([c["en_hits"] for c in counts], float)
        en_n = np.array([c["en_tokens"] for c in counts], float)
        out[policy] = {
            "cs_wer": round(float(err.sum() / ref_n.sum()), 6), "cs_wer_ci95": ratio_ci(err, ref_n),
            "en_recall": round(float(hit.sum() / en_n.sum()), 6), "en_recall_ci95": ratio_ci(hit, en_n),
            "reference_words": int(ref_n.sum()), "english_tokens": int(en_n.sum()),
        }
    out["per_utt"] = per_utt
    return out


def score_model(eval_dir: Path, model: str) -> dict:
    out_dir = eval_dir / "runs_senvoice" / model
    metrics: dict = {"model": model}
    vi_summary = eval_dir / "runs" / f"{model}__{VI_SET}" / "summary.json"
    if vi_summary.exists():
        m = json.loads(vi_summary.read_text())["metrics"]
        lo, hi = m["one_minus_wer"]["ci95"]
        metrics["VI-WER"] = {"wer": round(m["wer"], 6), "ci95": [round(1 - hi, 6), round(1 - lo, 6)],
                             "n": m["n"], "n_missing": m["n_missing"], "asr": "ZipFormer-vi (tts-eval)"}
    asr_en = out_dir / f"asr_{EN_SET}.jsonl"
    if asr_en.exists():
        metrics["EN-WER"] = {**score_en(read_jsonl(eval_dir / "evalsets" / f"{EN_SET}.jsonl"), transcripts(asr_en)),
                             "asr": "whisper-large-v3 language=en"}
    for name, evalset in CS_FORMS.items():
        asr = out_dir / f"asr_{evalset}.jsonl"
        if asr.exists():
            metrics[name] = {**score_cs(read_jsonl(eval_dir / "evalsets" / f"{evalset}.jsonl"), transcripts(asr)),
                             "asr": "whisper-large-v3 language=vi"}
    utmos = {}
    for evalset in UTMOS_SETS:
        utmos_path = out_dir / f"utmosv2_{evalset}.json"
        if utmos_path.exists():
            data = json.loads(utmos_path.read_text())
            utmos[evalset] = {k: v for k, v in data.items() if k != "per_utt"}
    if utmos:
        metrics["UTMOSv2"] = utmos
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    return metrics


def table(eval_dir: Path, models: list[str]) -> str:
    head = ["model", "VI-WER", "EN-WER", *(f"{n} WER / EN-recall" for n in CS_FORMS), "UTMOSv2 (vi_clone_300)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model in models:
        m = json.loads((eval_dir / "runs_senvoice" / model / "metrics.json").read_text())
        cell = lambda key, f: f(m[key]) if key in m else "n/a"  # noqa: E731
        row = [model, cell("VI-WER", lambda v: f"{v['wer']:.4f}"), cell("EN-WER", lambda v: f"{v['wer']:.4f}")]
        row += [cell(n, lambda v: f"{v['strict']['cs_wer']:.4f} / {v['strict']['en_recall']:.4f}") for n in CS_FORMS]
        u = m.get("UTMOSv2", {}).get("vi_clone_300")
        row.append(f"{u['mean']:.3f}" if u else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def self_test() -> None:
    assert cs_words("Năm 2025 có 25% list") == ["năm", "hai", "nghìn", "không", "trăm", "hai", "mươi", "lăm", "có",
                                                "hai", "mươi", "lăm", "phần", "trăm", "list"]
    ref, lang = cs_languages("Sau đây là list và reign years", ["list", "reign", "years"])
    assert lang == ["vi", "vi", "vi", "en", "vi", "en", "en"]
    strict = cs_counts(ref, cs_words("sau đây là list và reign year"), lang, False)
    tolerant = cs_counts(ref, cs_words("sau đây là list và reign year"), lang, True)
    assert (strict["errors"], strict["en_hits"]) == (1, 2) and (tolerant["errors"], tolerant["en_hits"]) == (0, 3)
    assert missing_suffix("commandments", "commandment") and not missing_suffix("is", "i")
    assert en_words("Chapter 21, aren't") == ["chapter", "twenty", "one", "are", "not"]
    assert edit_errors("a b c".split(), "a x c d".split()) == 2
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--table", nargs="+")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.table:
        print(table(args.eval_dir, args.table))
    else:
        m = score_model(args.eval_dir, args.model)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_utt"} if isinstance(v, dict) else v
                          for k, v in m.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
