"""Evaluation: EM and F1 on validation data, and the no-answer threshold.

Owner: Mauricio. TODO:
- Compute EM and F1, split by answerable and unanswerable questions.
- Tune the no-answer threshold on validation data and record the chosen value.
- Run public_sample.json through the service and compare with the reference answers.

Usage (from the repo root, with the model in models/distilbert-full):
    python evaluation/evaluate_model.py --out evaluation/threshold_sweep.json

The model is run once over the validation split. For each question we keep the best span and
gap = null_logit - span_logit. The service answers "no answer" when gap > threshold, so every
threshold can be scored afterwards without running the model again.
"""

import argparse
import collections
import json
import os
import re
import string
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The hidden test set is about 25% unanswerable; SQuAD 2.0 validation is about 50%.
UNANSWERABLE_WEIGHT = 0.25
THRESHOLDS = np.round(np.arange(-8.0, 8.0001, 0.05), 3) + 0.0  # + 0.0 turns -0.0 into 0.0


def normalize_answer(text):
    """Official SQuAD normalization: lowercase, drop punctuation, articles and extra whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction, reference):
    pred_tokens, ref_tokens = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def run_model(examples):
    """Return one (id, span_text, gap, references) row per example, using the service's own scoring."""
    os.environ.setdefault("QA_TEAM", "evaluation")
    from service import app as qa

    qa._load_model()
    if qa._model is None:
        raise SystemExit("The model failed to load; check models/distilbert-full.")

    rows = []
    for i, ex in enumerate(examples, start=1):
        text, _, span_logit, _, null_logit, _ = qa._scan(ex["context"], ex["question"])
        gap = null_logit - span_logit if text is not None else float("inf")
        rows.append((ex["id"], text, gap, ex["answers"]["text"]))
        if i % 500 == 0:
            print(f"{i}/{len(examples)}", flush=True)
    return rows


def span_scores(text, references):
    """EM and F1 (0-1) of a predicted span against the non-empty references."""
    refs = [r for r in references if normalize_answer(r)]
    if text is None or not refs:
        return 0.0, 0.0
    em = max(float(normalize_answer(text) == normalize_answer(r)) for r in refs)
    f1 = max(f1_score(text, r) for r in refs)
    return em, f1


def sweep(rows, thresholds=THRESHOLDS):
    """Score every threshold. Unanswerable questions count 1 when the answer is null (EM = F1)."""
    gaps = np.array([r[2] for r in rows])
    is_answerable = np.array([any(normalize_answer(t) for t in r[3]) for r in rows])
    scores = [span_scores(r[1], r[3]) for r in rows]
    span_em = np.array([s[0] for s in scores])
    span_f1 = np.array([s[1] for s in scores])

    results = []
    for tau in thresholds:
        null = gaps > tau
        has_em = np.where(null, 0.0, span_em)[is_answerable]
        has_f1 = np.where(null, 0.0, span_f1)[is_answerable]
        no_ok = null[~is_answerable].astype(float)
        total = len(rows)
        results.append({
            "threshold": float(tau),
            "exact": 100 * (has_em.sum() + no_ok.sum()) / total,
            "f1": 100 * (has_f1.sum() + no_ok.sum()) / total,
            "HasAns_exact": 100 * has_em.mean(),
            "HasAns_f1": 100 * has_f1.mean(),
            "NoAns_exact": 100 * no_ok.mean(),
            "weighted_f1": 100 * ((1 - UNANSWERABLE_WEIGHT) * has_f1.mean() + UNANSWERABLE_WEIGHT * no_ok.mean()),
        })
    return results, int(is_answerable.sum()), int((~is_answerable).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_examples", type=int, default=None, help="Quick test on the first N examples.")
    parser.add_argument("--cache", default=None, help="JSON file to save/reuse the per-question model output.")
    parser.add_argument("--out", default=None, help="Where to write the sweep results (JSON).")
    args = parser.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        rows = [tuple(r) for r in json.loads(cache.read_text(encoding="utf-8"))]
        print(f"Loaded {len(rows)} rows from {cache}")
    else:
        from datasets import load_dataset

        dataset = load_dataset("rajpurkar/squad_v2", split=args.split)
        if args.max_examples:
            dataset = dataset.select(range(args.max_examples))
        rows = run_model([dataset[i] for i in range(len(dataset))])
        if cache:
            cache.write_text(json.dumps([[r[0], r[1], None if r[2] == float("inf") else r[2], r[3]] for r in rows]), encoding="utf-8")
    rows = [(r[0], r[1], float("inf") if r[2] is None else r[2], r[3]) for r in rows]

    results, n_answerable, n_unanswerable = sweep(rows)
    best = max(results, key=lambda r: r["weighted_f1"])
    plateau = [r["threshold"] for r in results if r["weighted_f1"] >= best["weighted_f1"] - 0.1]
    at_zero = next(r for r in results if r["threshold"] == 0.0)

    def show(label, r):
        print(f"{label:>10}  threshold={r['threshold']:+.2f}  EM={r['exact']:.1f}  F1={r['f1']:.1f}  "
              f"HasAns EM/F1={r['HasAns_exact']:.1f}/{r['HasAns_f1']:.1f}  NoAns={r['NoAns_exact']:.1f}  "
              f"weighted F1 (25% NoAns)={r['weighted_f1']:.1f}")

    print(f"\n{len(rows)} questions: {n_answerable} answerable, {n_unanswerable} unanswerable")
    show("threshold 0", at_zero)
    show("best", best)
    print(f"Thresholds within 0.1 point of the best: {min(plateau):+.2f} to {max(plateau):+.2f}")

    if args.out:
        summary = {
            "split": args.split,
            "n_examples": len(rows),
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "unanswerable_weight": UNANSWERABLE_WEIGHT,
            "threshold_0": at_zero,
            "best_threshold": best,
            "plateau_range": [min(plateau), max(plateau)],
            "sweep": results,
        }
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
