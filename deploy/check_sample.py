#!/usr/bin/env python3
"""Run the public sample set against a running service and print the table for REPORT.md section 5.

Standard library only. Usage (the service must be running, QA_API_TOKEN set in this terminal):

    python deploy\\check_sample.py --url http://127.0.0.1:8000 --sample public_sample.json

Verdicts follow the grading rules of the brief:
  Yes      exact match with any listed reference (after SQuAD normalization), or null on an unanswerable item
  Partial  the span overlaps a reference but is not an exact match (the instructor's LLM judge decides these)
  No       wrong span, null on an answerable item, a span on an unanswerable item, or an invalid response
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 5.0


def normalize(text):
    """Official SQuAD normalization: lowercase, drop punctuation and articles, collapse spaces."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    common = collections.Counter(p) & collections.Counter(r)
    overlap = sum(common.values())
    if not p or not r or overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def predict(url, token, item):
    body = json.dumps({"id": item["id"], "context": item["context"], "question": item["question"]}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "check-sample/1.0")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:  # timeout, connection refused...
        return None, repr(exc)


def contract_problem(item, status, body):
    """Return a short reason if the response breaks the contract, else None."""
    if status != 200 or not isinstance(body, dict):
        return f"status {status}"
    if body.get("id") != item["id"]:
        return "id mismatch"
    if any(k not in body for k in ("answer", "start", "score")):
        return "missing field"
    answer, start = body["answer"], body["start"]
    if (answer is None) != (start is None):
        return "answer/start null mismatch"
    if answer is not None and item["context"][start:start + len(answer)] != answer:
        return "context[start:start+len(answer)] != answer"
    return None


def verdict(item, body):
    answer = body["answer"]
    if item["is_impossible"]:
        return ("Yes", "") if answer is None else ("No", "answered, expected null")
    if answer is None:
        return "No", "returned null on an answerable question"
    refs = [a["text"] for a in item["answers"]]
    if any(normalize(answer) == normalize(r) for r in refs):
        return "Yes", ""
    if max(token_f1(answer, r) for r in refs) > 0:
        return "Partial", "overlaps a reference but is not an exact match"
    return "No", "different span"


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--sample", default="public_sample.json")
    args = ap.parse_args()

    token = os.environ.get("QA_API_TOKEN")
    if not token:
        sys.exit("Set QA_API_TOKEN in this terminal first (never put it in a file).")

    with open(args.sample, encoding="utf-8") as fh:
        items = json.load(fh)["items"]

    rows, notes, counts = [], [], collections.Counter()
    for item in items:
        status, body = predict(args.url, token, item)
        expected = "`null`" if item["is_impossible"] else " / ".join(f"`{a['text']}`" for a in item["answers"])
        problem = contract_problem(item, status, body)
        if problem:
            result, why, got, start = "No", f"invalid response: {problem}", "-", "-"
        else:
            result, why = verdict(item, body)
            got = "`null`" if body["answer"] is None else f"`{body['answer']}`"
            start = "null" if body["start"] is None else body["start"]
        counts[result] += 1
        rows.append(f"| {item['id']} | {cell(expected)} | {cell(got)} | {start} | {result} |")
        if why:
            notes.append(f"- {item['id']}: {why}. Question: {item['question']}")

    print("| Item | Expected | Your answer | `start` | Correct? |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nSummary: {counts['Yes']} yes, {counts['Partial']} partial, {counts['No']} no, of {len(items)} items.")
    if notes:
        print("\nNotes for the 'comment on any wrong answer' paragraph:")
        print("\n".join(notes))


if __name__ == "__main__":
    main()
