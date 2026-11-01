#!/usr/bin/env python3
"""Measure /predict latency and check the API contract, locally and through the tunnel.

Only needs the standard library. Usage:

    export QA_API_TOKEN=<development token>      # PowerShell: $env:QA_API_TOKEN = "..."
    python measure_latency.py \
        --url http://localhost:8000 \
        --url https://<name>.ngrok-free.app \
        --sample public_sample.json --repeat 5

The first --url is treated as the "local" baseline; every other URL is compared against it
to estimate the overhead added by the tunnel. The 5 s timeout of the harness applies to the
full round trip, so the script uses the same value.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

TIMEOUT_S = 5.0          # per-request timeout used by the harness
GLOBAL_CAP_S = 150.0     # global cap for the 50 hidden items
N_ITEMS = 50


def load_items(path):
    """Read public_sample.json (a list, or a dict holding a list) -> [{id, context, question}]."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("items", "data", "examples", "questions"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    items = []
    for row in data:
        items.append({"id": str(row["id"]), "context": row["context"], "question": row["question"]})
    return items


def long_item(items, target_chars=4900):
    """Build a context close to the 5000-character limit: the slowest realistic case."""
    pieces, total = [], 0
    while total < target_chars:
        for it in items:
            pieces.append(it["context"])
            total += len(it["context"]) + 1
            if total >= target_chars:
                break
    context = " ".join(pieces)[:5000]
    return {"id": "long-synthetic", "context": context, "question": items[0]["question"]}


def call(url, path, token=None, body=None):
    """Return (status, parsed_json_or_None, seconds)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method="POST" if body else "GET")
    req.add_header("User-Agent", "latency-check/1.0")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw, status = resp.read(), resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    except Exception as exc:  # timeout, connection refused, DNS...
        return None, {"transport_error": repr(exc)}, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, None, elapsed


def check_access(url, token):
    """The five checks from the brief (health, info with no/wrong/valid token)."""
    print("  access-control checks")
    results = [
        ("health, no token", call(url, "/health")[0], 200),
        ("info, no token", call(url, "/info")[0], 401),
        ("info, wrong token", call(url, "/info", token="wrong")[0], 401),
        ("info, valid token", call(url, "/info", token=token)[0], 200),
    ]
    ok = True
    for name, got, want in results:
        flag = "OK  " if got == want else "FAIL"
        ok &= got == want
        print(f"    [{flag}] {name}: got {got}, expected {want}")
    return ok


def valid_response(item, status, body):
    """Contract check: same id, required fields, and context[start:start+len(answer)] == answer."""
    if status != 200 or not isinstance(body, dict):
        return False, f"status {status}"
    if body.get("id") != item["id"]:
        return False, "id mismatch"
    for field in ("answer", "start", "score"):
        if field not in body:
            return False, f"missing '{field}'"
    answer, start, score = body["answer"], body["start"], body["score"]
    if (answer is None) != (start is None):
        return False, "answer/start null mismatch"
    if answer is not None and item["context"][start:start + len(answer)] != answer:
        return False, "answer is not the context slice at start"
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return False, "score outside [0, 1]"
    return True, ""


def percentile(values, q):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def run(url, token, items, repeat):
    print(f"\n=== {url}")
    access_ok = check_access(url, token)

    # Warm-up: the first request after startup is slower (lazy init of torch kernels).
    call(url, "/predict", token=token, body=items[0])

    rows = []  # (label, seconds, ok, why)
    for it in items + [long_item(items)]:
        for _ in range(repeat):
            status, body, secs = call(url, "/predict", token=token, body=it)
            ok, why = valid_response(it, status, body)
            if secs >= TIMEOUT_S:
                ok, why = False, f"took {secs:.2f}s (>= {TIMEOUT_S}s timeout)"
            rows.append((it["id"], secs, ok, why))

    print("  per item (max over repeats)")
    for label in dict.fromkeys(r[0] for r in rows):
        same = [r for r in rows if r[0] == label]
        worst = max(r[1] for r in same)
        bad = [r[3] for r in same if not r[2]]
        print(f"    {label:<18} {worst:6.3f} s   {'OK' if not bad else 'FAIL: ' + bad[0]}")

    times = [r[1] for r in rows]
    stats = {
        "mean": statistics.mean(times),
        "p50": percentile(times, 0.50),
        "p95": percentile(times, 0.95),
        "max": max(times),
        "failed": sum(1 for r in rows if not r[2]),
        "n": len(rows),
    }
    print(f"  requests: {stats['n']}   failed: {stats['failed']}")
    print(f"  latency  mean {stats['mean']:.3f} s | p50 {stats['p50']:.3f} s | "
          f"p95 {stats['p95']:.3f} s | max {stats['max']:.3f} s")
    est = stats["mean"] * N_ITEMS
    print(f"  estimate for {N_ITEMS} items at the mean: {est:.1f} s "
          f"(cap {GLOBAL_CAP_S:.0f} s) -> {'OK' if est < GLOBAL_CAP_S else 'TOO SLOW'}")
    stats["access_ok"] = access_ok
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", required=True, help="base URL; first one = local baseline")
    ap.add_argument("--sample", default="public_sample.json")
    ap.add_argument("--repeat", type=int, default=5, help="repetitions per item")
    args = ap.parse_args()

    token = os.environ.get("QA_API_TOKEN")
    if not token:
        sys.exit("Set QA_API_TOKEN in the environment first (never put it in the code).")

    items = load_items(args.sample)
    all_stats = [(u, run(u, token, items, args.repeat)) for u in args.url]

    if len(all_stats) > 1:
        base_url, base = all_stats[0]
        print("\n=== Tunnel overhead (vs first URL)")
        for url, st in all_stats[1:]:
            print(f"  {url}: p50 +{st['p50'] - base['p50']:.3f} s, p95 +{st['p95'] - base['p95']:.3f} s")

    sys.exit(0 if all(s["failed"] == 0 and s["access_ok"] for _, s in all_stats) else 1)


if __name__ == "__main__":
    main()
