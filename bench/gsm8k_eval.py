#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GSM8K through an OpenAI-compatible chat endpoint, for comparing engines on the same model.

  gsm8k_eval.py --url http://127.0.0.1:8090 --data test.jsonl --n 500 --out flashrt.jsonl
  gsm8k_eval.py --compare A.jsonl B.jsonl

Each item is one chat request: greedy (temperature 0), thinking off (chat_template_kwargs
enable_thinking = false), --max-tokens per answer, one request at a time. The prompt asks for the
final answer on a last line "#### <number>"; the answer is that number, or else the response's last
number. A result line per item goes to --out (items already there are skipped, so a run resumes).
--compare prints both accuracies with 95% Wilson intervals and the paired table: items right in one
run only, with McNemar's exact two-sided p-value, and how often the two responses are identical.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request

PROMPT = ("Solve the following math problem step by step. After your solution, write the final answer "
          "(a number only) on the last line in the form \"#### <answer>\".\n\nProblem: {q}")
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def number(s):
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def extract(text):
    m = re.findall(r"####\s*\$?\s*(" + NUM + ")", text)
    if m:
        return number(m[-1])
    m = re.findall(NUM, text)
    return number(m[-1]) if m else None


def gold(answer):
    return number(answer.split("####")[-1].strip())


def post(url, body, timeout):
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def run(a):
    items = [json.loads(l) for l in open(a.data)][: a.n]
    done = set()
    try:
        for l in open(a.out):
            done.add(json.loads(l)["idx"])
    except FileNotFoundError:
        pass
    out = open(a.out, "a")
    right = total = 0
    t_all = time.time()
    for i, it in enumerate(items):
        if i in done:
            continue
        body = {"model": "m", "messages": [{"role": "user", "content": PROMPT.format(q=it["question"])}],
                "temperature": 0, "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        r = post(a.url, body, a.timeout)
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        text = msg.get("content") or ""
        g, p = gold(it["answer"]), extract(text)
        ok = p is not None and g is not None and abs(p - g) < 1e-6
        rec = {"idx": i, "gold": g, "pred": p, "correct": ok, "finish": r["choices"][0].get("finish_reason"),
               "completion_tokens": r.get("usage", {}).get("completion_tokens"), "seconds": round(dt, 3), "content": text,
               "reasoning": msg.get("reasoning_content") or ""}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        right += ok
        total += 1
        if total % 25 == 0:
            print(f"{a.label} {i + 1}/{len(items)}: {right}/{total} right this run, {(time.time() - t_all) / total:.1f} s per item",
                  flush=True)
    print(f"{a.label}: done, {right}/{total} right this run")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(pa, pb):
    A = {r["idx"]: r for r in map(json.loads, open(pa))}
    B = {r["idx"]: r for r in map(json.loads, open(pb))}
    common = sorted(set(A) & set(B))
    for name, R in ((pa, A), (pb, B)):
        k = sum(R[i]["correct"] for i in common)
        lo, hi = wilson(k, len(common))
        trunc = sum(R[i]["finish"] == "length" for i in common)
        toks = [R[i]["completion_tokens"] for i in common if R[i]["completion_tokens"]]
        print(f"{name}: {k}/{len(common)} = {100 * k / len(common):.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}); "
              f"{trunc} cut at max tokens; mean {sum(toks) / max(1, len(toks)):.0f} tokens")
    both = sum(A[i]["correct"] and B[i]["correct"] for i in common)
    a_only = sum(A[i]["correct"] and not B[i]["correct"] for i in common)
    b_only = sum(B[i]["correct"] and not A[i]["correct"] for i in common)
    neither = len(common) - both - a_only - b_only
    same_text = sum(A[i]["content"] == B[i]["content"] for i in common)
    same_pred = sum(A[i]["pred"] == B[i]["pred"] for i in common)
    print(f"paired over {len(common)}: both right {both}, first only {a_only}, second only {b_only}, neither {neither}; "
          f"McNemar exact p = {mcnemar(a_only, b_only):.3f}")
    print(f"identical responses {same_text} ({100 * same_text / len(common):.1f}%), same extracted answer {same_pred} "
          f"({100 * same_pred / len(common):.1f}%)")
    diff = [i for i in common if A[i]["correct"] != B[i]["correct"]]
    print("items right in one run only:", " ".join(f"{i}({'1st' if A[i]['correct'] else '2nd'})" for i in diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)


if __name__ == "__main__":
    sys.exit(main())
