#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibration and evaluation text for the per-expert quant search (docs/research/dynamic-quant.md).

Streams ungated Hub datasets and writes, per domain, a calibration text for llama-imatrix and
route_trace, plus two KLD evaluation texts built from documents disjoint from calibration:

  calib-<domain>.txt   about --calib-chars characters per domain
  eval-search.txt      KLD set used by the search (domain blocks, about 2 chunks each)
  eval-holdout.txt     KLD set never seen by the search
  manifest.json        sources, document counts and character counts

Chat-like sources are wrapped in the model's own template (<|im_start|>, <think>), so run
the consumers with --parse-special. CPU and network only; memory stays small (streaming).

  calib_data.py OUT_DIR [--calib-chars 900000] [--eval-chunk-chars 9000]
"""
import argparse
import json
import os
import sys

from datasets import load_dataset


def im(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def fineweb_edu():
    for r in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True):
        yield r["text"]


def openthoughts():
    # {"system": str, "conversations": [{"from": "user"|"assistant", "value": str}]};
    # assistant turns carry <|begin_of_thought|> ... markers, mapped to <think>.
    for r in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True):
        out = im("system", r["system"]) if r.get("system") else ""
        for t in r["conversations"]:
            v = t["value"]
            if t["from"] == "assistant":
                v = (v.replace("<|begin_of_thought|>", "<think>\n").replace("<|end_of_thought|>", "\n</think>\n")
                      .replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", ""))
            out += im("assistant" if t["from"] == "assistant" else "user", v.strip())
        yield out


def code():
    for r in load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True):
        yield r["content"]


def agentic():
    # hermes-function-calling-v1: {"conversations": [{"from": system|human|gpt|tool, "value"}]}
    roles = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    files = ["func-calling.json", "json-mode-agentic.json", "func-calling-singleturn.json"]
    for f in files:
        for r in load_dataset("NousResearch/hermes-function-calling-v1", data_files=f, split="train", streaming=True):
            yield "".join(im(roles.get(t["from"], "user"), t["value"].strip()) for t in r["conversations"])


LANGS = ["deu_Latn", "fra_Latn", "spa_Latn", "cmn_Hani", "jpn_Jpan", "rus_Cyrl", "fin_Latn", "arb_Arab"]


def multilingual():
    # Round-robin over languages so every slice (calibration, search, holdout) sees all of them.
    streams = [iter(load_dataset("HuggingFaceFW/fineweb-2", lang, split="train", streaming=True)) for lang in LANGS]
    for rows in zip(*streams):
        for r in rows:
            yield r["text"]


DOMAINS = {"general": fineweb_edu, "chat": openthoughts, "code": code, "agentic": agentic,
           "multilingual": multilingual}


def take(gen, chars, max_doc):
    """Documents until `chars` characters, each clipped to max_doc characters."""
    out, n = [], 0
    for doc in gen:
        doc = doc.strip()
        if len(doc) < 200:
            continue
        doc = doc[:max_doc]
        out.append(doc)
        n += len(doc)
        if n >= chars:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--calib-chars", type=int, default=900_000)   # about 250K tokens of English
    ap.add_argument("--eval-chunk-chars", type=int, default=9_000)  # a 2048-token chunk, generously
    ap.add_argument("--search-chunks", type=int, default=3)         # per domain
    ap.add_argument("--holdout-chunks", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest = {"domains": {}, "langs": LANGS}
    search, holdout = [], []
    for name, fn in DOMAINS.items():
        gen = fn()
        # Evaluation documents come first from the stream, calibration after, so they are disjoint.
        s = take(gen, a.search_chunks * a.eval_chunk_chars, a.eval_chunk_chars)
        h = take(gen, a.holdout_chunks * a.eval_chunk_chars, a.eval_chunk_chars)
        c = take(gen, a.calib_chars, 40_000)
        search.append((name, "\n\n".join(s)))
        holdout.append((name, "\n\n".join(h)))
        with open(os.path.join(a.out, f"calib-{name}.txt"), "w") as f:
            f.write("\n\n".join(c))
        manifest["domains"][name] = {"calib_docs": len(c), "calib_chars": sum(map(len, c)),
                                     "search_docs": len(s), "holdout_docs": len(h)}
        print(name, manifest["domains"][name], flush=True)
    # One block per domain, about --search-chunks (or --holdout-chunks) chunks each, in order.
    for fname, blocks in (("eval-search.txt", search), ("eval-holdout.txt", holdout)):
        with open(os.path.join(a.out, fname), "w") as f:
            f.write("\n\n".join(text for _, text in blocks))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # Streaming datasets leave pyarrow threads that crash the interpreter at shutdown
    # (PyGILState_Release) after the files are written; skip the finalisation.
    os._exit(0)
