#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Token ranking for the MTP drafter's trimmed LM head.

Tokenizes each corpus file with the model's tokenizer (built from the GGUF's vocabulary and
merges and llama.cpp's qwen35 pre-tokenizer pattern; checked against a llama.cpp tokenization
with --verify), turns each file's counts into relative
frequencies (so every domain weighs the same, whatever its size), and ranks tokens by the sum.
Writes one token id per line, most frequent first; the engine takes the top N (plus the tokens
of the prompt) as the drafter's vocabulary. With --check, reports the share of each check
file's tokens covered by the top N, for several N.

  mtp_vocab.py --gguf MODEL.gguf --out ranks.txt CORPUS.txt ... [--check FILE ...] [--verify TEXT IDS]
"""
import argparse
import sys
from collections import Counter

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

# llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35 (MIT)
QWEN35_PATTERN = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                  r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def gguf_tokenizer(path, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    r = GGUFReader(path)

    def strings(key):
        f = r.fields[key]
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]

    tokens = strings("tokenizer.ggml.tokens")
    merges = [tuple(m.split(" ", 1)) for m in strings("tokenizer.ggml.merges")]
    bpe = models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges, fuse_unk=False)
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def ids_of(tok, path):
    text = open(path, encoding="utf-8", errors="replace").read()
    # chunks keep the tokenizer's memory small; the seams change a handful of tokens
    out = []
    for i in range(0, len(text), 1 << 20):
        out.extend(tok.encode(text[i:i + (1 << 20)], add_special_tokens=False).ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gguf-py", default="$LLAMA_CPP/gguf-py")
    ap.add_argument("--verify", nargs=2, metavar=("TEXT", "IDS"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("corpus", nargs="+")
    a = ap.parse_args()
    tok = gguf_tokenizer(a.gguf, a.gguf_py)
    if a.verify:
        ref = [int(x) for x in open(a.verify[1]).read().split()]
        ids = tok.encode(open(a.verify[0], encoding="utf-8").read()[:200000], add_special_tokens=False).ids
        n = min(len(ids), len(ref)) - 64   # the text was cut mid-token
        same = sum(1 for x, y in zip(ids[:n], ref[:n]) if x == y)
        print(f"verify: {same} of {n} ids match the reference tokenization")
    score = Counter()
    for path in a.corpus:
        ids = ids_of(tok, path)
        c = Counter(ids)
        for t, n in c.items():
            score[t] += n / len(ids)
        print(f"{path}: {len(ids)} tokens, {len(c)} distinct")
    ranked = [t for t, _ in score.most_common()]
    with open(a.out, "w") as f:
        f.write("\n".join(str(t) for t in ranked) + "\n")
    print(f"{a.out}: {len(ranked)} ranked tokens")
    for path in a.check:
        ids = [int(x) for x in open(path).read().split()] if path.endswith("_ids.txt") else ids_of(tok, path)
        rank = {t: i for i, t in enumerate(ranked)}
        line = []
        for n in (16384, 32768, 40960, 49152, 65536):
            cov = sum(1 for t in ids if rank.get(t, 1 << 30) < n) / len(ids)
            line.append(f"{n}: {100 * cov:.2f}%")
        print(f"coverage of {path} ({len(ids)} tokens): " + ", ".join(line))


if __name__ == "__main__":
    main()
