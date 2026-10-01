# Reference inputs (phase 6, X-1)

What a second machine needs, besides the model files, to repeat the published comparisons.

| File | What | Source |
|---|---|---|
| `w9-ids.json` | Window 9's prompts as token ids: `[{"depth": D, "ids": [...]}]` for 1K, 32K, 134K and 250K (a synthetic context followed by the same instruction). The same ids went to every engine. | `$BENCH/strata-ids.json` on the box (2026-09-27); sha256 `a34b6678…7590` |
| `mtp-vocab-ranks.txt` | The drafter's vocabulary ranking: one token id per line, most frequent first. The engine takes the top 32,768 plus the prompt's tokens (`--draft-vocab`). | `bench/mtp_vocab.py` (2026-09-28); sha256 `536982f2…fb01` |
| `llama.cpp-flashnext.patch` | The llama.cpp reference engine: the expert cache, sparse QSA gather, whole-block top-k, pooled-key cache and the MTP draft head. Apply it to upstream llama.cpp **ec928150501c2572fec05cb949061672bb424914** (2026-09-18) with `git apply`. | The box's dev tree as built on 2026-09-27 06:52, uncommitted changes included. It is the build behind window 9's llama.cpp rows and the KLD base. 27 files. |
| `kld-base.sh` | Rebuilds the KLD gate's base `kl8k-f16.bin` (4.1 GB, not committed). | The P1 gate (`bench/results/2026-09-27-p1-kld`) |

**The llama.cpp patch** is MIT, as llama.cpp is (copyright the ggml authors). It contains:
- the GPU expert cache by csantiago78 (2026-08-28), with fixes and an admission gate (2026-09-21);
- the qwen4exp MTP draft head of upstream PR ggml-org/llama.cpp#28243, applied with fixes for
  offloaded MoE;
- an AVX2 Q2_0 × Q8_0 dot product, and the QSA top-k gather and indexer changes (2026-09-20 to 26).

The window 9 settings for it are in `bench/results/2026-09-27-w9-validation` (`--moe-expert-cache
48 --moe-expert-cache-inserts 2`, `LLAMA_MOE_CACHE_ADMIT=3`, `LLAMA_MOE_CACHE_WINDOW=32`).

**Strata** (the other reference engine; flashrt's clean-room rule applies: read its docs, not its
source):
- repository github.com/Niko1221/Strata, main at **f02dc49dfa4859427701e558d0d5ec8bf340e936**
  (engine 0.1.6, 2026-09-27);
- with its PRs #19 (`3e0836f`, sampling: min_p and penalties) and #18 (`a40fe52`, serve:
  max_tokens) merged locally (`bf3eacf`);
- a local change to `setup.py`, not inspected under the clean-room rule;
- tuned flags `--vram-reserve-mib 1024 --pcie-frac 0.35 --pool-workers 8`.

**Wikitext-2** (the KLD gate's text and the decode prompt) is not committed. It is CC BY-SA, from the
usual `wikitext-2-raw-v1` release; `kld-base.sh` gives the test file's sha256. The decode prompt
`$BENCH/p0c-20260927/wiki.prompt_ids.txt` (250,000 ids; sha256 `ec478f99…09dc`) is the test and
validation files concatenated and tokenized with BOS (`bench/p0/window_c.sh`, step 0):

```bash
cat $DATA/wikitext-2-raw/wiki.test.raw $DATA/wikitext-2-raw/wiki.valid.raw > wiki.txt
build/route_trace --model $M --text wiki.txt --n-prompt 250000 --tokenize-only --out wiki   # wiki.prompt_ids.txt
```
