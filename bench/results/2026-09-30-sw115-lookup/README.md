# sw115: prompt-lookup drafts stacked on the MTP head (P-4), simulated: not adopted (2026-09-30)

**Question.** When the text repeats something earlier in the sequence (code being re-emitted, a
document quoted back), an n-gram match can draft many tokens for free. Would lookup drafts,
alone or stacked on the MTP head, beat the head's fixed K = 2?

**Method.** Offline, on real sequences, with the head's own drafts at every position:
- `fr_bench --mtp --draft 3` without `--spec` drafts 3 tokens from the true streams at every
  position; `--round-log` (new in this mode) keeps them and `--save-tokens` (new) the text.
- `lookup.py` walks the text in rounds. A lookup draft is the continuation (of known text only)
  after the most recent earlier occurrence of the last g tokens, up to M; drafts are kept while
  they equal the text (greedy is exact). Policies, all against A (the head, K = 2):
  - B: lookup when there is a match, else the head;
  - C: lookup only when its first token equals the head's first draft (exact for sampled drafts
    too: position 1 keeps the head's draw, later drafts depend only on the prefix);
  - D: the head's 2 drafts extended by lookup.
- Costs: the draft phase from sw114; the verify phase per window length measured for T = 2..8 by
  `sw115c.sh` (teacher-forced `--spec 1..7`) on the 32K wikitext state and the code text, from
  sw114 for T <= 4 elsewhere (extrapolated above, marked in the outputs).

**Texts.**
- `sw115.sh`: greedy text from the 32K and 245K wikitext states and window 9's prompt. Greedy text
  degenerates once the answer ends (it loops, or repeats an end token: window 9 from token 524), so
  `lookup.py` cuts it at the first end-of-turn or repeated 64-gram (828, 909 and 524 tokens used).
- `sw115b.sh` (teacher-forced, natural continuations): the wikitext file after 32K and 245K, window
  9's 384-token reference answer, and a code edit (`make_code_edit.py`: a chat request for
  `moe_fast.cu` back with two identifiers renamed; the answer is that file, 18,177 tokens after an
  18,219-token prompt; tokenized with the new `flashrt-server --tokenize`).

**Window cost** (`sw115c.out`, verify ms by window length T):

| T | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| 32K wikitext | 11.2 | 13.1 | 15.2 | 20.4 | 23.7 | 25.5 | 28.3 |
| code edit (~18K) | 14.8 | 19.1 | 23.9 | 31.8 | 37.1 | 44.0 | 49.0 |

Every extra token of a window routes to its own experts, and the union of CPU misses grows with
it: about 5 ms per token on code.

**The head already copies.** On the code edit the head alone keeps 100% of its drafts at K = 1-3
and all 7 drafts in 83% of rounds at K = 7. Draft + verify rate by K: 125 / 142 / 148 / 138 tok/s
for K = 1 / 2 / 3 / 4, and 135 at K = 7. Lookup can only save the head's draft steps (about 1 ms
each), while a longer window costs 5 ms per token.

**Results** (`lookup-teacher.out`, `lookup-greedy.out`; g = 4, the best n-gram length; policy C,
the best policy):

| context | M = 2 | M = 4 | M = 7 |
|---|---|---|---|
| code edit, teacher-forced | +1.8% | −0.3% | +1.2% |
| wikitext 32K, teacher-forced | +0.2% | −1.0% | −3.3% |
| wikitext 245K, teacher-forced | +0.4% | −0.3% | −2.0% |
| window 9 reference answer (321 tokens, T > 4 extrapolated) | +1.5% | +5.4% | +4.8% |
| wikitext 32K greedy (828 tokens) | | −3.3% | −8.2% |
| wikitext 245K greedy (909 tokens, T > 4 extrapolated) | | +4.6% | +5.0% |
| window 9 greedy (524 tokens: 2-3 lookup rounds) | | +0.5% | +0.1% |

B and D are worse than C nearly everywhere. With T > 4 costs extrapolated from sw114's trend, the
code edit simulated at +48% (M = 8); the measured window costs removed all of it.

**Conclusion.** Not adopted: from −8% to +5% depending on the text, near zero on the case lookup
is known for (code editing), because the MTP head already predicts copied text and the offloaded
MoE makes each extra window token cost a union of CPU misses. What would change this: windows
that cost less per token (fewer or cheaper CPU misses), or longer copies than 8 tokens (the window
graphs' limit).

**Also found.** `fr_bench`'s greedy windows run past the end of the answer (it does not stop at
end-of-turn): sw114's greedy window 9 rows likely include rounds on the repeated end token
(window 9's greedy answer ends at token 524 here), which flatters long K there.

## Files

- `sw115.sh`, `sw115b.sh`, `sw115c.sh` and their `.out`; `greedy/`, `teacher/`, `verify/`: logs,
  texts (`.gen`), per-position head drafts (`.drafts`), round logs.
- `lookup.py DIR SW114_LOGDIR NAME:IDS:N:COST[:VERIFY] ...`: the replay. It reads the prompt ids from
  `DIR/ids/` (not committed: the wikitext ids are `$BENCH/p0c-20260927/wiki.prompt_ids.txt`, window
  9's `$BENCH/sw100/w9_teacher.ids`, the code edit's from `make_code_edit.py`).
- `make_code_edit.py`: the code-edit text.
