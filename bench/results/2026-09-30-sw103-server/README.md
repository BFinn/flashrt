# sw103: the server after the head's chunk buffers (2026-09-30)

sw102 gave the MTP head a second buffer set, about 450 MiB, during every chunked prefill: a change
to the prefill's VRAM budget. `fr_bench` cannot catch a server-side out-of-memory (sw86), so the
server was run on this build (`sw103.sh`). Engine: `--mtp --spec 2 --draft-vocab`, 262K context,
defaults otherwise.

- **`server_smoke.py`: all checks pass** (`smoke.txt`). The server was ready after 32 s.
- **A long chat prompt:** the first 200,000 characters of wikitext twice, then "Summarise the text
  above in three sentences" (95,388 tokens), greedy, thinking off (`long.txt`):
  - prefilled cold in 17.4 s: 5,494 tok/s with the head, in chunks with its chunk set;
  - 137 tokens decoded at 93.8 tok/s, 85 of 108 drafts accepted, expert cache 63% hits;
  - a correct summary of the text's topics. No errors or out-of-memory in `server.log`.

(The first two attempts sent the wrong input: a file of token ids read as text, 420,023 tokens,
and then all of wikitext, 1,117,228 tokens. The server rejected both with a 400 naming the
prompt's length and the context, as it should.)
