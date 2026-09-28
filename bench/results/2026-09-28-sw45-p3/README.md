# sw45: chunked prefill in the engine, through the server (2026-09-28)

`flashrt-server` over `flashrt-engine --mtp ... --spec 1 --ctx 65536 --cache-prior ...`: prompts
adding 256 or more tokens prefill in chunks of 4,096; the expert cache's VRAM is lent to them
and the cache is rebuilt after (refilled from the prefill's routing counts plus the prior).
`bench/server_smoke.py`: all checks pass. Raw completions (`longprompt.py`), temperature 1.0:

| Request | prompt (reused) | prefill | decode |
|---|---|---|---|
| wiki text | 2,035 (0) | 1.23 s (16.5 s on the old path) | 84.5 tok/s |
| + its output + more | 2,166 (2,163) | 0.06 s | 104.5 |
| wiki text | 33,587 (0) | **16.7 s: 2,008 tok/s** with the cache rebuild | 104.7 |
| + its output + more | 33,718 (33,715) | 0.06 s | 98.1 |
