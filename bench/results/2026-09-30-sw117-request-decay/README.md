# sw117: flashrt-server slows over many short requests: diagnosis (2026-09-30)

`sw117.sh` runs the first N GSM8K items through flashrt-server (the MTP head, `--spec 2`) and keeps
the server's per-request `timings` (expert-cache hits and misses, drafts). The first run (N = 200)
showed the expert cache's hit rate falling from 83% to 12% while draft acceptance held; the second
(`N=100 TAG=-check EXTRA="--engine-arg --cache-check"`) showed the cache losing usable slots:
7,179 resident of 7,800 after one request, 2,961 after 60, the rest orphaned (owned, never free).
The cause, the fix and the before/after data are in `2026-09-30-sw118-cache-leak`.
