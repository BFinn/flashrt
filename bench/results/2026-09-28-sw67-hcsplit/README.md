# sw67: which hc matrices tolerate Q8P (2026-09-28)

The fast decode path's KLD (`fr_kld --ctx 8192 --chunks 2 --fast --prefill-chunk 2048 --kv-hot
512`), with one kind of hc matrix as Q8P:

| `FLASHRT_HC_Q8` | KLD mean | same top-1 |
|---|---|---|
| `down` (`hc_*_down`: 10,240 → 320, the normed streams to the low rank) | **0.008744** | 96.85% |
| `up` (`hc_*_up`: 320 → 10,240, the gates of the gated mean) | 0.009656 | 96.51% |
| both (sw66) | 0.010502 | 96.58% |
| neither (this session's runs) | 0.0084-0.0086 | |

The down matrices convert at no measurable cost; the up matrices carry the loss (+0.0011).
**Down only is now the default**. sw68 measures it.
