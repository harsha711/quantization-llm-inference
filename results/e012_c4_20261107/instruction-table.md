Instruction counts are warp-level counts ignoring instruction predicates. Millions mean 10^6. Shares describe diagnostic instruction counts, not runtime. Known regions exclude ambiguous inline callers.

| K / N | M | Warp instructions M | Known main loop % | Known coordination/epilogue % | Ambiguous phase % | Source coverage % |
|---|---:|---:|---:|---:|---:|---:|
| 4096 / 11008 | 4 | 3.606077 | 49.82 | 1.82 | 43.80 | 100.00 |
| 4096 / 11008 | 128 | 10.958725 | 63.30 | 2.21 | 31.43 | 100.00 |
| 11008 / 4096 | 4 | 3.538074 | 50.33 | 1.15 | 44.77 | 100.00 |
| 11008 / 4096 | 128 | 10.702177 | 64.17 | 1.35 | 31.80 | 100.00 |
