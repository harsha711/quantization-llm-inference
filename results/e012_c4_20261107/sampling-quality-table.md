Sampling uses one pass at 32 cycles with a 32 MiB buffer. All eight captures fail the zero-dropped-data gate. M128 also fails 95% source coverage. These counts document a measurement limitation, not accepted stall attribution.

| K / N | M | Series | Recorded samples | Dropped bytes | Unmapped all-ones PC samples | Source coverage % |
|---|---:|---:|---:|---:|---:|---:|
| 4096 / 11008 | 4 | 1 | 179996 | 77001 | 8733 | 95.15 |
| 4096 / 11008 | 128 | 1 | 772262 | 587751 | 66931 | 91.33 |
| 11008 / 4096 | 4 | 1 | 178085 | 73446 | 8310 | 95.33 |
| 11008 / 4096 | 128 | 1 | 761271 | 572919 | 65038 | 91.46 |
| 11008 / 4096 | 128 | 2 | 760887 | 563819 | 64075 | 91.58 |
| 11008 / 4096 | 4 | 2 | 176029 | 72292 | 8186 | 95.35 |
| 4096 / 11008 | 128 | 2 | 772003 | 598942 | 68136 | 91.17 |
| 4096 / 11008 | 4 | 2 | 181284 | 79265 | 8996 | 95.04 |
