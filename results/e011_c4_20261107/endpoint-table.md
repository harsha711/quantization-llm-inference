Application values are cycle median/4 in us, series 1 / series 2. Profiler duration is the sum of selected sequential kernel durations, excluding gaps. Traffic and instructions cover the fixed seed 6 operation, not all four matrices. MiB means 2^20 bytes and instruction millions mean 10^6. Tensor activity and eligible warps remain per kernel, in GEMM/reduction order for the two-kernel operation. Percentages are not peak FLOP utilization.

| K / N | M | Implementation | Application us/GEMM S1 / S2 | Profiler sum us | DRAM read / write MiB | L2 MiB | Warp inst M | Tensor active % | Eligible warps |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 / 11008 | 4 | Marlin A | 61.244 / 60.792 | 62.144 | 22.212 / 0.120 | 25.367 | 3.606 | 24.30 | 0.19 |
| 4096 / 11008 | 4 | FP16 cuBLAS | 268.968 / 268.972 | 273.920 | 87.782 / 4.701 | 168.663 | 9.273 | 10.89 / 0.00 | 0.10 / 0.25 |
| 4096 / 11008 | 128 | Marlin A | 250.924 / 249.764 | 254.304 | 22.366 / 3.280 | 94.553 | 10.959 | 90.86 | 0.14 |
| 4096 / 11008 | 128 | FP16 cuBLAS | 257.800 / 257.792 | 275.040 | 86.091 / 4.420 | 346.876 | 15.635 | 90.76 | 0.19 |
| 11008 / 4096 | 4 | Marlin A | 59.208 / 59.220 | 60.352 | 22.211 / 0.171 | 25.319 | 3.538 | 24.98 | 0.18 |
| 11008 / 4096 | 4 | FP16 cuBLAS | 214.840 / 214.820 | 220.064 | 86.127 / 1.711 | 88.905 | 6.211 | 52.47 | 0.08 |
| 11008 / 4096 | 128 | Marlin A | 247.188 / 245.036 | 249.984 | 22.464 / 3.019 | 93.270 | 10.702 | 92.81 | 0.14 |
| 11008 / 4096 | 128 | FP16 cuBLAS | 292.508 / 296.336 | 304.224 | 101.320 / 2.998 | 345.171 | 11.783 | 80.54 | 0.12 |
