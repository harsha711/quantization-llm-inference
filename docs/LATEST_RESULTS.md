# Latest results registry (which numbers to quote)

Created 2026-11-19 at the user's request: mark results that are stale for "latest kernel" statements so they are not reused. Old experiment records stay immutable; this file only says which number is current. Our latest kernel is `tc_ladder` (E050: `tc_m32` at M >= 64, `tc` below). Every disclosure in AGENTS.md still applies to the current numbers.

Rule: when a statement is about OUR CURRENT kernel or path, quote the "current" column. A stale number may be quoted only as history of its own rung/arm, labelled with its experiment and kernel (e.g. "E044 `tc`, rung 6").

## Stale for latest-kernel statements, with the current replacement

| Topic | Stale (kernel, source) | Current (source) |
|---|---|---|
| Fraction of Marlin speed, cyclic, M >= 64 | `tc` 0.71–0.75 (E045, c7) | `tc_ladder` 0.83–0.87 on the rectangles, 0.83–0.94 incl. 4096² (E050, c8, fresh seeds 7–10) |
| Fraction of Marlin speed, cyclic, M 4/16 | E045 `tc` 0.87–0.95 (c7) | E050 same job as above: 0.87–0.96 on rectangles (`tc_ladder` runs `tc` there); 4096² M4/16 unstable (Marlin moves 10–13% between series) |
| Roofline coordinates | E045 `tc` points (66–75% of compute roof at M >= 64) | NOT YET RECORDED: recompute from E050 series 1 (one job for `tc_ladder`, Marlin, cuBLAS). Calculation done in conversation, not in a record: 4096×11008 M64/M128 `tc_ladder` 80%/85%, Marlin 93%/97% of the nominal 47.4 TFLOP/s roof; M4 363 vs 381 GB/s. Check E050 clocks before quoting compute fractions |
| Warp instructions vs Marlin (cyclic profile) | `tc` 5.2× (E049) | `tc_ladder` 3.1× (E051 d) |
| Single matrix vs cuBLAS Ŵ at M128 | `tc` 1.10–1.20× slower (E044) | `tc_ladder` 1.04–1.13× slower (E051 a) |
| GPU over serial CPU T1 (processor ratios, never parallel speedups) | scalar kernels 161–691× (E041, c58, no flush) | `tc_ladder` 795 / 949 / 1866× at M1/4/128 flushed (E051 c, c8, same node as its CPU T1/T16). No-flush 1488 / 1777 / 2239× and over T16 63 / 74 / 153× flushed are calculations from `results/e051_20261118/full-c8-343361/c-{cpu,gpu}.csv`, not analyzer outputs |
| E009 cycle at M4 vs Marlin | (no earlier our-kernel value) | 0.88–0.96 with 2/4 cycled matrices, 0.59–0.70 one repeated matrix (E051 b) |
| Model decode with our GEMM | Marlin paths GQ/GQA (E020–E029) are Marlin results, not ours | `tc_ladder` GTA (E052; E054 on 10 nodes / 6 GPU types) |
| Marlin vs FP16 decode curve (hook) | E026/E029 45.5/42.5/38.9/32.5/25.2% (older jobs) | Prefer E052 same-job GQA vs GRA: 45/42/39/32/25% at 128/2048/4096/8192/16384 (both valid; E052 is the same job as our kernel) |
| Versus vLLM / llama.cpp / ONNX Runtime | GQA 37–51% slower than vLLM (E027/E028) | GTA 32–35% (128/512) and 46% (2048) slower than vLLM; 51–60% llama.cpp Q4_K_M; 70–90% ORT INT4 (E052 g) |
| Why slower than vLLM (calls/launches) | E030: 196 vs 112 GEMM calls, 1264 vs 276 other launches — measured on the MARLIN path | PENDING E056 (same-job re-profile with `tc_ladder`). Until then quote E030 only as "measured on the Marlin path" |
| 4-bit output head gain | Marlin GQAH 17.0% / 14.7% (E031) | GTAH 17.5% / 15.0% below GTA (E052 e); vLLM VH 23.2–24.2% |
| Calibrated-weight timing | Marlin arms (E038) | cGTA/cGTAH within 0.33% of plain (E052 f) |
| Robustness across nodes | Marlin, 2 nodes (E024) | `tc_ladder` vs Marlin, 10 nodes / 6 GPU types (E054) |

## Still current (not stale)

- E041 CPU serial baseline and OpenMP scaling (CPU kernel `src/cpu_gemm.cpp`, independent of any GPU kernel); E051 c re-measured T1 within 1% on c8.
- Rung histories E001–E003, E043, E044, E046–E048: valid as results of THOSE rungs; never present them as the current kernel.
- Marlin-only work E004–E013 (E005/E006 race fix, E008–E010 cyclic method): not replayed by design (AGENTS.md).
- Quality E032–E042: measured on the weights in PyTorch FP16, not through a kernel; current for the weights. E055 shows the kernel does not change perplexity, but ONLY for plain-rounded weights; GPTQ/AWQ weights were never scored through `tc_ladder` — do not say they were.
- E052–E055: current.
