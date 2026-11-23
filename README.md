# Mixed-precision quantized CUDA GEMM for LLM inference

CSC 548 Parallel Systems, NC State. Samarth Shah and Harsha Vardhan Puvvadi. **[Presentation (PDF)](presentation/W4A16_presentation.pdf)**

We built fused W4A16 GPU kernels (4-bit weights, FP16 activations) step by step, each step chosen from a profile, and measured them against Marlin, the kernel inside vLLM. Our final kernel, **W4A16-TC** (`tc_ladder` in the code), reaches 0.83–0.96 of Marlin's speed at M = 4–128 and decodes Qwen2.5-1.5B within 5% of Marlin's time.

## Problem

`C = A·B` with `A` an M×K FP16 matrix and `B` a K×N matrix stored as packed INT4 with one FP16 scale per 128 weights. Weights are reconstructed as `FP16((q-8)*s)`, accumulated in FP32 and written as FP16. Every kernel is checked against an FP64 CPU reference: `|error| <= 1e-3*|ref| + 1e-3*rms(ref)`. See [CONTRACT](docs/CONTRACT.md).

## Kernel steps

Median µs, one 4096×4096 matrix, store-flushed, RTX 5060 Ti (E051, range over two series):

| Kernel | Change | Experiment | M = 1 | M = 128 |
|---|---|---|---:|---:|
| `naive` | fused baseline | | 204.4–205.2 | 2,554–2,563 |
| `inblock` | split K inside a block | E001 | 52.9 | 3,384–3,398 |
| `inblock_m4` | four rows share each weight | E003 | 61.1–61.8 | 1,473–1,480 |
| `inblock_m4_vec` | 16-byte loads, with row reuse | E043 | 53.2–53.6 | 885–889 |
| `tc` | Tensor Cores | E044 | 32.4–33.1 | 135.5 |
| **`tc_ladder`** | 32-row tile from M = 64, `tc` below | E050 | 32.4–33.1 | 126.7–129.3 |
| cuBLAS, pre-reconstructed FP16 weights | comparator | | 94.2 | 114.7 |
| dequantize + cuBLAS | unfused comparator | | 178.5 | 225.3 |

The store flush inflates fast kernels' times, so these are not compared with Marlin. A parallel scalar series (`v0`–`v3`, [OURS](docs/OURS.md)) targets M = 1.

## Main results

| Question | Result | Experiment |
|---|---|---|
| Speed versus Marlin, cycled weights | 0.86–0.96 at M 4/16 and 0.83–0.87 at M 64/128 (4096×11008, 11008×4096); 0.90–0.96 at 4096² with 16 matrices | E050, E058 |
| Scalar kernel at M = 1 | `v3` comparable with Marlin: 1.00–1.05 of its speed on five large shapes; far slower from M = 4 | E059 |
| 32-row tile | 13–17% less time than `tc` at M 64/128 on rectangles; 12/12 cells pass | E050 |
| Scalar row reuse | `v3_m2` 22–37% less time than `v3` at M 4/8; still slower than `tc_ladder` | E053 |
| CPU baseline | 26.6/31.3/241 ms serial at M 1/4/128; 12.3–13.1× speedup on 16 threads | E041 |
| Measured GPU roofs | 427 GB/s read, 51.8 TFLOP/s Tensor Core | E057 |
| Marlin shared-memory race | one added barrier removes the reported hazard; timing cost −0.35% to +0.39% | E006, E060 |
| Decode versus FP16 | `tc_ladder` 47% to 25% less time, context 128 to 16,384 | E052 |
| Decode and prefill versus Marlin | decode 0.7–3.7% less time ("comparable"); prefill 7–12% longer to 4,096 tokens, comparable beyond | E052 |
| Other GPUs | decode stays within 5% of Marlin on 10 nodes, 5 GPU models; 2,048-token prefill 47% longer on A100, 88–96% on A6000 | E054 |
| Versus vLLM | our decode takes 32–46% longer; about 1,000 extra unfused launches per step explain most of it | E052, E056 |
| Output quality | Marlin and `tc_ladder` give the same perplexity (9.735) | E055 |
| Cost of 4-bit weights | +13.1% perplexity with plain rounding, +5.6% with calibrated GPTQ (missed the 5% target) | E032, E037 |

Limits: model results use one model, batch 1, 64 generated tokens and plain-rounded weights, and the FP16 baseline uses weights reconstructed from the same 4-bit codes. The model path passed only a third numerical gate (FP64 control, E021), after two earlier gates against FP16 (E017, E019) rejected it. The M = 64 threshold was chosen after profiling and re-tested on fresh weights. Kernel comparisons use synthetic weights, mainly on one RTX 5060 Ti.

Tried and rejected: split-K on Tensor Cores (E047) and 32-row tiles at every M ≥ 32 (E048).

## Build and test

```bash
bash scripts/build.sh                                   # host only if nvcc is absent
ctest --test-dir build --output-on-failure
bash scripts/check_inblock.sh                           # on a GPU, before measuring
./build/w4a16_bench --M 128 --N 4096 --K 4096 --group 128 --kernels tc_ladder --csv run.csv
```

## Repository

- [src/](src/), [include/](include/), [tests/](tests/), [tools/](tools/): kernels, CPU baseline, tests and benchmarks
- [scripts/](scripts/), [results/](results/): harness and raw records per experiment (`eNNN`)
- [EXPERIMENTS](docs/EXPERIMENTS.md): full log, including failures; [LATEST_RESULTS](docs/LATEST_RESULTS.md): which numbers are current
- [report/](report/): ACM report source (covers work through E050)

Raw records of the rejected or stopped experiments E013–E016 and E019, and E017's bulk copies, are left out of this tree. They remain in [history](https://github.com/harsha711/quantization-llm-inference/tree/423314742db086a057a6390c99ff3e03fa3bcb79).

Marlin (IST-DASLab) and vLLM are upstream work. GPTQ, llama.cpp and ONNX Runtime are external comparators or methods; see [REFERENCES](docs/REFERENCES.md).
