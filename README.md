# W4A16 mixed-precision GEMM (CSC 548)

**Project synthesis complete:** [findings, contributions and completion gaps](docs/PROJECT_SYNTHESIS.md), [four reproducible figures](docs/figures/project_synthesis/), and [offline reproduction](scripts/reporting/README.md). The record supports a bounded course report. The next task is an ACM report draft, with explicit resolution of the course's serial-baseline/worker-scaling wording. This review adds no experiment. Corrected E006 A remains unchanged.

**Current state (2026-11-10):** with corrected Marlin on all 196 block projections of Qwen2.5-1.5B, 63-step decode is 28.9%/29.0% faster than reconstructed FP16 at context 2048 and 3.6%/4.6% at context 128, with about 60% lower peak allocated GPU memory ([E020](docs/EXPERIMENTS.md#e020--descriptive-primary-timing-of-the-numerically-unaccepted-marlin-path)). The path is accepted only under a third, post hoc gate against an FP64-projection control ([E021](docs/EXPERIMENTS.md#e021--acceptance-of-the-marlin-path-against-the-fp64-projection-control)); the two gates against the FP16 reference (E017, [E019](docs/EXPERIMENTS.md#e019--corrected-marlin-end-to-end-with-a-control-calibrated-model-gate)) remain failed, and [E018](docs/EXPERIMENTS.md#e018--locate-the-e017-qr-logit-divergence-diagnostic) attributes that divergence to propagated FP16 rounding. Replaying the Marlin decode step as a CUDA graph ([E023](docs/EXPERIMENTS.md#e023--cuda-graph-replay-of-the-marlin-decode-step)) raises the gain to 45% at context 128 and 30% at context 2048 versus eager FP16 (44% and 30% versus graph-replayed FP16), bitwise equal to eager Marlin. A direct single-query decode attention ([E026](docs/EXPERIMENTS.md#e026--single-query-grouped-decode-attention-under-graph-replay)) then brings Marlin decode to 325–374 ms across contexts 128–2048, 47–58% below graph-replayed FP16 with stock attention and 42–45% below FP16 given the same attention; the gains are reproduced across contexts, passages and (through E024) a second GPU. Eager Marlin decode was host-bound ([E022](docs/EXPERIMENTS.md#e022--decode-step-projection-time-breakdown-diagnostic), [bottleneck assessment](docs/BOTTLENECK_ASSESSMENT.md)). See [STATUS](docs/STATUS.md).

**Earlier investigation:** [E017 corrected-Marlin model bridge](docs/INFERENCE_MARLIN_BRIDGE.md) stops at the full-model logit gate:591 of9,723,904 values fail despite exact64 tokens. Operator checks pass and setup tensor storage falls62.92%, but no timing is released. [Predeclaration](docs/EXPERIMENTS.md#e017--corrected-marlin-in-the-pinned-full-model), [tooling](scripts/e017/README.md) and [STATUS](docs/STATUS.md).

**Previous investigation:** [E016 complete-decode CUDA graph](docs/INFERENCE_FULL_DECODE_GRAPH.md) is complete. One graph covers all 63 decode steps and preserves every checked logit/token/cache state. Fresh matched decode gains versus per-token replay range from −0.197% to +0.048%, below 5% at both contexts. Graph setup rises from 42–55 ms to 2.46–3.31 s. Retain existing defaults. All 160 primary requests are preserved, with no held-out timing. [Tooling](scripts/e016/README.md), [final evidence](results/e016_20261109/final-arc/) and [STATUS](docs/STATUS.md).

[E015 per-token graph replay](docs/INFERENCE_DIRECT_GRAPH.md) remains complete with subthreshold 1.08–1.87% gains versus its fresh eager control. E016 uses new G denominators and does not change that historical result.

[E014 compiled inference](docs/INFERENCE_OVERHEAD.md) remains stopped at its numerical gate, with no accepted timing. Its failed2048-token logit check and original evidence remain unchanged.

[E013](docs/EXPERIMENTS.md#e013-final-results-and-decision) remains rejected: the smaller M128 tile regresses 7.6–8.7% across both rectangles/two reversed series. **Keep corrected Marlin A and its default tile.** Historical evidence and environments remain unchanged.

The preceding [E012 diagnostic](docs/EXPERIMENTS.md#e012-final-findings-and-decision) remains inconclusive: ambiguous instruction callers and failed sampling-quality gates justify no kernel rewrite. Its original records remain unchanged.

C = A·B with A an M×K FP16 activation matrix and B a K×N weight matrix stored as
packed INT4 with group-wise FP16 scales. Fused kernels dequantize in registers,
accumulate in FP32, and write FP16. This is a controlled implementation and measurement study. The retained scalar
kernels and pinned external Marlin comparator have separate evidence and
performance conclusions.

## Numerical contract

Reconstructed weight, identical on the CPU reference, the GPU dequant kernel,
the fused kernel, and the pre-dequantized cuBLAS baseline:

```
s     = round_to_fp16( max_{k in group} |W[k,n]| / 7 )
z     = clamp( nearbyint(W/s), -7, +7 )     # 0 if s == 0
q     = z + 8                               # stored code in {1,...,15}
W_hat = round_to_fp16( (q - 8) * s )        # FP16 bits
```

Code 0 is unused. The range is symmetric about zero: the extreme codes 1 and 15
decode to −7s and +7s. `(q-8)*s` is exact in FP32 but is rounded to FP16 before
use; the kernel converts that FP16 back to FP32 for the FMA. The benchmark
checks every W_hat element: GPU dequant bits must match the host bits.

## Memory sizes (K = N = 4096, group = 128)

| Buffer        | Formula                         | Size    |
|---------------|---------------------------------|---------|
| FP16 weights  | `2·K·N`                         | 32 MiB  |
| packed INT4   | `K·N/2`                         | 8 MiB   |
| FP16 scales   | `2·(K/group)·N`                 | 0.25 MiB|
| L2 on c7      | `cudaDeviceProp.l2CacheSize`    | measured at runtime |

On the RTX 5060 Ti (c7), `cudaDeviceProp.l2CacheSize` and Nsight
`device__attribute_l2_cache_size` are both 32 MiB. `nvidia-smi -q` does not
print L2. The 8.25 MiB packed working set for M=1 fits in that L2. The default
experiment uses an **intended cold policy**: a store kernel writes `2·L2` bytes on
the same stream immediately before the CUDA start event. Allocations, host
quantization, and H2D copies are never inside the timed interval. The unfused
path does **not** flush between dequant and GEMM, so W_hat can stay in cache.
The store flush does not prove input eviction. Compare it with `--no-flush-l2`,
and keep Nsight `--cache-control all` profiling separate from event timing.

## Metrics

- **median/min/max/p90 µs**: CUDA event elapsed time over `--iters` launches
  after `--warmup` untimed launches.
- **TFLOP/s**: `2·M·N·K / median_s / 1e12` for GEMM-like kernels; 0 for `dequant`.
- **eff_GB/s**: `logical_bytes / median_s / 1e9`. This is **effective
  bandwidth** from a traffic model, not measured DRAM. Formulas:
  - `naive` / `inblock` / `inblock_specialized` / `inblock_m2` / `inblock_m4`: \|A\| + \|C\| + \|qweight\| + \|scales\|
  - `cublas` / `cublas_dq`: \|A\| + \|C\| + \|W_fp16\|  (32 MiB weights)
  - `dequant`: \|qweight\| + \|scales\| + \|W_hat\|     (W_hat is a write)
  - `unfused`: \|A\| + \|C\| + \|qweight\| + \|scales\| + \|W_hat\|
    (W_hat write+read counted once; ncu reports actual DRAM separately)
- **pass**: every checked output satisfies `|got-ref| ≤ 1e-3·|ref| + 1e-3·rms(ref)`.
  `got` is the FP16 output; `ref` is an FP64 CPU GEMM. Fused / `cublas_dq` /
  `unfused` use reconstructed W_hat; `cublas` uses the original FP16 W.
- **rel_l2**: `‖got-ref‖₂ / ‖ref‖₂` against that kernel's reference.
- **rel_l2_vs_fp16**: same formula against the original-weight CPU GEMM
  (quantization error of W_hat, not a kernel bug).

## Baselines

1. FP64 CPU GEMM on reconstructed W_hat (correctness only).
2. `cublas`: cuBLAS FP16 GEMM, `CUBLAS_COMPUTE_32F`, original W.
3. `cublas_dq`: same cuBLAS call on pre-dequantized W_hat already on device.
4. `dequant` + `unfused`: GPU dequant latency, GEMM-on-W_hat latency, and
   combined latency around both operations.
5. `naive`: fused W4A16, one thread per output, no shared-memory reuse.
6. `inblock`: fused W4A16, block=(32,8), eight disjoint packed-word K slices,
   FP32 shared-memory reduction and one FP16 write per output.
7. `inblock_specialized`: E002 version with compile-time aligned/tail variants.
   The E001 `inblock` source and selector remain unchanged for comparison.
8. `inblock_m2`: E003, two output rows per thread sharing each reconstructed weight.
9. `inblock_m4`: E003, four output rows per thread sharing each reconstructed weight.
   E002's one-row source remains unchanged. Both candidates pass the larger-M
   criterion, while small-M and row-tail regressions remain documented in
   [E003](docs/EXPERIMENTS.md#e003--reuse-reconstructed-weights-across-input-rows).

All fused kernels and GPU dequantization handle arbitrary positive K and N,
including partial final words and groups. Group size remains a multiple of eight.
Packed/scales storage uses ceiling counts. Select the new version with
`--kernels inblock_specialized` or compare with
`--kernels inblock,inblock_specialized`. Dispatch uses K divisibility by eight,
including aligned K with a partial final quantization group. Compare row reuse with
`--kernels naive,inblock_specialized,inblock_m2,inblock_m4`. Row variants use
block=(32,8), grid y=ceil(M/Rows), and 2048/4096 bytes of static shared partials.
They guard incomplete row tiles and retain each row's arithmetic order. Untimed
validation compares every candidate output bit against E002, including rows
outside the sampled FP64 reference. No automatic row-count selection is added.

E004 integrates pinned vLLM Marlin as a direct external comparator, with lossless
repacking, isolated dependencies, strict all-row numerical checks, and a fresh
performance map. See the [E004 record](docs/EXPERIMENTS.md#e004--maintained-marlin-integration-and-diagnostic-comparison)
and [current status](docs/STATUS.md) for results, failed attempts, reproducible
commands, and provenance. E005 implements the bounded A/B/C reservation/grid
ablation in an isolated harness, then stops at a racecheck report reproduced in
its unchanged control and the official wheel. Numerical checks, memcheck and
synccheck pass, but E005 performance and held-out evaluation remain unmeasured.
See the [E005 record](docs/EXPERIMENTS.md#e005--specialization-aware-reservation-and-one-versus-two-block-residency).
Existing project kernels and the E004 wheel remain unchanged.

E006 investigates the inherited report and adds one source-supported block barrier
before shared-memory reuse. The corrected matched A/B/C body passes the recorded
numerical and sanitizer matrix. The unpatched reproducer remains preserved and
still reports hazards. Original output corruption was not demonstrated, and no
performance measurements were resumed. See the [E006 record](docs/EXPERIMENTS.md#e006--bounded-inherited-marlin-racecheck-investigation)
and [access analysis](results/e006_20261106/ACCESS_ANALYSIS.md). Any corrected-body
E005 performance continuation is a separate next decision.

Each new session starts with [AGENTS.md](AGENTS.md), [current status](docs/STATUS.md),
the [numerical and measurement contract](docs/CONTRACT.md), and relevant
[experiment and design decisions](docs/EXPERIMENTS.md). The
[optimization map](docs/OPTIMIZATION_MAP.md) connects candidate approaches to
evidence, and [references](docs/REFERENCES.md) records source applicability.
Historical corrections, unsuccessful attempts, raw result paths, and revisit
conditions live in the experiment log.

## Build and run

```bash
bash scripts/build.sh       # host-only if nvcc is absent
ctest --test-dir build --output-on-failure
./build/w4a16_bench --M 16 --N 4096 --K 4096 --group 128 --csv results/run.csv
```

On ARC, verify the current allocation, use `prun` inside it, and follow E001's
[reproduction record](docs/EXPERIMENTS.md#e001--in-block-k-splitting). Run
`bash scripts/check_inblock.sh` on the allocated GPU before measuring. Save new
results in a distinct directory. Legacy `scripts/remote.sh` defaults to c7 and
uses `rsync --delete`. Legacy baseline scripts overwrite historical records.
