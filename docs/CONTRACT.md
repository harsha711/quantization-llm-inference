# Numerical and measurement contract

## 1. Operation, layout, and quantization

Compute row-major `C[M,N] = A[M,K] * B[K,N]`. A and C use FP16 storage. Original W is FP16. Fused kernels accumulate with FP32 FMA and round the final result to FP16. The CPU reference accumulates in FP64. The study measures controlled kernels and baselines for LLM-like GEMM shapes, not end-to-end model inference.

For each column n and K-group g, quantize with the stored FP16 scale:

```text
s[g,n] = FP16(max(abs(W[k,n])) / 7), over valid k in group g
z[k,n] = clamp(nearbyint(FP32(W[k,n]) / FP32(s[g,n])), -7, 7)
          or 0 when the stored scale is zero
q[k,n] = z[k,n] + 8
W_hat[k,n] = FP16(FP32(q[k,n] - 8) * FP32(s[g,n]))
```

FP16 conversion uses round-to-nearest-even. `nearbyint` follows the host floating-point rounding mode, normally nearest-even. Do not change that environment silently. Valid weights have codes 1 through 15. Code 8 represents zero. Code 0 is unused for valid weights.

**FP16 reconstruction is required.** Round the FP32 product `(q-8)*s` to FP16 before promotion to FP32 for the fused multiply-add. Keeping the product in FP32 computes a different operation from the CPU reference, GPU dequantization, and pre-dequantized cuBLAS path. See H001 in [EXPERIMENTS.md](EXPERIMENTS.md).

Packed storage is `qweight[k8*N+n]`, a uint32 word containing eight consecutive K codes in nibble positions `4*j`, increasing with K. Adjacent columns occupy adjacent words. Allocate `ceil(K/8)*N` words. Scales use `scales[g*N+n]`, with `ceil(K/group)*N` FP16 elements. Words never cross group boundaries because group size is a multiple of eight.

Supported logical shapes have positive M, N, K and group, group divisible by eight, subject to available memory and index/launch limits. K may be below eight. Final words/groups may be partial and N may be odd. Unused tail nibbles are zero and must never be consumed. E001–E003 preserve the same packed layout and quantization group rules. Synthetic inputs are finite FP16 values. NaN/Inf quantization is not a specified interface.

### Launch mapping and row tiles

The selectors retain fixed mappings. Static shared bytes below are source-declared partial arrays, excluding any driver-reserved storage. Record actual compiled resources separately.

| Selector | Rows per block tile | Block | Grid | Static partial bytes/block |
|---|---:|---|---|---:|
| naive | 1 | 256 threads | `(ceil(N/256), M)` | 0 |
| inblock / inblock_specialized | 1 | `(32,8)` | `(ceil(N/32), M)` | 1024 |
| inblock_m2 | 2 | `(32,8)` | `(ceil(N/32), ceil(M/2))` | 2048 |
| inblock_m4 | 4 | `(32,8)` | `(ceil(N/32), ceil(M/4))` | 4096 |

E003's `inblock_m2` and `inblock_m4` reuse each loaded packed word, scale, and FP16-reconstructed weight across two or four independent row accumulators. Column lanes remain contiguous. Each of eight K slices visits packed words `[floor(W*y/8), floor(W*(y+1)/8))`, where `W=ceil(K/8)` and y is threadIdx.y. Per-row K traversal, FP32 FMA order, eight-partial summation order, and FP16 output conversion remain those of E002. Shared partials are `float partial[Rows][8][32]`. Rows never reduce into each other.

The row tile begins at `m=blockIdx.y*Rows`. Compile-time K-tail and M-tail variants are selected by `K%8` and `M%Rows`. When the final row tile is incomplete, invalid rows keep zero accumulators, skip activation reads, and skip output writes. Invalid N lanes and empty K slices also publish zero partials. Every thread reaches the block barrier. Valid rows retain their ordinary arithmetic order. M need not be divisible by two or four, and K may end in a partial word or group. K=136 with group=128 uses an aligned packed-word path despite its partial final group. These bounds specializations do not select between the two-row and four-row kernels automatically.

## 2. Reference and correctness

| Kernel | Operation and reference |
|---|---|
| naive/inblock/inblock_specialized/inblock_m2/inblock_m4 | Fused GEMM on W_hat, reconstructed-weight FP64 CPU GEMM reference |
| cublas | Original FP16 W, `CUBLAS_COMPUTE_32F`, original-weight FP64 reference |
| cublas_dq | Pre-dequantized W_hat already on device, same cuBLAS call, reconstructed-weight FP64 reference |
| dequant | Materialize FP16 W_hat, bitwise agreement with every CPU-reconstructed weight |
| unfused | GPU dequantization then cuBLAS, reconstructed-weight FP64 reference |

For every checked output require `abs(got-ref) <= 1e-3*abs(ref) + 1e-3*rms(ref)`. Got is FP16 output promoted to FP32. Ref is FP64, with RMS over checked reference elements. `rel_l2_err = norm(got-ref)/norm(ref)`. `rel_l2_vs_fp16` compares with the original-weight reference and includes quantization error for W_hat kernels. It is not the pass criterion. Never relax tolerance to accept changed reduction order.

Small cases check every row. Main sweeps use up to 64 deterministically sampled rows, with exact indices printed in the log, and every column of those rows. M=128 main FP64 comparisons therefore do not check every output row. Every W_hat element is checked bitwise when GPU dequantization runs. Do not use `--nocheck` for acceptance measurements. Stop on correctness/launcher errors and preserve diagnostics.

In addition to FP64 validation, `inblock_specialized` is compared bitwise against E001 `inblock`, and E003 `inblock_m2`/`inblock_m4` are each compared bitwise against E002 `inblock_specialized`. These untimed checks cover every M*N FP16 output, even when the FP64 reference samples rows. They run when the candidate is checked, without requiring its comparison selector in the timed list. Any mismatch stops the run before candidate timing. Full-output agreement with a retained kernel does not expand the sampled FP64 check to all rows.

## 3. Timing and comparisons

Use M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, and at least 100 timed iterations. Compare naive/inblock/inblock_specialized/inblock_m2/inblock_m4 in the same executable, build, seed, node, and cache policy. Retain and record each kernel's defined launch mapping. Changing executable, flags, kernel, rank/thread count, placement, or topology requires rerunning every M point in the series.

E003 adds a separately identified supplemental M=5 point with the same K=N=4096, group=128, seed=1, 10 warmups, at least 100 timed iterations, and both cache policies. It tests incomplete two-row and four-row tiles at the main matrix dimensions. It does not replace or redefine the standard M={1,8,32,128} series. The current event-timing script runs all nine selectors at those four standard points and the supplemental point.

CUDA events surround one launch or the complete unfused pair. Allocations, host quantization, H2D copies, and flushing are outside the interval. Report median, min, max, and p90 in microseconds. Even-count median averages the middle pair. p90 uses sorted index `floor(0.9*(iters-1))`. E001 speedup is `naive median_us / inblock median_us`. E002 speedup is `inblock median_us / inblock_specialized median_us`. E003 evaluates each selector separately with `inblock_specialized median_us / inblock_m2 median_us` or `inblock_specialized median_us / inblock_m4 median_us`. Compare the same M and policy. E002 retains the E001 mapping and arithmetic order. E003 changes rows per tile while retaining each row's arithmetic order. Both experiments require full-output bitwise agreement with their retained comparison, checked outside timing in addition to the unchanged FP64 threshold. Within-run ranges do not establish between-run variability.

Record both application policies:

- Intended cold: a store kernel writes `2*L2` bytes on the same stream immediately before each start event. Historical L2=32 MiB gives a 64 MiB buffer. This does not demonstrate eviction of each input line and may affect later DRAM activity.
- Reused inputs: `--no-flush-l2`, repeated launches without application flushing. This does not guarantee permanent cache residency.

There is no flush between dequantization and GEMM in `unfused`, so W_hat may remain cached. Nsight `--cache-control all` flushes caches between replay passes and is a separate method. Profile separately from latency runs and query metrics on the target GPU. Profiler-replayed and event timings are not directly interchangeable.

## 4. Operation and byte models

GEMM-like TFLOP/s is `2*M*N*K / median_seconds / 1e12`, counting multiply and add separately. Dequant reports zero TFLOP/s. Define buffer sizes in bytes:

```text
|A| = 2*M*K                   |C| = 2*M*N
|Q| = 4*ceil(K/8)*N            |S| = 2*ceil(K/group)*N
|W| = |W_hat| = 2*K*N
```

`eff_gbps = logical_bytes / median_seconds / 1e9` uses the existing model below. It is not measured DRAM bandwidth and omits repeated cache accesses across output rows.

| Path | Existing logical_bytes model |
|---|---|
| naive/inblock/inblock_specialized/inblock_m2/inblock_m4 | `|A|+|C|+|Q|+|S|` |
| cublas/cublas_dq | `|A|+|C|+|W|` |
| dequant | `|Q|+|S|+|W_hat|` |
| unfused | `|A|+|C|+|Q|+|S|+|W_hat|` |

The unfused model counts W_hat once although it is written and then read. Preserve this historical CSV convention and identify it when interpreting `eff_gbps`. A model counting both logical transfers adds another `|W_hat|`, but neither model is measured DRAM traffic. A changed model must be separately labeled.

For K=N=4096 and group=128, Q=8 MiB, S=0.25 MiB, W_hat=32 MiB. The 8.25 MiB packed weight/scale set fits the observed 32 MiB L2. Use finite ceiling counts for tail shapes. If adding rooflines, recompute every coordinate/ceiling, distinguish primary full-machine from auxiliary roofs, and keep operation/ISA/clock/bandwidth assumptions consistent.

## 5. E004 external comparator and verified timing method

E004 selects released vLLM v0.33.0, exact source `4ba787ec9959e566d3b587a796dff07673324da5`, rather than the inspected moving main commit. The wheel's generated version file reports `g4ba787ec9`. Use dense `torch.ops._C.marlin_gemm`, FP16 A/C/scales and `uint4b8`, group=128, no bias/activation scale/global scale/zero-point tensor, `use_atomic_add=False`, `use_fp32_reduce=True`, `is_zp_float=False`. This is INT4 W4A16, not NVFP4, MXFP4, W4A4, W4A8 or MoE. Import only the released operator library and inspected pure source helpers, without importing a serving engine.

Existing packed `[K/8,N]` words are reinterpreted as int32 bits and passed to upstream `gptq_marlin_repack(...,num_bits=4,is_a_8bit=False)`. Offset codes are unchanged. The upstream helper reshapes logical weights into 16x16 tiles and permutes 1024-code chunks for Tensor Core fragments, then packs eight permuted codes per word. Output storage is `[K/16,2*N]` int32. Grouped scales retain all FP16 bits while each 64-element chunk is permuted by `[i+8*j for i in range(8) for j in range(8)]`. No logical matrix transpose or independent requantization is performed. CPU inversion must recover every code and scale bit, then reconstruct every original W_hat FP16 element. GPU dequantization of the project's original packed layout is independently bit-checked.

Source inspection finds exact FP16 integers q-8 followed by `__hmul2` of group scales before MMA. On sm_120 the grouped FP16 path uses FP32 MMA accumulation. `use_fp32_reduce=True` also preserves FP32 global partials, whereas false permits FP16 intermediate storage. Arithmetic order differs from retained scalar kernels, so compare Marlin to the reconstructed-weight FP64 reference at the unchanged threshold, not bitwise C agreement. Original-weight cuBLAS has its separate original-weight reference. E004's adapter checks all rows up to M=128, independently expanding verification for this experiment without changing old sampled claims.

Repack requires K%16=0 and N%64=0. Execution additionally requires valid configured K/N tiles and group divisibility. All requested 4096/11008 shapes satisfy the inspected source conditions and passed actual execution without padding. Avoid K=128 in grouped smoke tests: one scale row is interpreted as per-column quantization and scales after reduction. Use K=256 or larger. E004 does not extend Marlin with padding. Locks use 4*SMs zero-initialized int32 elements, reusable on ordered calls. Preallocate output. The operator still allocates FP32 C_tmp internally: `4*SMs*min(ceil(M/16)*16,64)*256` bytes, plus empty optional tensor wrappers. Record these allocations outside device timing and diagnose host dispatch gaps.

E004 adds a separately labeled common `graph_events` timing method for every selector. Capture store flush (when enabled), external CUDA start event, GEMM operation(s), and external stop event for each sample in a CUDA graph, then replay. Per-sample CUDA event intervals exclude store flushing, host dispatch, allocation, repacking and capture. The no-flush policy still omits application flushing and the store policy still writes exactly 2*runtime L2 bytes with the retained kernel. Event-node overhead and graph scheduling differ from historical E001–E003 application timings, so rerun every E004 comparator/shape and never combine old medians as denominators. A separately labeled eager-event diagnostic tests host-gap sensitivity. Graph replay timings remain distinct from cache-invalidated profiler replay. Record capture/setup costs separately. No tuning or external kernel modification is included.

E004's all-row screen exposed two original-weight cuBLAS errors under the legacy default handle math mode at M=128, K=N=4096. A bounded mode 0/16 ablation reproduced those errors and passed with `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` (value 16), while retaining `CUBLAS_COMPUTE_32F`. The complete E004 comparison sets mode 16 explicitly for original/reconstructed/unfused cuBLAS and labels it in every result. This forces intermediate reductions to compute precision. The retained source and E001–E003 behavior remain unchanged. Do not splice the partial default-mode screen into the complete rerun or attribute all differences solely to precision rather than algorithm selection.


E004 validation is complete on c58 sm_120: all 384 screening rows, both 64-row repeats and both 12-row timing-method diagnostics pass at the unchanged threshold. Code and scale inversion recover every bit and source grouped reconstruction matches the FP16 contract. Traces/profiles establish actual Marlin selection. The all-row reference checks are specific to this new adapter and do not expand historical sampled checks. Repack first-use wall/event costs and capture/first-validation-call costs are saved separately. Setup event intervals can contain lazy initialization and host gaps and are not isolated repack-kernel throughput.

E004's three Nsight reports use cache-control none after 10 direct warmups and kernel replay, no application flushing and unlocked clocks. They are not the application graph-event sample series. Counters retain 2.4–2.5 MB DRAM reads, so neither complete residency nor a unique cold/warm classification is established. `profile_summary.csv` normalizes Kbyte/Mbyte to decimal bytes, retaining exact metric names. Tensor-pipe activity is not FLOP utilization. No roofline ceiling or classification was constructed.


## 6. E005 experimental launch policy and stopped gate

E005 uses the unchanged E004 FP16 reconstruction, dense uint4b8/G128 format, FP32 local/global reduction, atomic=False, no optional scales/bias/zero-points, exact lossless repacking and unchanged all-output FP64 condition. It does not alter retained kernels, cuBLAS settings or the official E004 wheel. The local build calls the original upstream dispatcher through a narrow typed C ABI. Only FP16/uint4b8/G128 kernel selections are instantiated. A preserves original selection/accounting. B/C activate only for sm_120, M<=8, the selected eight-row 128x128 tile,256 threads,four stages and those fixed numerical settings. Other selections, including small-N fallback, M16 and M64, retain original launch policy.

Reservation is device-derived opt-in block shared size for A, and opt-in/2-1024 for B/C, checked against 41984 B enabled spatial layout and actual CUDA two-block occupancy capacity. On the verified device A requests 101376 B with 36 CTAs, B 49664 B with 36, and C 49664 B with 72. The same compiled device entrypoint serves all three. Dynamic allocation is not the used footprint. Maximum occupancy is not achieved residency. C changes striped K partitioning, inter-block locks and global FP32 reduction order, so FP16 output-bit identity with A is not required. Fallback A/B/C output bits are checked outside timing.

The caller allocates4*SM zeroed int32 locks plus guards, and `SM*min(ceil(maxM/16)*16,64)*256` FP32 scratch elements plus guards for each validation invocation. At maxM<=8 this is 589824 scratch bytes on 36 SMs. A/B/C share these allocations. Explicit index proof bounds the 72-CTA rectangle's maximum exclusive accessed scratch byte at 580608 and its maximum lock index at 70. First reducers initialize scratch and last reducers reset locks. Ordered same-stream calls reuse storage with no per-call memset. Initial locks/scratch/output poisoning and guards are outside capture. Tests poison C before each graph replay and check every output, lock reset and guards afterward.

E005 did not reach performance measurement. Its prepared common-event plan preserves E004's explicit graph stream, per-sample external CUDA event nodes and both application cache policies, with device-required maintenance inside the operation. Required setup remains outside timing. `measure_arc.sh` requires successful validation and a matching binary hash, but the VALIDATED sentinel is absent. Correctness JSONL policy labels do not indicate that cache flushing or timing was performed when iters=0. No E005 samples/medians/p90 or profiler counter series exists.

External memcheck/synccheck pass for the recorded representative cases. Racecheck fails on A/B/C and reproduces in unchanged local A and isolated official wheel at main square M8. Source spatial bounds do not establish safe temporal B/reduction reuse. Passing numerical checks and other sanitizers do not justify ignoring this report. Preserve the failed gate and stop dependent performance/held-out profiling work. Any safety/build/body change requires a separately authorized resolution, a new matched denominator and the full required series.
