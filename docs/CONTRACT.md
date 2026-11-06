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

Supported logical shapes have positive M, N, K and group, group divisible by eight, subject to available memory and index/launch limits. K may be below eight. Final words/groups may be partial and N may be odd. Unused tail nibbles are zero and must never be consumed. No new packed layout is part of E001. Synthetic inputs are finite FP16 values. NaN/Inf quantization is not a specified interface.

## 2. Reference and correctness

| Kernel | Operation and reference |
|---|---|
| naive/inblock | Fused GEMM on W_hat, reconstructed-weight FP64 CPU GEMM reference |
| cublas | Original FP16 W, `CUBLAS_COMPUTE_32F`, original-weight FP64 reference |
| cublas_dq | Pre-dequantized W_hat already on device, same cuBLAS call, reconstructed-weight FP64 reference |
| dequant | Materialize FP16 W_hat, bitwise agreement with every CPU-reconstructed weight |
| unfused | GPU dequantization then cuBLAS, reconstructed-weight FP64 reference |

For every checked output require `abs(got-ref) <= 1e-3*abs(ref) + 1e-3*rms(ref)`. Got is FP16 output promoted to FP32. Ref is FP64, with RMS over checked reference elements. `rel_l2_err = norm(got-ref)/norm(ref)`. `rel_l2_vs_fp16` compares with the original-weight reference and includes quantization error for W_hat kernels. It is not the pass criterion. Never relax tolerance to accept changed reduction order.

Small cases check every row. Main sweeps use up to 64 deterministically sampled rows, with exact indices printed in the log, and every column of those rows. M=128 main runs therefore do not check every output row. Every W_hat element is checked bitwise when GPU dequantization runs. Do not use `--nocheck` for acceptance measurements. Stop on correctness/launcher errors and preserve diagnostics.

## 3. Timing and comparisons

Use M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, and at least 100 timed iterations. Compare naive/inblock in the same executable, build, seed, node, and cache policy. Retain and record each kernel's defined launch mapping. Changing executable, flags, kernel, rank/thread count, placement, or topology requires rerunning every M point in the series.

CUDA events surround one launch or the complete unfused pair. Allocations, host quantization, H2D copies, and flushing are outside the interval. Report median, min, max, and p90 in microseconds. Even-count median averages the middle pair. p90 uses sorted index `floor(0.9*(iters-1))`. Speedup is `naive median_us / inblock median_us` at the same M and policy. Within-run ranges do not establish between-run variability.

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
| naive/inblock | `|A|+|C|+|Q|+|S|` |
| cublas/cublas_dq | `|A|+|C|+|W|` |
| dequant | `|Q|+|S|+|W_hat|` |
| unfused | `|A|+|C|+|Q|+|S|+|W_hat|` |

The unfused model counts W_hat once although it is written and then read. Preserve this historical CSV convention and identify it when interpreting `eff_gbps`. A model counting both logical transfers adds another `|W_hat|`, but neither model is measured DRAM traffic. A changed model must be separately labeled.

For K=N=4096 and group=128, Q=8 MiB, S=0.25 MiB, W_hat=32 MiB. The 8.25 MiB packed weight/scale set fits the observed 32 MiB L2. Use finite ceiling counts for tail shapes. If adding rooflines, recompute every coordinate/ceiling, distinguish primary full-machine from auxiliary roofs, and keep operation/ISA/clock/bandwidth assumptions consistent.
