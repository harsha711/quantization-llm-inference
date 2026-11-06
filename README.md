# W4A16 mixed-precision GEMM (CSC 548)

C = A·B with A an M×K FP16 activation matrix and B a K×N weight matrix stored as
packed INT4 with group-wise FP16 scales. Fused kernels dequantize in registers,
accumulate in FP32, and write FP16. This is a controlled implementation and
measurement study, not a claim of novelty or of a speedup over cuBLAS.

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
experiment is therefore **cold-cache**: a store kernel writes `2·L2` bytes on
the same stream immediately before the CUDA start event. Allocations, host
quantization, and H2D copies are never inside the timed interval. The unfused
path does **not** flush between dequant and GEMM, so W_hat can stay in cache.

## Metrics

- **median/min/max/p90 µs**: CUDA event elapsed time over `--iters` launches
  after `--warmup` untimed launches.
- **TFLOP/s**: `2·M·N·K / median_s / 1e12` for GEMM-like kernels; 0 for `dequant`.
- **eff_GB/s**: `logical_bytes / median_s / 1e9`. This is **effective
  bandwidth** from a traffic model, not measured DRAM. Formulas:
  - `naive` / `inblock`: \|A\| + \|C\| + \|qweight\| + \|scales\|
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

Both fused kernels and GPU dequantization handle arbitrary positive K and N,
including partial final words and groups. Group size remains a multiple of eight.
Packed/scales storage uses ceiling counts. Select the new version with
`--kernels inblock` or compare with `--kernels naive,inblock`.

Project continuation rules are in [AGENTS.md](AGENTS.md). See
[design decisions](docs/design-decisions.md) for considered designs and evidence,
and the [experiment record](docs/experiment-inblock.md) for validation and results.

## Build and run

```bash
scripts/build.sh            # host-only if nvcc is absent
./build/test_cpu
./build/w4a16_bench --M 16 --N 4096 --K 4096 --group 128 --csv results/run.csv
```

On ARC (needs an allocation on the node):

```bash
scripts/remote.sh scripts/build.sh
scripts/remote.sh scripts/run_baseline.sh
scripts/remote.sh scripts/ncu_baseline.sh
```

These legacy remote commands default to c7 and overwrite baseline records.
For the new experiment, use the verified current allocation and a separate result
directory as described in the experiment record. Run `bash scripts/check_inblock.sh`
on the GPU before measuring.
