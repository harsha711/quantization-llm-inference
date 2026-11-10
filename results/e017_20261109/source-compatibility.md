# E017 source compatibility inspection

This is a source inspection, not fresh operator or full-model validation. No source-level K/N/M incompatibility is identified for the predeclared 196 transformer-block projections. Actual module inventory, runtime dispatch, captured operands, numerical checks and fresh sanitizers remain prerequisite gates.

## Preserved implementation identity

The corrected E006 archive is `results/e006_20261106/matched-source.tar.gz`, SHA256 `e08be8352dfb500f0417bc8e8eedc74fbf3036ed646a2215713018ebac846e6a`. Its `external/patched/csrc/quantization/marlin/` members have these SHA256 values:

| File | SHA256 |
|---|---|
| `marlin.cu` | `9b5b16aa3047abf58d845c2aa913b86ad89ab8f08dd140c4d22a3bdc7cf9b7d2` |
| `marlin_template.h` | `64d8ee386465ead3c757866143d99b40056b0f7698a147c92c09094a4d0f5dec` |
| `kernel_selector.h` | `7cbc2ce052284965d529101c6a8e55fb8fcbcd6770d3e86c05d9907f529a9d93` |

The preserved corrected library is SHA256 `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`, as recorded in `results/e006_20261106/patched/build-source.sha256` and `analysis.json`. Its historical path is `/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so`. Future availability and live identities must be verified. Wrapper `scripts/e005/local.cu` is SHA256 `122e8680e98ab66118b5d8ce5f3fc1d43245d44be4c0d8cbe4635600485183b7`.

## ABI, layouts and model coverage

`scripts/e005/local.cu:13–28` exposes `e005_call(mode,dry,A,packed,scales,C,scratch,locks,M,N,K,stream,info)`. It requires positive M/N, K>=256, K divisible128 and N divisible64. It fixes contiguous row-major A/C with lda=K, FP16 inputs/scales/output, U4B8 weights, G128, no zero point, no fused bias, no atomic reduction and FP32 global reduction. Mode0 is original A. It does not accept an arbitrary PyTorch input stride. Flatten leading batch/sequence dimensions to M and ensure contiguous FP16 activation storage before this ABI. Actual layers must match the declared K/N and output reshape.

Declared shapes are K1536/N1536 (q/o), K1536/N256 (k/v), K1536/N8960 (gate/up), and K8960/N1536 (down). All satisfy the ABI and packing divisibility. `scripts/e005/prepare_source.py:79–85` generates bounded selectors for FP16/U4B8/G128, thread_m_blocks1–4, the existing small/large tile families and eight-row decode. Source selectors do not impose a maximum M128. Prefill M128/M2048 and decode M1 still require actual execution validation.

Pinned `results/e014_20261108/runtime-source/transformers.models.qwen2.modeling_qwen2.py:134–137` declares q/k/v with bias and o without bias. Its MLP line40–42 declares gate/up/down without bias. Both replacement variants must perform the same explicit FP16 addition after their FP16 GEMM output for q/k/v. This preserves matched replacement semantics without assuming the original fused FP16 linear bias boundary is bitwise identical. The original FP16 model remains the separate quantization-change comparison. The model declares tied `lm_head.weight` at line404 and a separate head module at line412. Exclude both embedding and LM head from replacement and inventory actual tied storage rather than counting aliases twice.

## Dispatch and workspace observations

Archived corrected `marlin.cu:310–319` sets max_par16, increased to128 for N<=4096, and splits large M at max_par*64 rows. Thus M2048/N8960 normally launches two1024-row operations. M2048/N1536 or256 can remain one operation. With the declared 28 blocks, 56 gate/up module calls therefore add56 prefill launches, yielding252 Marlin launches for196 selected M2048 projections if no configuration-capacity fallback changes the split. Verify this from the actual trace. The dry-run return at line442 stops after the first split and does not enumerate every actual launch.

Archived `marlin.cu:344–354` can replace the initial selected tile with the existing128K×64N/128-thread configuration for low parallelism. N256 k/v can exercise this path at small M. It may use a different dispatch family at M128 and M2048. Representative down_proj coverage alone therefore does not establish coverage of all declared shapes. Inspect actual resource selections and validate the distinct families.

`scripts/e005/bench.py:69–77` retains4*SMs zero-initialized int32 locks and `SMs*min(ceil(maxM/16)*16,64)*256` FP32 scratch elements, with guards. `check_workspace` at line92 verifies zero locks and intact guards after repeated calls. Same-stream ordered calls can reuse scratch/locks, but outputs need distinct valid lifetimes while q/k/v or MLP intermediates remain live. On36 SMs, the maximum scratch capacity is589824 FP32 elements (2.25MiB), plus guards. Avoid retaining a full FP16 reconstructed GPU weight in the compressed candidate merely for validation.

## Existing contract reuse

`src/quant.cpp:32–66` is the authoritative quantizer/reconstructor: FP16 maxabs/7 scale, FP32 division, nearest-even rounding, clamp[-7,7], offset8 and FP16 reconstructed multiplication. PyTorch `nn.Linear` stores N×K, so transpose once to contiguous K×N before applying this contract. Existing E004 generation exports synthesize weights and cannot directly quantize an arbitrary model weight. A minimal host wrapper can call the existing quantizer rather than substitute a new numerical rule.

Reuse `scripts/e004/bench.py:26–49` for inspected pure packing helpers and inverse unpacking. `scripts/e005/bench.py:54–63` demonstrates repack, exact code recovery, inverse scale permutation and FP16 reconstructed-weight bit checks. The wheel is needed for repacking only. Corrected GEMM execution must use the verified local library through the current PyTorch CUDA stream, as shown at `scripts/e005/bench.py:83–89`. Full-model numerical agreement is not implied by this source inspection.
