# Design decisions: in-block K splitting

## 1. Scope and starting evidence

The requested experiment retains the naive W4A16 kernel and adds one separately selectable in-block reduction. Baseline commit `8a68227` was clean before work started. The working branch is `optimize/inblock-k`. Historical measurements remain in their original files. They establish context, but speedups must use rerun naive measurements on the same node as the new kernel.

Inspection confirmed `qweight[k8,n]` with adjacent columns contiguous in memory. Each naive thread owns one output and traverses all K values. At N=4096, its grid has 16 blocks per row. The original host validator required K divisible by a group size that is itself divisible by eight. Its packing, dequantizer, and byte models all truncated K tails. Consequently, the handoff's small-K and partial-group requirements required a shared tail extension before meaningful GPU validation.

## 2. Alternatives considered

The following alternatives explain the selected experiment. Unimplemented alternatives have no measured performance result.

| Design | Decision | Reason and limit |
|---|---|---|
| Retain one output per naive thread | Keep as baseline | Preserves the original comparison. Long K accumulation and only 16 blocks at M=1 remain. |
| Eight contiguous packed-word K slices, block=(32,8) | Implement | Adjacent lanes load adjacent columns. Each warp owns one K interval. Adds 128 blocks per row and shortens each thread's FP32 dependency chain. Performance and occupancy still require measurement. |
| Interleave packed words among the eight warps | Considered, not implemented | Also covers K, but changes the access order without evidence that it improves this first experiment. Contiguous intervals are straightforward to audit. |
| Cross-block split-K, atomics, or global partials | Excluded | Outside the requested experiment. Would introduce another launch or global reduction mechanism. |
| Tensor Cores, asynchronous copies, new layout, or A/W tiling | Deferred | These change additional mechanisms and prevent attribution to in-block K splitting. |
| Warp shuffle reduction | Not selected | The eight partials for one column reside in different warps. Shared memory directly expresses the required cross-warp communication. |
| Reject all tail shapes as before | Not selected | Cannot satisfy the required correctness cases. Extend shared packing, quantization, dequantization, and naive bounds instead. |
| Arbitrary non-multiple-of-eight group sizes | Not implemented | The existing layout guarantees one scale per packed word. Partial final groups do not require changing that guarantee. |

## 3. Chosen mapping and numerical behavior

For W=ceil(K/8) packed words and slice y in [0,7], the warp traverses words in [floor(W*y/8), floor(W*(y+1)/8)). These half-open intervals cover every word exactly once, including when W<8 or W is not divisible by eight. Thread x owns column n=32*blockIdx.x+x. BlockIdx.y selects row m. Each thread reconstructs every valid weight through FP16, then uses FP32 FMA with its A value. It writes one FP32 partial to `partial[y][x]`.

All 256 threads reach the barrier. Invalid N lanes and empty slices contribute zero. After the barrier, warp zero adds the eight FP32 partials for its columns in slice order and writes each output once as FP16. The partial array occupies 8*32*4=1024 bytes. There are no atomics or global partial outputs. The group scale, nearbyint quantization, dequantization rounding, accumulation precision, and output precision are preserved. The addition order changes and is checked using the original tolerances.

The retained naive kernel only gains a final-word bounds check. The GPU dequantizer gains the same check and a ceiling grid extent. Host packing and scale arrays use ceiling counts. Group maxima and quantization loops stop at K. Existing aligned input shapes retain their data format and quantization behavior.

## 4. Validation decisions and current evidence

The existing FP64 reference and tolerance remain authoritative. Small tests check every output row, all reconstructed-weight FP16 bits, odd N, partial output tiles, uneven word partitions, K below eight, partial final groups, and zero weights. Main sweep validation preserves the default maximum of 64 sampled rows. The benchmark prints the exact checked row indices. Dequantization mismatches or output failures now stop execution before timing the failed kernel.

Local CPU build and CTest passed after the tail extension. GPU compilation, correctness, sanitizer, timing, and profiler results are pending at this checkpoint. Do not interpret the local pass as CUDA validation.

## 5. Operational attempts and failures

1. The first SSH attempt failed with `socket: Operation not permitted` in the local sandbox. An approved network escalation established the requested gateway-to-c58 connection. This was an access restriction, not a cluster failure.
2. Creating the work branch required escalation because `.git` is read-only in the sandbox. The branch was created without changing the baseline commit.
3. `/etc/sysconfig/prun` does not exist on c58. An exploratory command with `set -e` stopped while reading it. The installed launcher has a default configuration and subsequently launched one hostname process successfully through Open MPI. No launcher policy change was needed.
4. The first source archive used an exclusion pattern `build*`, which also excluded `scripts/build.sh`. The resulting remote build stopped with a missing-script error. Source packaging now names the required source directories explicitly. This was a packaging failure, not a kernel compile failure.

## 6. Continuation rules

Update this document with measured successes, regressions, and evidence paths as each phase completes. Keep hypotheses distinct from observations. Do not report unimplemented alternatives as failed experiments. Read `AGENTS.md` and `docs/experiment-inblock.md` for continuation commands and allocation checks. Stop after evaluating this optimization.
