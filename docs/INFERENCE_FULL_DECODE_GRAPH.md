# E016: one graph for the complete fixed-length decode phase

**Broader capture preserves the checked outputs but does not improve warmed latency meaningfully.** Neither context reaches the predeclared 5% criterion. All 160 fresh primary samples are retained, and held-out timing is omitted by design. Existing defaults remain unchanged. This completes the bounded follow-up to E015, without establishing global optimality.

## 1. Capture boundary and validation

E016 expands CUDA graph capture while preserving the model forward and greedy selection. The fresh control G uses E015's per-token forward-plus-argmax graph. Candidate H captures the remaining 63 decode steps in one Python-unrolled graph, including device position updates, causal-mask construction and staging, logits/output copies, and token feedback. Prefill generates the first token eagerly. Tokenization, input transfer, cache reset, prefill, first-token CPU visibility, and final output transfer/detokenization remain outside capture and inside request timing.

Both variants use pinned `Qwen/Qwen2.5-1.5B` revision `8faed761d45a263340a0528343f099c05c9a4323`, FP16 weights/cache, efficient SDPA, batch 1, contexts 128/2048, and 64 fixed greedy outputs. The accepted c52 validation uses an RTX 5060 Ti 16 GB, driver 615.71.09, PyTorch 2.14.0+cu130, and Transformers 4.57.1 with one `prun` rank and 16 host threads. Both variants always retain full logits and device position/feedback histories in GPU buffers, including during timing. This shared observability cost makes fresh G the applicable denominator.

H snapshots the actual post-prefill cache and persistent buffers. Three side-stream warmups restore that snapshot before each execution. After warmup and capture, H defensively restores and verifies all saved bits, zero decode-cache tail, occupied lengths, and stable storage before genuine replay. Python state changes are not assumed to replay. Graph construction occurs during the first request and is absent from warmed measurements. Model loading and persistent-buffer construction precede request timing. Reported `graph_setup_s` includes snapshot/restoration checks and warmups. Inherited `static_setup_s` covers E014 cache/setup and excludes E016's additional buffer allocations.

All four full validation cells pass primary/held-out/primary requests after cache resets. Each request checks 9,723,904 finite logits, exact 64-token agreement, every recorded position and feedback token, stable storage, and complete per-layer cache hashes. The unchanged gates require `abs(delta) <= 0.03 + 0.01*abs(reference)` and relative L2 at most 0.003. Observed logits are bitwise identical. G also passes preserved E015 full-logit comparisons and uncached endpoint checks. Weights remain unchanged. Candidate/reference tensors and diagnostics precede assertions. Independent offline audits pass 18 tensor comparisons and provenance checks. Retained-kernel CUDA build, CTest, FP64 checks, memcheck and synccheck pass. These sanitizer checks do not cover the whole model.

Full-length traces show 63 G graph launches and one H launch at each context. G records 441 device-to-device memcpy events where H records 441 `memcpy32_post` kernels. Other kernel-name counts agree, but copy representation differs. These observations establish actual broader replay, not complete operational equivalence or a causal latency explanation.


## 2. Fresh matched latency

Each fresh-process cell has three full request warmups and 20 samples. Two series reverse context and variant order, independently verified by recorded boundary timestamps. All 160 requests pass exact output/state checks and unchanged-weight/compiler checks. The denominator is fresh same-series G. Decode is synchronized wall-clock time from after first-token input staging through all 63 cached steps and the final synchronization. Common mask/position preparation and GPU diagnostic copies remain included. TTFT begins before tokenization/H2D/cache reset and ends at CPU-visible token 1. Complete-request latency includes final token transfer and detokenization. Post-timer diagnostic audits are excluded from request latency.

Gain below is `100*(1-H/G)`, with positive values indicating lower H latency. A context requires at least 5% in BOTH series before the predeclared 160-request held-out campaign for both contexts is released.

| Context tokens | Series | G decode ms | H decode ms | H gain |
|---|---:|---:|---:|---:|
| 128 | 1 | 606.395 | 607.588 | -0.197% |
| 128 | 2 | 607.687 | 608.241 | -0.091% |
| 2048 | 1 | 882.669 | 883.848 | -0.134% |
| 2048 | 2 | 884.400 | 883.975 | +0.048% |

Short-context H medians are 0.554–1.193 ms higher. Long-context differences range from 1.179 ms higher to 0.425 ms lower. These results provide no meaningful additional decode reduction under this policy. No confidence interval, stable-tail claim or statistical-significance claim is made.

The next table reports TTFT and complete-request medians in milliseconds, lower being better. Each entry lists G/H.

| Context tokens | Series | TTFT ms G/H | Complete request ms G/H |
|---|---:|---:|---:|
| 128 | 1 | 14.260 / 14.281 | 620.724 / 621.955 |
| 128 | 2 | 14.336 / 14.342 | 622.098 / 622.653 |
| 2048 | 1 | 184.160 / 184.531 | 1066.886 / 1068.422 |
| 2048 | 2 | 185.377 / 185.075 | 1069.854 / 1069.209 |

Complete-request changes are +0.198%/+0.089% at 128 and +0.144%/−0.060% at 2048. TTFT changes range from −0.163% to +0.201%. There is no accepted request-latency improvement. Clocks are unlocked. Raw snapshots span a cold P8 boundary and later P0 boundaries, with temperature 35–55°C and one after-cell memory-clock difference. They do not measure continuous clock behavior or explain these small differences. Every sample and range/p90 record is retained.

## 3. Construction, first use and memory

Graph setup below includes the instrumented warmup/capture/restoration work. First-request E2E includes first-use effects and construction after the model is loaded. These are separate from warmed timing, not full-process startup or isolated pure CUDA capture timings.

| Context tokens | G graph setup | H graph setup | G first request | H first request |
|---|---:|---:|---:|---:|
| 128 | 41.7–43.5 ms | 2.465–2.470 s | 7.660–7.705 s | 10.107–10.125 s |
| 2048 | 55.2–55.3 ms | 3.282–3.306 s | 8.100–8.114 s | 11.349–11.352 s |

Warmed allocator peak reserved storage is 3274→3294 MiB at 128 (+20 MiB) and 3458 MiB for both at 2048. Warmed allocated peaks differ by less than 0.3 MiB. During the first request, H increases peak allocated storage by 31.93 MiB/54.80 MiB at 128/2048. Cold reserved storage rises 40 MiB at 128 and is unchanged at 2048. Memory records include the common post-timer state checks. They are PyTorch allocator statistics, not total peak device usage or host graph-metadata storage. The 75% allocator-reservation safety bound is never reached.

Model-load records are approximately 0.57–0.58 s, and inherited static setup is approximately 61–63 ms. File/parameter hashing, imports and E016's extra persistent buffer initialization are not all represented by those two timers. No complete initialization or amortization claim is made.

## 4. Interpretation and preserved evidence

1. One graph successfully covers all 63 decode steps with changing positions and token feedback. Numerical, full-cache and repeated-reset checks pass on the exact timed GPU path.
2. Replacing 63 launches and the host loop with one launch does not meet the 5% target. The largest observed H gain is 0.048%, while three comparisons have higher H medians. Larger capture adds seconds of construction cost and some allocator overhead.
3. These observations bound this fixed-output capture choice. They do not identify a unique remaining compute/bandwidth bottleneck or establish that inference is globally optimal. The data do not isolate whether host work was already hidden, whether graph/copy costs offset a benefit, or another cause. No fusion, compilation, attention change, quantization, custom kernel or next experiment follows.

The [experiment log](EXPERIMENTS.md#e016--complete-fixed-length-decode-cuda-graph) preserves every attempt. c9 full validation passes before its retrieval allocation ends. c58's prerequisite passes, but another allocation takes that node before full validation. Unstarted node-specific jobs are cancelled, with all c9/c58 records retained. Scheduler-selected c52 receives fresh source/runtime/topology/storage/small checks, complete numerical validation and the entire primary campaign in one retained allocation. No c9/c58 timing is spliced into the denominator. The corrected evidence-preservation ordering receives its own inspected prerequisite. Bootstrap source-glob, purged-dependency, approval-review-timeout and retrieval-manifest failures remain explicit. The initial self-including primary hash manifest is retained separately, then a corrected manifest independently verifies all 85 sealed primary files without changing measurements.

[Final c52 evidence](../results/e016_20261109/final-arc/), [raw/derived primary decision](../results/e016_20261109/final-arc/primary-analysis.json), [summary](../results/e016_20261109/final-arc/primary-summary.csv), [independent recomputation](../results/e016_20261109/final-arc/independent-primary-calculations.json), [source/build archive](../results/e016_20261109/final-arc/executed-source-build.tar.gz) and [tooling](../scripts/e016/README.md) retain exact versions, operands, logs, traces, tensors, clocks and costs. Independent complete-primary analysis and eleven damaged-data scenarios pass. Historical E014/E015 inputs and corrected Marlin A remain unchanged. GPU source/runtime identity is verified, and no whole-model sanitizer campaign is claimed. Shared final root: `/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109`. Job 332017 completes after primary retrieval, with no held-out timing. All node/job IDs are historical. Stop after this comparison.
