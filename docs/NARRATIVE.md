# The project in one sitting

**Summary.** We set out to make 4-bit-weight matrix multiplication fast on a GPU and to turn that into faster language-model inference. Our own kernels gave real but shape-specific gains; the fastest kernel we used is an existing open-source one (Marlin), in which we found and fixed a synchronization hazard. Inside a full model that kernel, plus graph replay and a rewritten decode attention step, made generation about 1.7–2.4 times faster than our own 16-bit baseline. That speedup is only accepted under a numerical check chosen after two earlier checks had failed. Measured against three production inference systems, our path is slower than all of them, including vLLM running the very same 4-bit weights. **The project did not beat the state of the art.** What it did produce is a validated account of where the 4-bit saving goes and what recovers it.

## 1. The goal

Large language models spend most of each generated token multiplying a few activation rows by large weight matrices. Storing those weights as 4-bit integers with a 16-bit scale per group of 128 values, while keeping activations and results in 16-bit floating point, is called **W4A16**. It cuts the bytes read per token by about four, so a kernel that unpacks the weights on the fly ("fused") should be faster when only a few rows are processed at once. Every kernel was checked against a 64-bit CPU reference under a fixed error bound.

## 2. Our own fused kernels

We wrote a simple fused kernel and improved it in three steps, each measured against the unchanged previous version in the same program ([E001](EXPERIMENTS.md#e001--in-block-k-splitting)–[E003](EXPERIMENTS.md#e003--reuse-reconstructed-weights-across-input-rows)).

| Change | Gain | Cost |
|---|---|---|
| Split each dot product across eight threads (E001) | 3.85× at one row | 1.3× slower at 128 rows |
| Compile separate versions for aligned and tail sizes (E002) | 1.31–1.33× at 32–128 rows | small |
| Reuse each unpacked weight for two or four rows (E003) | 1.55–1.75× at 32–128 rows | slower at one row and at odd row counts |

These are honest, shape-specific improvements with clear regressions, and they are scalar code that does not use the GPU's Tensor Cores. They were never placed inside the model.

## 3. A maintained comparator, and what tuning it taught us

To compare against a state-of-the-art kernel we integrated **Marlin**, the 4-bit Tensor Core kernel maintained in the vLLM project, feeding it our exact weights without requantizing ([E004](EXPERIMENTS.md#e004--maintained-marlin-integration-and-diagnostic-comparison)). When we tested a launch setting, NVIDIA's race detector reported a shared-memory hazard in Marlin itself, including in the official release ([E005](EXPERIMENTS.md#e005--specialization-aware-reservation-and-one-versus-two-block-residency)). We traced it and fixed it with one added barrier; the corrected kernel passes all checks ([E006](EXPERIMENTS.md#e006--bounded-inherited-marlin-racecheck-investigation)). We could not show that the original ever produced a wrong answer.

Attempts to tune the corrected kernel mostly failed our rule that a change must be at least 5% faster: a different memory reservation and grid (E005-R1), skipping unused loads (E007) and a smaller tile (E013, 7.6–8.7% slower). One launch setting was 13% faster when the same matrix is reused, but the gain disappeared once several different matrices were cycled through, as in a real model ([E005-R2](EXPERIMENTS.md#e005-r2--independent-rectangular-ac-confirmation), [E008](EXPERIMENTS.md#e008--weight-reuse-distance-with-preserved-ac)).

Against 16-bit cuBLAS on the same weights, corrected Marlin was 3.6–4.4 times faster at four rows, and 8.0–9.3 times faster than unpacking first and then calling cuBLAS ([E009](EXPERIMENTS.md#e009--compressed-execution-versus-cublas-under-cyclic-weights)). The lead narrows as rows grow, to 2–3% for one shape at 128 rows, without ever reversing ([E010](EXPERIMENTS.md#e010--four-matrix-batch-size-crossover)). Detailed profiling showed more instruction work at larger batches but could not pin down a cause precise enough to justify rewriting the kernel ([E011](EXPERIMENTS.md#e011--endpoint-profiles-with-preserved-cyclic-weight-access), [E012](EXPERIMENTS.md#e012--bounded-source-pc-attribution-of-corrected-marlin)).

## 4. The full model, first in 16-bit

We moved to a real model, Qwen2.5-1.5B, generating 64 tokens one at a time from a 128- or 2048-token prompt. Each generated token after the first is a **decode step**: one pass through the model for a single new token. We first tried to speed up the plain 16-bit model. Compiling it changed some outputs beyond our bound and was stopped ([E014](INFERENCE_OVERHEAD.md)). **Graph replay**, recording the GPU work of a decode step once and replaying it for each token so Python and launch overhead disappear, was exact but saved only 1–2% ([E015](INFERENCE_DIRECT_GRAPH.md), [E016](INFERENCE_FULL_DECODE_GRAPH.md)). The 16-bit model is limited by reading its weights, not by overhead.

## 5. Marlin inside the model, and the numerical gates

We put corrected Marlin under all 196 weight matrices of the model's blocks and compared it with 16-bit multiplication on the same reconstructed weights. Before any timing, each experiment applies a **gate**: a predeclared numerical test that must pass, or no speed is reported. The first gate compared every output score with the 16-bit path; 591 of 9.7 million values failed, although all 64 tokens matched ([E017](INFERENCE_MARLIN_BRIDGE.md)). A diagnosis showed every individual Marlin call was within bounds, and that 32- and 64-bit reference versions without Marlin differed from the 16-bit path just as much: the difference is ordinary 16-bit rounding propagating through 28 layers ([E018](EXPERIMENTS.md#e018--locate-the-e017-qr-logit-divergence-diagnostic)). A second gate still failed, once on an exact tie in the 16-bit path's own scores ([E019](EXPERIMENTS.md#e019--corrected-marlin-end-to-end-with-a-control-calibrated-model-gate)). A third gate, judging Marlin against the 64-bit reference instead, passed all ten cases ([E021](EXPERIMENTS.md#e021--acceptance-of-the-marlin-path-against-the-fp64-projection-control)). Because that gate was chosen after the first two failed, every speedup below carries that disclosure; the first two remain failed.

## 6. Why the kernel speedup was nearly invisible, and what recovered it

With Marlin, decoding was 29% faster at a 2048-token prompt but only about 4% at 128 tokens ([E020](EXPERIMENTS.md#e020--descriptive-primary-timing-of-the-numerically-unaccepted-marlin-path)). Timing the 196 calls showed why: Marlin's work took 2.95 ms per step, 99% of it host-side Python and launch time, so its device saving was hidden ([E022](EXPERIMENTS.md#e022--decode-step-projection-time-breakdown-diagnostic)). Graph replay, useless for 16-bit, now made Marlin decoding 45% faster than 16-bit at 128 tokens and 30% at 2048 ([E023](EXPERIMENTS.md#e023--cuda-graph-replay-of-the-marlin-decode-step)), and this held across five prompt lengths, two texts, reversed run orders and a second GPU ([E024](EXPERIMENTS.md#e024--robustness-of-the-e020e023-decode-gains)). A profile then showed that attention over the cached prompt grew with length and dominated long prompts ([E025](EXPERIMENTS.md#e025--decode-step-device-time-by-operation-diagnostic)). Computing single-token attention directly made decode nearly flat in prompt length ([E026](EXPERIMENTS.md#e026--single-query-grouped-decode-attention-under-graph-replay)):

| 63 decode steps | 128-token prompt | 2048-token prompt |
|---|---:|---:|
| 16-bit, graph replay, library attention | 614–615 ms | 900 ms |
| 16-bit, graph replay, direct attention | 596–597 ms | 650 ms |
| Marlin, graph replay, direct attention | 325–326 ms | 374 ms |

Against our 16-bit path with the same attention, that is 1.8× at 128 tokens and 1.7× at 2048; against our original 16-bit path, 1.9× and 2.4× (ratios of the medians above). The attention change helps 16-bit too, so it is not a 4-bit result.

## 7. The reality check

We then asked whether this is competitive. We ran vLLM, a production server, on the same 4-bit weights; it produced the same 64 tokens and was faster at every prompt length ([E027](EXPERIMENTS.md#e027--our-decode-path-versus-vllm-on-the-same-quantized-weights)). llama.cpp and ONNX Runtime, each with its own 4-bit format, were faster still ([E028](EXPERIMENTS.md#e028--our-decode-path-versus-llamacpp-and-onnx-runtime-genai)). Their weights differ from ours (they also compress the large output matrix we left in 16-bit), so those two are whole-system comparisons, and nobody has measured output quality for any of the 4-bit models.

| 63 decode steps | 128-token prompt | 2048-token prompt |
|---|---:|---:|
| Ours, Marlin, graph replay, direct attention | 326.8–327.0 ms | 375.5–376.0 ms |
| vLLM, identical weights | 238.1–238.8 ms | 248.1–248.9 ms |
| llama.cpp, own 4-bit format | 208.5–208.6 ms | 227.5 ms |
| ONNX Runtime, own 4-bit format | 185.2–185.4 ms | 191.6–192.2 ms |

Our 4-bit path does decode faster than llama.cpp's and ONNX Runtime's 16-bit paths (vLLM's 16-bit path was not run), but our own 16-bit path is slower than theirs. On identical weights vLLM spends 3.8–3.9 ms per step to our 5.2–6.0 ms. Where that 1.4–2.0 ms goes has not been measured.

## 8. Conclusion and what is open

The project did not produce a faster inference system than existing ones. It did produce: three measured kernel optimizations with their regressions; a fix to a synchronization hazard in a widely used kernel; a numerically validated integration of that kernel into a full model; and a measured explanation of why a 4-bit kernel's saving is hidden in a plain PyTorch model and how much graph replay and a decode-attention rewrite recover. The scope is one model, one request at a time (which favours us against serving systems), 64 generated tokens and one GPU type.

Still open: runs at prompts of 4,096–16,384 tokens are in progress with no result yet (E029 and long-context extensions of the comparisons); a profile locating the gap to vLLM is queued (E030); our own kernels were never run inside the model; compressing the output matrix, batches larger than one, and any quality evaluation are untested.

## Where to look

- [Project synthesis (report draft)](PROJECT_SYNTHESIS.md)
- [Current status](STATUS.md)
- [Experiment log, the source of all numbers](EXPERIMENTS.md)
- [Bottleneck assessment](BOTTLENECK_ASSESSMENT.md)
- [Numerical contract](CONTRACT.md)
- [Repository overview](../README.md)
