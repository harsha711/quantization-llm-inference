# The project in one sitting

**Summary.** We set out to make 4-bit-weight matrix multiplication fast on a GPU and to turn that into faster language-model inference. Our own kernels gave real but shape-specific gains; the fastest kernel we used is an existing open-source one (Marlin), in which we found and fixed a synchronization hazard. Inside a full model that kernel, plus graph replay and a rewritten decode attention step, made generation about 1.7–2.4 times faster than our own 16-bit baseline. That speedup is only accepted under a numerical check chosen after two earlier checks had failed. Measured against three production inference systems, our path is slower than all of them, including vLLM running the very same 4-bit weights. **The project did not beat the state of the art.** Profiling showed that both our path and vLLM spent their largest single kernel on a 16-bit output matrix; compressing it with our 4-bit format made vLLM itself about 23–24% faster, tied with the fastest system we measured. We then measured quality: that 4-bit output matrix raises perplexity by 8.6%, simple fixes did not get it under 5%, and an 8-bit version keeps about 70% of the speedup in vLLM at a negligible measured cost. The 4-bit compression of the rest of the model itself raises perplexity by 13.1%.

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

## 7. The reality check, and what it led to

We then asked whether this is competitive. vLLM, a production server, ran the same 4-bit weights, produced the same 64 tokens and was faster at every prompt length ([E027](EXPERIMENTS.md#e027--our-decode-path-versus-vllm-on-the-same-quantized-weights)). llama.cpp and ONNX Runtime, each with its own 4-bit format that also compresses the large output matrix we left in 16-bit, were faster still; those are whole-system comparisons ([E028](EXPERIMENTS.md#e028--our-decode-path-versus-llamacpp-and-onnx-runtime-genai)).

| 63 decode steps | 128-token prompt | 2048-token prompt |
|---|---:|---:|
| Ours, Marlin, graph replay, direct attention | 326.8–327.0 ms | 375.5–376.0 ms |
| vLLM, identical weights | 238.1–238.8 ms | 248.1–248.9 ms |
| llama.cpp, own 4-bit format | 208.5–208.6 ms | 227.5 ms |
| ONNX Runtime, own 4-bit format | 185.2–185.4 ms | 191.6–192.2 ms |

Longer prompts made it worse. At 4,096–16,384 tokens our path stays 38.9% to 25.2% faster than our own 16-bit path ([E029](EXPERIMENTS.md#e029--our-decode-paths-at-contexts-beyond-2048)) but falls further behind all three systems; at 16,384 it is slower even than llama.cpp's and ONNX Runtime's 16-bit paths, so "our 4-bit path beats their 16-bit path" holds only up to 8,192 tokens.

**Locating the gap.** A kernel-level profile of our path and vLLM on identical weights put the difference on the GPU ([E030](EXPERIMENTS.md#e030--where-the-per-step-gap-to-vllm-lies-diagnostic)). vLLM merges weight matrices, making 112 Marlin calls per step to our 196, and fuses small operations, launching 276 other kernels to our 1264. The profile also showed one cost neither system had removed: the 16-bit output matrix (151,936 × 1,536) took 1.17 ms per step in both, the largest single kernel in either.

**Compressing the output matrix.** We quantized it with the project's own 4-bit format and ran it through Marlin, in our path and in unmodified vLLM, which already supports this ([E031](EXPERIMENTS.md#e031--4-bit-output-head-through-marlin-in-our-path-and-in-vllm)). Both passed the 5% bar in every measurement on three GPUs (one node shown; the others agree):

| 63 decode steps | 128-token prompt | 2048-token prompt |
|---|---:|---:|
| Ours, 16-bit output matrix | 326.8–326.9 ms | 375.4–375.6 ms |
| Ours, 4-bit output matrix | 271.4–271.5 ms | 320.2–320.3 ms |
| vLLM, stock | 237.4–238.6 ms | 248.6–249.2 ms |
| vLLM, our 4-bit output matrix | 180.7–180.8 ms | 190.9–191.1 ms |
| ONNX Runtime, own 4-bit format (same GPU, separate job) | 185.1–185.4 ms | 191.6–192.2 ms |

Our path gets 17.0% and 14.7% faster; vLLM gets 23.1–24.4% faster, level with ONNX Runtime within our 5% rule. That is a tie, from separate jobs with different quantizers, not a win, and our own path is still slower than stock vLLM.

Two caveats always go with this. **It changes what the model computes:** the output matrix was 16-bit in the original model, the scores move by a relative 0.12–0.17, and the 64 generated tokens stayed identical in three of four tests but only 47 of 64 in the fourth. **It makes the model measurably worse** (next paragraph). Our side also carries the after-the-fact gate disclosure of section 5.

**Measuring quality.** We then scored each model's next-word predictions on a standard Wikipedia text (WikiText-2, 307,050 tokens scored; lower perplexity is better) ([E032](EXPERIMENTS.md#e032--output-quality-of-the-quantized-model-with-and-without-the-4-bit-head)). The original 16-bit model scores 8.606. Compressing the 196 block matrices to 4 bits raises that to 9.735 (+13.1%), and the 4-bit output matrix to 10.572, a further 8.6%. So every 4-bit speedup in this account runs on a model 13.1% worse than the original (our 16-bit baselines used the same rounded weights, so the speed comparisons stay fair), and the output-matrix speedup costs another 8.6%. Simple fixes to the output matrix (smaller groups, better scales) got no closer than 5.8%, still above our 5% line ([E034](EXPERIMENTS.md#e034--can-a-better-head-quantizer-keep-the-4-bit-heads-speed-at-a-smaller-quality-cost)). An 8-bit output matrix costs only 0.04%; in vLLM it decodes 15.6–16.5% faster than stock, about 70% of the 4-bit gain, with every token unchanged ([E035](EXPERIMENTS.md#e035--how-much-of-the-head-speedup-survives-with-an-8-bit-head-vllm)). Our own kernel handles only 4-bit codes, so our path gains nothing from it, and that vLLM result has no numerical gate beyond token agreement. A same-job rerun confirmed the 4-bit tie with ONNX Runtime ([E033](EXPERIMENTS.md#e033--vllm-with-the-4-bit-head-same-job-comparison-with-onnx-runtime-and-batch-sizes-18)). One text, one metric, and our quantizer is the simplest kind (plain rounding, no calibration), so these numbers describe it, not 4-bit compression in general.

**A smarter quantizer, same format.** As a last test we replaced plain rounding with a calibrated method (GPTQ, published by others): it rounds one column of a matrix at a time and adjusts the not-yet-rounded columns to cancel the error, using statistics of real inputs from sample text. The stored format does not change, so the same kernels can run the result. For the output matrix the quality loss fell from 8.6% to about 1%, which passes our 5% bar. For the blocks it fell from 13.1% to 6.8%, about half, which still fails the bar. With both, the whole model is 7.7% worse than the original instead of 22.8% ([E036](EXPERIMENTS.md#e036--does-a-calibrated-quantizer-in-the-same-4-bit-format-remove-the-quality-cost)). Two cautions: we measured quality only and did not time these weights (the format is identical, so we expect the same speed, but did not measure it), and the sample text came from the same Wikipedia collection as the test text, which can flatter the result.

**Closing both cautions.** In a final pair of experiments we removed them. First, we took the sample text from a different source (web pages, the C4 collection), tried ten versions of the calibrated method, and chose one using a separate validation text before looking at the test score. The best version loses 5.6% in the blocks and 1.6% in the output matrix; the whole model is 7.3% worse than the original instead of 22.8%. The blocks still miss our 5% bar, narrowly. A more elaborate rounding method we wrote ourselves did not help. Second, we saved those weights and ran them through the same checks and timings as before. They pass the numerical check, and they run at the same speed as the plainly rounded weights: every one of 64 timings is within 0.3%, in our program and in vLLM. So the speedup from the 4-bit output matrix (23–24% in vLLM, 15–17% in ours) now comes with a 1.6% quality loss instead of 8.6% ([E037](EXPERIMENTS.md#e037--which-calibrated-quantizer-in-the-same-4-bit-format-costs-least-with-out-of-domain-calibration), [E038](EXPERIMENTS.md#e038--do-the-calibrated-tensors-pass-the-gate-and-run-at-the-speed-of-the-plain-rounded-ones)). This is not a new speedup: it is the old speed at about a third of the old quality cost, and our program is still slower than vLLM.

## 8. Conclusion and what is open

The project did not build a faster inference system than existing ones. It produced three measured kernel optimizations with their regressions; a fix to a synchronization hazard in a widely used kernel; a validated integration of that kernel into a full model; a measured account of why its saving was hidden and how graph replay and a decode-attention rewrite recover it; a profile locating the gap to vLLM; one concrete, measured improvement to vLLM's default setup, which costs 8.6% perplexity at 4 bits or a negligible 0.04% at 8 bits with about 70% of the speedup; and a quality measurement showing the 4-bit block compression itself costs 13.1%. Marlin and vLLM are upstream work; ours is the idea tested, the quantized weights, conversion, validation and measurement. Scope: one model (Qwen2.5-1.5B), one request at a time (which favours us against serving systems), 64 generated tokens, one GPU type (RTX 5060 Ti).

Still open: closing the blocks' remaining 5.6% loss (more calibration text, finer groups, the published tools); an 8-bit path in our own kernel; task-level quality benchmarks; merged weights and fused kernels in our path; prompt processing, which is 54–84% of a request at 4,096–16,384 tokens; batches above one; and our own kernels inside the model.

## Where to look

- [Project synthesis (report draft)](PROJECT_SYNTHESIS.md)
- [Current status](STATUS.md)
- [Experiment log, the source of all numbers](EXPERIMENTS.md)
- [Bottleneck assessment](BOTTLENECK_ASSESSMENT.md)
- [Numerical contract](CONTRACT.md)
- [Repository overview](../README.md)
