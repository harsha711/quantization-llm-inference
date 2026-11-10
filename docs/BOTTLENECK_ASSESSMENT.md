# Bottleneck assessment

## Full-model decode after the attention change, 2026-11-10 (current)

E025 measured the split below on an eager Marlin step; E026 then removed most of the attention and copy rows.

| Device time per decode step (E025, eager Marlin, c9) | context128 | context2048 |
|---|---:|---:|
| Marlin kernels, 196 projections | 2.44 ms | 2.49 ms |
| FP16 output head GEMV, unquantized | 1.17 ms | 1.17 ms |
| Stock attention kernel | 0.47 ms | 4.00 ms |
| Copies incl. key/value head expansion | 0.37 ms | 1.42 ms |
| Everything else | 1.08 ms | 1.10 ms |
| Total | 5.53 ms | 10.18 ms |

After E026 the replayed Marlin step is 5.16 ms at context128 and 5.94 ms at context2048 (E026 medians / 63), so attention plus copies now cost roughly 0.5 ms and 1.2 ms (calculation: step minus the other E025 rows; not re-profiled).

**Current bottleneck: the Marlin projections (about 2.4 ms, near their weight traffic of 0.68 GB per step) and the unquantized output head (1.2 ms, 467 MB per step), then about 1.1 ms of small elementwise kernels.** For a whole long-context request, the eager prefill (about 196 ms of 570 ms at context2048) is now a third of the time.

Candidates, none run: quantizing the output head (a quantization-scope change that alters model outputs and needs its own acceptance and a quality argument); the 56 small k/v projections, which use Marlin's fallback dispatch and whose share of the 2.4 ms is unmeasured; prefill.

## Full-model decode after graph replay, 2026-11-10 (E023 snapshot)

E023 tested the hypothesis below and it holds at short context only.

| Decode step, ms (E023 median / 63) | FP16 eager | FP16 graph | Marlin eager | Marlin graph |
|---|---:|---:|---:|---:|
| context128 | 9.94 | 9.75 | 9.46 | 5.43 |
| context2048 | 14.44 | 14.26 | 10.26 | 10.04 |

- Context128: removing host time exposes the Marlin device saving; the step falls 43%. FP16 gains under 2% from the same technique because it was already device-bound.
- Context2048: Marlin gains only about 2% from graph replay, so it is device-bound there. Both paths cost about 4.6 ms more per step than at context128.

**Current bottleneck: at long context, device work outside the 196 projections** (attention over the 2112-entry static cache, norms, masking, output head are the candidates). This is an inference from the step-time differences; no profile attributes it. At short context the graph-replayed Marlin step is 5.4 ms, of which E022 bounds the projections at no more than 2.5 ms; the rest is likewise unattributed.

Cheapest discriminating measurement (not run): one profiler trace of a replayed decode step at each context, summing kernel time by category.

## Full-model decode before graph replay, 2026-11-10 (E022 snapshot)

Evidence checkpoint: E020–E022 on c7, RTX 5060 Ti 16 GB, Qwen2.5-1.5B, batch 1, 196 block projections, corrected E006 A unchanged. The kernel-level sections further down are historical and unchanged.

**Current bottleneck: with Marlin, a decode step is limited by host-side Python/launch time, not by the GPU.** With reconstructed FP16 it is limited by device weight traffic.

| Per decode step | FP16 (R) | Marlin (Q) | Source |
|---|---:|---:|---|
| 196 projections, host time until last call returns | 1.93 ms | 2.89–2.93 ms | E022, measured |
| 196 projections, total after synchronize | 6.96 ms | 2.93–2.97 ms | E022, measured |
| Host share of projection total | 28% | 99% | calculation |
| Raw Marlin calls without the module wrapper, total | — | 2.50 ms (host 1.70 ms) | E022, measured |
| Selected weight bytes read per step | 2.62 GB | 0.68 GB | E022 inventory |
| Whole decode step, context128 | 9.93 ms | 9.57 ms | E020 median / 63 |
| Whole decode step, context2048 | 14.42 ms | 10.25 ms | E020 median / 63 |

Observations:
- FP16 projections wait on the device: 2.62 GB in 6.96 ms is about 376 GB/s of modeled weight traffic (not a measured DRAM counter).
- Marlin projections finish essentially when Python returns. The module wrapper (reshape, output allocation, bias add) costs about 0.45 ms of the 2.95 ms; ctypes plus launch is the rest.
- Marlin saves about 4.0 ms of projection time per step, but the context128 step improves by only 0.36 ms.

Interpretation (hypothesis, consistent with E015/E016/E020 but not directly measured): the eager decode step has roughly 9.5 ms of host work. At context128 FP16's device work finishes about when the host does, so removing 4 ms of device work shows little. At context2048 FP16's device work exceeds the host time and Marlin's does not, giving the measured 29%. This would also account for E020's "unexplained" long-context FP16 penalty and for graph replay gaining only 1–2% on FP16 (E015/E016): FP16 was device-bound once the host was removed.

What this does not establish: the host/device split of the non-projection work (attention, norms, cache, sampling) with Marlin; whether Marlin's external ctypes launches can be captured in a CUDA graph; any gain from doing so.

Supported next hypothesis (one, not run): replaying the Marlin decode step as a CUDA graph removes the host limit and exposes the device saving at both contexts. An upper bound on its scope is the roughly 9.5 ms host step versus Marlin's unknown device-only step; a cheap discriminating check is whether capture of one Marlin projection replays bitwise. It needs its own predeclared validation (E021's gate) and fresh timing.

# Corrected-kernel bottleneck assessment (historical, kernel level)

E008 follow-up: The unchanged-kernel complete-cycle experiment reproduces both single-matrix controls on fresh inputs, but neither two nor four matrices passes the5% meaningful-gain gate in both series. The transpose with four matrices loses 0.757%/0.431%. This bounds the earlier reused-input opportunity. Keep corrected A default, retain C as reuse-dependent experimental configuration, and do not infer a general decoding gain or a new partition algorithm. See the E008 experiment record for full timings, outliers and limits. The original assessment below remains historical.

**Follow-up:** [E005-R2](EXPERIMENTS.md#e005-r2--independent-rectangular-ac-confirmation) has completed the bounded confirmation on c10 with the exact corrected E006 library. Both rectangles pass the5% no-flush gate in both reversed series. K=4096/N=11008 improves13.33%/13.28% but regresses 5.48%/6.06% with store flushing. The conditional matched profile pair shows more eligible warps/issue activity with 5.060% more warp instructions,1.132% more L2 bytes and high replay DRAM throughput. This supports a restricted existing-launch opportunity with added coordination and cache-policy costs. It does not identify a concrete improved partitioner or justify dispatch. Corrected A remains default. The assessment below is the saved-data snapshot that motivated this completed test. Its statements about missing rectangular evidence and proposed measurements describe that earlier checkpoint.

2026-11-06. Evidence checkpoint: `ab54819`. **Keep E006 corrected A as the baseline. E005-R1 and E007 remain rejected performance experiments.** This assessment uses saved measurements only and implements no E008 or dispatch policy.

## 1. What currently limits the kernel?

The square M8 profiles support limited ready work and a tradeoff between concurrency and partition overhead. They do not establish a unique memory, compute or dequantization bottleneck. These are corrected-body profiles on c58's RTX 5060 Ti, K=N=4096, no profiler cache flush and clocks unlocked. There are no corrected rectangular profiles.

The matched [E005-R1 counters](../results/e005_r1_20261106/profile_summary.csv) show how each launch changes execution. CTA activity is averaged over SM active cycles, eligible warps over scheduler active cycles, and issue activity over scheduler active cycles. Higher activity alone does not imply lower latency.

| Observation | A: 36 blocks, 101376 B | B: 36 blocks, 49664 B | C: 72 blocks, 49664 B |
|---|---:|---:|---:|
| Active CTAs per SM active cycle | 1.00 | 1.00 | 1.76 |
| Active warps, % of peak during active cycles | 16.62 | 16.63 | 28.95 |
| Eligible warps per scheduler active cycle | 0.427 | 0.426 | 0.646 |
| Issue-active cycles, % | 35.37 | 35.32 | 42.59 |
| Executed warp instructions | 1442222 | 1442222 | 1575593 |
| L2 traffic, bytes | 11231840 | 11261344 | 11519552 |

A's shared reservation permits only one CTA per SM. B removes that capacity constraint but retains the 36-block grid, produces essentially unchanged activity, and gives no consistent primary latency gain. C increases activity, but executes 9.248% more warp instructions and transfers 2.562% more L2 bytes. Its barrier and scoreboard metrics also rise. Application M8 no-flush latency improves only 2.653% and 2.765% in the two series, below the 5% criterion. Thus capacity alone is insufficient, while extra parallelism carries measurable overhead.

Square replay DRAM throughput is about 36% of sustained peak in A/C, and Tensor-pipe activity is about 45%/47% of elapsed cycles. Neither establishes saturation. L2 throughput was not collected. Stall percentages are not disjoint runtime fractions, and Tensor activity is not peak FLOP utilization. Low eligible-warp and issue activity are consistent with imperfect latency hiding, without isolating its cause. These metric distinctions follow the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Replay counters remain separate from application timing.

[E007](../results/e007_20261106/profile_summary.csv) provides little support for another terminal-fetch tweak. Participating LDS instructions fall 1.931%, issued LDS/LDSM counts stay equal, total warp instructions rise 1.757%, and registers 126→124 still allocate 128. Application M8 medians tie. These are within-experiment comparisons, not comparisons between E007's seven-pass and R1's thirteen-pass replay durations.

## 2. Which candidate has enough scope?

**Rectangular K-stripe distribution and reduction balance are the strongest unconfirmed lead.** In the single seed-2 [held-out series](../results/e005_r1_20261106/comparisons.csv), M4 C improves no-flush latency by 9.081% at K4096/N11008, from 28.192 to 25.632 us, versus 1.834% at K11008/N4096. Store-flush changes are +3.054% and +0.047%, respectively. Equal weight volume with different gains motivates a geometry-dependent explanation, but does not prove one.

Unlike a terminal fetch, stripe distribution affects work throughout the K traversal. The existing [finite partition model](../scripts/e005/safety.py) predicts that, for K4096/N11008, A assigns 72 complete output tiles before its remainder stripes, while C assigns none in that first phase. The number of 128-column output tiles requiring split/global reduction increases from 14 to 68. For the transpose, both configurations have 32 such tiles. Concurrency, work balance and reduction costs therefore change together. C is a retained diagnostic probe, not a proposed default. No specific new partition algorithm is justified yet.

For an isolated phase occupying runtime fraction `f`, speeding that phase up by `s` reduces total latency by `f*(1-1/s)`. A 5% total gain requires at least 10% of runtime for a twofold phase speedup. No such phase fraction has been measured here. Instruction-count percentages cannot supply it.

## 3. Cheapest discriminating measurement

First confirm repeatability without profiling or code changes. Use the preserved corrected E006 library and A/C at M4, seed 2, both rectangular orientations and both cache policies. Collect two independent balanced timing series with reversed A/C order, 10 warmups and 100 samples per row. This is 16 rows/1600 samples. Preserve operand hashes, full numerical/workspace checks and the existing methodology. Verify current binary/environment/allocation identity and prerequisites before any future ARC run. This confirmation is proposed, not executed or authorized by this assessment.

If the apparent gain disappears or depends on order, stop without a launch policy. If a meaningful gain reproduces, collect one matched A/C profile pair for K4096/N11008/M4. Query supported metrics first. Measure CTA/eligible/issue/Tensor activity, executed instructions, L2/DRAM bytes and throughput, and barrier/scoreboard stalls using the retained replay method. Improved ready work and issue activity would support the parallelism explanation. High memory utilization and relevant waits would favor a memory-service explanation. Increased reduction/instruction costs could explain cancellation. These remain competing explanations. If aggregate counters cannot separate them, seek source-PC evidence at the lock/reduction phase before editing.

The next decision is whether to authorize that bounded confirmation. E008 implementation, broader tuning and shape-dependent dispatch remain deferred.
