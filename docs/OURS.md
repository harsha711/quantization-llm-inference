# Bandwidth-roofline series (branch `samarth/bandwidth-roofline`)

Our own fused W4A16 kernels, each scored by how close it gets to the memory-bandwidth floor at M=1. This series is separate from the E001–E038 log in [EXPERIMENTS.md](EXPERIMENTS.md), and nothing in it changes those records.

## Question

At M=1 (decode), a W4A16 multiply is a read stream of its packed weights. Its best possible time is therefore

```
floor_us = logical_bytes / ceiling_GBps        (logical_bytes: README "Metrics")
pct_floor = floor_us / measured_median_us * 100
```

- **ceiling_GBps** is measured by `w4a16_bwprobe` (a 128-bit streaming read far larger than L2). The 448 GB/s datasheet figure for the RTX 5060 Ti is quoted only for context.
- **Yardstick:** cuBLAS FP16 on the same shape.
- **Reference:** Marlin. We expect to stay slower than it, and the goal is to explain the gap.
- Harsha's naive kernel at M=1, K=N=4096 on c7 takes about 204 µs, against a floor of roughly 19 µs at spec (`results/baseline_c7.csv`). That's about 10% of the floor.

## Method

- **Hardware:** ARC node c7, RTX 5060 Ti (sm_120), the same GPU as Harsha's baselines.
- **Correctness first:** `KERNELS=vN bash scripts/ours/check.sh` (FP64 reference tolerance, tail shapes, compute-sanitizer). No timing for a kernel that fails.
- **Timing:** `bash scripts/ours/sweep.sh` (cold L2, 10 warmup / 200 iters).
  - M ∈ {1, 2, 4, 8, 16}.
  - Shapes (K×N): 4096×4096, plus the Qwen2.5-1.5B layers 1536×1536, 1536×8960 and 8960×1536.
- **Figures:** `uv run scripts/ours/plot.py results/ours_<ts>/sweep.csv`.
- **One idea per version.** Write the expected effect down before measuring.

## Log

### Ceiling

- `w4a16_bwprobe` on c7 (2026-11-17, job 342380): **427.4 GB/s**, which is 95% of the 448 GB/s spec. 256 MiB and 1 GiB buffers and unroll depths 1/4/8 all agree to within 1%.

### Timing caveat: the store flush inflates fast kernels (2026-11-17)

The original cold-L2 flush (`launch_l2_flush`) writes 2×L2 = 64 MiB right before the start event. That leaves L2 full of dirty lines, and their write-back competes with the timed kernel's weight reads. A read-only flush (`--flush-read`, `src/cuda/flush_read.cu`) evicts the inputs without leaving anything dirty. M=1, 300 iterations, c7:

| median µs | store flush | read flush | ncu `--cache-control all` (base clocks) | no flush (warm) |
|---|---:|---:|---:|---:|
| inblock_specialized 4096² | 51.5 | 33.1 | 35.5 | 28.2 |
| inblock_specialized 8960×1536 | 69.3 | 57.0 | 60.1 | 32.3 |
| cuBLAS FP16 4096² | 94.2 | 84.0 | n/a | 83.3 |
| naive 4096² | 205.2 | 194.2 | 206.1 | 107.8 |

- The read flush agrees with Nsight's cold-cache duration, which runs at locked base clocks and so is expected to be slightly slower.
- It stays clearly slower than warm, so it does evict.
- The store flush adds up to about 18 µs, which lands hardest on exactly the fast kernels this series is about.

**From here on the sweep defaults to `FLUSH=read`.** The first sweep below used the store flush and is kept for the record only.

This doesn't invalidate Harsha's E001–E003 comparisons, which used one flush policy throughout, but absolute times for fast kernels under the store flush are pessimistic. Worth flagging to him.

### Profile: why the baseline kernels are slow (`results/ours_20261117_010442/ncu_baseline.txt`)

Nsight Compute, M=1, cold caches:

| kernel, shape | blocks launched (36 SMs) | achieved occupancy | DRAM throughput |
|---|---:|---:|---:|
| naive 4096² | 16 | 16.5% | 9.5% |
| naive 8960×1536 | 6 | 16.6% | 3.5% |
| inblock_specialized 4096² | 128 | 55.9% | 55.4% |
| inblock_specialized 8960×1536 | 48 | 22.3% | 26.9% |

**The naive mapping leaves most SMs empty.** One thread per output column with 256-thread blocks gives N/256 blocks: 16 or 6 blocks for 36 SMs, so 20–30 SMs do nothing. inblock's 8-way K split multiplies the thread count by 8. That's still under one full wave on the tall shape (48 blocks, 22% occupancy).

**Implication for v1:** the K split has to create enough independent work to fill all 36 SMs even when N is small. That means splitting K across *blocks* (or more slices per block), not just across 8 threads.

### Baseline sweep, read flush (current): `results/ours_20261117_011239/`

All points pass the FP64 check. Ceiling 427.4 GB/s. M=1, % of each kernel's own floor (µs):

| K×N | 4-bit floor | naive | inblock_specialized | cuBLAS FP16 | unfused |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 20.3 | 10.4% (194) | 61.2% (33) | 93.5% (84) | 69.7% (142) |
| 1536×8960 | 16.7 | 22.7% (73) | 61.7% (27) | 89.9% (72) | 82.4% (98) |
| 8960×1536 | 16.7 | 3.9% (424) | 29.9% (56) | 89.5% (72) | 84.2% (96) |
| 1536×1536 | 2.9 | 3.8% (75) | 22.6% (13) | 77.1% (14) | 75.4% (18) |

- **cuBLAS FP16 runs at about 90–94% of its own floor on the large shapes.** At M=1 it's essentially a perfect FP16 weight stream, so the most 4-bit can buy over it is about 4.1× (84 / 20.3 µs at 4096²).
- **inblock_specialized gets 2.5× of that at 4096², but only 1.3× on the tall 8960×1536 shape**, where it's starved of blocks (see the profile above).
- **The fused kernels lose to cuBLAS from M=4 on all three large shapes.** Each row re-reads the weights, so this is a traffic problem, not a compute-bound crossover.

### Baseline sweep, store flush (superseded): `results/ours_20261117_010442/`

Harness validation: `naive` at M=1, 4096² measures 205.2 µs here, against 204.4 µs in Harsha's `results/baseline_c7.csv`. All kernels pass the FP64 check at every point.

M=1, % of each kernel's own bandwidth floor (µs in brackets):

| K×N | floor (4-bit) | naive | inblock_specialized | cuBLAS FP16 | unfused |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 20.3 | 9.9% (205) | 39.9% (51) | 83.4% (94) | 55.3% (179) |
| 1536×1536 | 2.9 | 3.7% (78) | 17.1% (17) | 60.0% (18) | 48.0% (29) |
| 1536×8960 | 16.7 | 18.8% (88) | 43.2% (39) | 82.8% (78) | 60.0% (135) |
| 8960×1536 | 16.7 | 3.9% (426) | 24.0% (69) | 81.0% (80) | 59.1% (137) |

Observations (measured, not yet explained by a profile):
- **cuBLAS FP16 is already at 81–83% of its own floor on the three large shapes.** It reads 4× the bytes, so a 4-bit kernel at the same efficiency would be about 4× faster. The best existing fused kernel reaches 40% of the floor at most: 1.85× over cuBLAS at 4096².
- **naive is worst on the tall shape 8960×1536 (3.9%).** Only 1536 output columns means 48 warps for 36 SMs, each looping over K=8960 one word at a time. This is the case v1's K-split targets.
- **1536×1536 has a floor of only 2.9 µs.** Every kernel there is dominated by fixed costs (launch, ramp-up), so % of floor can't be compared fairly at this size.
- **The fused kernels fall behind cuBLAS by M=4–8.** That isn't a compute-bound crossover: grid.y = M means every row re-reads the whole weight matrix, so traffic grows with M. This is the "reuse across rows" stretch item.
- **unfused is slower than cuBLAS FP16 at every point**, as expected, because it writes and re-reads a full FP16 copy.

### v0: own naive kernel

- **Idea:** one thread per output, dequant in registers. This is the learning baseline.
- **Expected:** within noise of `naive`, about 10% of floor at M=1.
- **Correctness:** `check.sh` passes (all tail shapes, FP64 tolerance, memcheck/racecheck/synccheck).
- **First attempt, `results/ours_20261117_011450/`:** 1.23–1.33× slower than `naive` at every point, for example 248 vs 194 µs at M=1, 4096².
  - Cause: the inner loop over a word's 8 rows used a run-time bound `min(8, K - k0)`, so the compiler couldn't unroll it.
  - `naive` uses a constant 8, which gives 8 independent `A` loads, constant shift amounts and no loop branch.
  - The kernel is limited by memory latency (few warps per SM), not bandwidth, so every instruction a warp issues in sequence adds directly to the time.
- **Fix:** full words use a constant 8-iteration unrolled loop, and the partial final word gets a separate tail loop.
- **Result, `results/ours_20261117_011616/`:** within 0–3% of `naive` everywhere. At M=1: 10.4% of floor (194 µs) at 4096², 22.7% at 1536×8960, 3.9% at 8960×1536.
- **Takeaway for v1+:** keep inner trip counts compile-time constant, and handle tails outside the hot loop.

### v1: split K across warps and blocks (`src/cuda/ours_v1.cu`)

- **Idea:** fill all 36 SMs regardless of N.
  - Block = 32 lanes (adjacent columns) × 8 warps (K sub-ranges).
  - grid = (N/32 tiles, M, S splits).
  - The S partial sums of a tile are combined by whichever block finishes last, in a fixed order (deterministic, one kernel, no memset).
- **Expected (from the profile):** a large gain on 8960×1536, where naive and inblock launch 6 and 48 blocks for 36 SMs.
- **Correctness:** `check.sh` passes for all variants (FP64 tolerance, tails, memcheck/racecheck/synccheck).

**What it took to get there.** Each step is one change, measured at M=1 with the read flush:

| step | 4096² | 1536×8960 | 8960×1536 |
|---|---:|---:|---:|
| first v1: interleaved words per warp, run-time tail check, split rule "exceed one wave" (`v1_runtime_tail`, `results/ours_20261117_011925`) | 43.0 µs | 32.8 | 36.9 |
| + tail check compiled out when K % 8 == 0 | 34.8 | 28.7 | 30.7 |
| + contiguous word run per warp instead of every 8th word | 34.8 | 26.8 | 30.7 |
| + split rule "balance blocks per SM", gain ≥ 10% (final `v1`, `results/ours_20261117_012711`) | **32.8** | **27.8** | **27.3** |

- **The tail check is the big one (15–20%).** It's the same lesson as v0: in a latency-bound loop, every instruction per word costs time.
- **Contiguous vs interleaved:** within one ~2 µs timer step. Kept for parity with inblock.
- **Split rule.** Sweeping S with `W4A16_V1_SPLITS` showed that time follows `ceil(tiles·M·S / 36) / S`, i.e. the busiest SM's block count × work per block. Not "exceed one wave".
  - On 8960×1536, S=3 (144 blocks, exactly 4 per SM) took 27.3 µs; S=4/5 took 30.7; S=2 took 32.8; S=1 took 54.2.
  - Splitting only when the predicted gain is ≥ 10% stops over-splitting (without it, 1536×8960 got S=5: 30.7 vs 27.7 µs).

**Result at M=1** (mean µs; % of floor; speed-up over cuBLAS FP16):

| K×N | inblock_specialized | v1 | v1 vs cuBLAS |
|---|---:|---:|---:|
| 4096×4096 | 33.2 (61.1%) | 32.8 (61.8%), S=7 | 2.55× |
| 1536×8960 | 27.0 (61.6%) | 27.8 (60.0%), S=1 | 2.58× |
| 8960×1536 | 57.0 (29.2%) | **27.3 (61.1%)**, S=3 | 2.64× |
| 1536×1536 | 12.0 (23.9%) | 8.2 (34.8%), S=3 | 1.77× |

- **The hypothesis holds:** 2.1× over inblock on the block-starved tall shape.
- **The three large shapes now plateau at 60–62% of the floor.** ncu at M=1 (`results/ours_20261117_012514/ncu_v1.txt`, same S choices as final v1 for these two shapes): DRAM throughput 55%, achieved occupancy 89% (4096², S=7) and 60% (8960×1536, S=3).
- **The SMs are now busy but DRAM is still only about half used.** Each thread issues one 4-byte load per 8 weights, so too few bytes are in flight per warp. That's v2's target.

**Not v1's target, recorded as measured:**
- At M ≥ 2, v1 is 5–10% slower than inblock_specialized on most shapes, even with the same grid (S=1). Cause not profiled. Registers are 40 vs 34; occupancy is the same.
- Both lose to cuBLAS from M=4, because each row re-reads the weights (the stretch item).
- **Timer resolution:** CUDA event times on c7 come in about 2.05 µs steps. The sweep now records `mean_us` alongside the median, and differences of one step aren't treated as real.

### Node change: c7 → c8 (2026-11-18)

c7 was held by another user's 7-day job, so v2/v3 ran on **c8**, another node in the same `rtx5060ti16g` partition (RTX 5060 Ti, same clocks, driver 610.57.04, CUDA 13.4). The bandwidth probe on c8 gives the same **427.4 GB/s**. The full sweep below re-runs every baseline and v1 on c8 in the same job, so all comparisons in it are same-node; v1 reproduces its c7 numbers to within one timer step (32.8 / 27.9 / 27.1 / 8.2 µs vs 32.8 / 27.8 / 27.3 / 8.2).

### v2: 128-bit loads (`src/cuda/ours_v2.cu`)

- **Idea:** each lane loads one `uint4` (4 packed words = 16 B) per step instead of one 4 B word, so 4× the bytes are in flight per load instruction.
  - Layout consequence: the 4 neighbouring words are 4 neighbouring *columns*, so a lane computes 4 outputs (reusing the same 8 activations) and a block covers 128 columns — 4× fewer column tiles than v1.
  - Scales: one 8 B load per word; activations: one 16 B broadcast load per word.
  - Fast path needs N % 4 == 0 and K % 8 == 0 (alignment); other shapes fall back to v1. All model shapes qualify.
- **Expected:** v1's plateau (DRAM ~55% used, SMs busy) moves up, *if* the K split wins back the lost blocks.
- **Correctness:** `check.sh` passes, including new fast-path tails (N = 4, 36, 132, 260, 388; K = 8 … 1032), forced S ∈ {1, 2, 3, 7, 16}, and compute-sanitizer on the fast path with and without the cross-block reduction.

**Split sweep** (`results/ours_splits_v2_c8/`, M=1, S forced through `W4A16_V2_SPLITS`, mean µs):

| K×N | tiles | S=1 | best S | blocks at best | best mean µs |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 32 | 47.0 | 4–5 | 128–160 | 26.6 |
| 1536×8960 | 70 | 23.5 | 2 | 140 | 22.6 |
| 8960×1536 | 12 | 94.3 | 12 | 144 | 23.1 |
| 1536×1536 | 12 | 19.5 | 6–12 | 72–144 | 8.2 |

- **v1's balance rule doesn't transfer.** With 32 tiles it sees no imbalance at S = 1 and launches 32 blocks for 36 SMs (47 µs). Balance alone isn't enough; each SM needs several blocks in flight.
- **New rule: about 4 blocks per SM,** `S = round(4·SMs / (tiles·M))`, clamped to [1, words/8]. It picks S = 5 / 2 / 12 / 12 for the four shapes, each within one timer step of the measured best.
- One outlier not explained: 4096² at S = 6 (192 blocks) measured 32.5 µs, slower than both S = 5 and S = 8.

### v3: fast FP16 unpack (`src/cuda/ours_v3.cu`)

- **Idea:** build two FP16 weights at once from the packed word with the magic-number trick: `(word & 0x000F000F) | 0x64006400` is the half2 (1024 + q0, 1024 + q4); subtract 1032 → q − 8. The odd rows use mask `0x00F000F0` and `× 1/16 − 72`. Then one `__hmul2` by the scale. No int→float conversion, two weights per instruction.
- **Exactness:** every step is exact in FP16, and (q − 8) × s rounded once by `__hmul2` equals `__float2half_rn((float)(q − 8) × s)`. v3 keeps v2's FMA order and grid, so **its output is bit-identical to v2's**. The bench now checks this on every v3 run (0 differing outputs on every check and sweep point).
- **Expected:** little change at M=1 if v2 is still waiting on memory; a gain wherever the loop is instruction-bound.

### Sweep with v2 and v3 (`results/ours_20261118_201759/`, c8, read flush)

All 160 points pass the FP64 check. Ceiling 427.4 GB/s. M=1, mean µs (% of floor):

| K×N | inblock_specialized | v1 | v2 | v3 | cuBLAS FP16 | v3 vs cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| 4096×4096 | 33.2 (61%) | 32.8 (62%) | 26.6 (76%) | **25.6 (77%)** | 83.3 (94%) | 3.25× |
| 1536×8960 | 27.0 (62%) | 27.9 (58%) | 22.5 (74%) | **22.4 (74%)** | 71.6 (90%) | 3.20× |
| 8960×1536 | 56.9 (29%) | 27.1 (63%) | 23.1 (74%) | **22.5 (74%)** | 71.9 (90%) | 3.20× |
| 1536×1536 | 12.6 (23%) | 8.2 (35%) | 8.2 (35%) | 8.2 (35%) | 14.6 (77%) | 1.78× |

Nsight Compute, M=1 (`ncu_v123.txt`, cold caches, base clocks):

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| 4096²: DRAM throughput | 54% | 73% | 78% |
| 4096²: SM throughput | 74% | 55% | 42% |
| 8960×1536: DRAM throughput | 55% | 72% | 74% |
| 8960×1536: SM throughput | 66% | 55% | 40% |
| registers / theoretical occupancy | 40 / 100% | 44 / 83% | 43 / 83% |

- **v2's hypothesis holds.** Same SMs, same blocks-per-SM target, 4× the bytes per request: DRAM use 54% → 73% and the M=1 plateau moves from ~61% to ~75% of the floor on all three large shapes.
- **v3 barely matters at M=1, as predicted.** At most one timer step faster, but SM throughput drops 55% → 42%: the unpack got cheaper, and the time didn't, because the kernel is still waiting on memory. ncu attributes 74% of v3's stall cycles to waiting on loads (long scoreboard).
- **v3 matters a lot at M ≥ 2** (mean µs, 4096²: M=2 40.9 → 28.7, M=4 76.0 → 53.2, M=8 137 → 97; similar on the other shapes). On 1536×8960 v3 at M=2 (22.5) is the same as at M=1.
  - *Inference, not yet profiled:* each row is a separate block (grid.y = M) and all weights fit in the 32 MB L2, so the second row's reads mostly hit L2 and its cost is re-doing the unpack. That makes the M ≥ 2 slope an **instruction** problem, not the DRAM **traffic** problem written in the baseline section above. To confirm: ncu at M = 4 (L2 hit rate, SM throughput) for v2 vs v3.
- **Crossover with cuBLAS FP16 moves out.** v1 lost to cuBLAS from M = 4 on all three large shapes. v3 beats it through M = 4 everywhere (4096²: 53 vs 84 µs) and loses from M = 8 (4096²: 97 vs 84; 1536×8960: 74 vs 70; 8960×1536: 94 vs 72).
- **What's left at M=1 (~23–26% of the floor):** waiting on memory still dominates the stalls, and SMs are active for 89% of the elapsed cycles (a wave tail). Each thread issues one 16 B load and then waits for it before the next. Keeping two or more loads in flight per thread (unrolling or software-pipelining the k8 loop; Marlin uses a multi-stage `cp.async` pipeline) is the natural next step.
- **1536×1536 stays at 35%:** its floor is 2.9 µs and every version takes ~8.2 µs, so fixed costs (launch, ramp-up) dominate. Not a fair % of floor comparison.

### Marlin under the read flush (`results/ours_marlin_20261118_c8/`)

**Why:** Harsha's E004 timed Marlin either with the store flush (inflates fast kernels, see above) or with no flush, where the 8 MB of 4096² weights stay in the 32 MB L2. His no-flush Marlin time, 13.9 µs, is *below* the 20.3 µs DRAM floor, so it measures L2, not DRAM. Neither matches decode, where a real model's weights are far larger than L2.

*Update 2026-11-20:* this describes E004 only. Harsha's later Marlin comparisons (E045–E050, branch `kernel-ladder-e050`) cycle four weight matrices, which keeps the rectangular shapes out of L2; see the next section for how the two harnesses agree.

**Harness** (`scripts/ours/marlin_compare.{py,sh}`, `scripts/ours/marlin_bridge.cu`; E004's files unchanged): Marlin from the same pinned vLLM 0.33.0 wheel (hash-checked; only `_C_stable_libtorch.abi3.so` extracted), v1–v3 and cuBLAS FP16 through a new bridge. All run in one process on the same operands, with E004's CUDA-graph event timing and three cache policies. Repack verified to keep every code and scale bit. Every output passes the FP64 check before and after timing. 200 iterations, c8.

**Harness check:** it reproduces E004's Marlin (store flush 29.9 vs 29.6 µs; no flush 13.8 vs 13.9) and our C++ bench (v3 25.5 vs 25.6, v1 32.7 vs 32.8, cuBLAS 83.4 vs 83.3), all within one timer step.

**Read flush (cold weights), mean µs, v3 / Marlin in brackets:**

| K×N | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| 4096² Marlin / v3 | 26.2 / 25.5 (0.97) | 26.2 / 28.6 (1.09) | 26.3 / 53.1 (2.0) | 26.8 / 96.8 (3.6) | 27.1 / 173 (6.4) |
| 1536×8960 | 22.5 / 22.4 (0.99) | 22.5 / 22.9 (1.02) | 22.5 / 38.8 (1.7) | 22.5 / 73.7 (3.3) | 24.5 / 139 (5.7) |
| 8960×1536 | 23.1 / 22.5 (0.97) | 23.1 / 24.6 (1.06) | 23.5 / 40.9 (1.7) | 24.1 / 93.7 (3.9) | 24.6 / 179 (7.3) |
| 1536² | 9.2 / 8.1 (0.89) | 9.7 / 8.1 (0.84) | 9.6 / 10.7 (1.1) | 9.4 / 20.1 (2.1) | 10.1 / 33.7 (3.3) |

- **At M=1 with cold weights, v3 and Marlin tie** on the three large shapes: within one ~2 µs timer step, both at 74–78% of the floor. Not a win: Marlin is built for batches (16-row Tensor Core tiles), and M=1 wastes most of each tile.
- **Marlin is flat from M=1 to M=16** (26.2 → 27.1 µs at 4096²): it reads the weights once for all rows and multiplies on Tensor Cores. v3 grows about linearly with M. It's 2–9% behind at M=2, ~1.7–2× at M=4 and 5.7–7.3× at M=16. This is the real gap, and it's at M>1.
- **With warm weights (no flush), Marlin is 15–18% faster at M=1** (4096²: 13.8 vs 16.3 µs). Once memory isn't the limit, Marlin's compute path is leaner. The M=1 tie exists only because both kernels hit the same DRAM wall.
- **Both reach the same ~77% of the floor at M=1.** Marlin's 4-stage `cp.async` pipeline keeps several loads in flight per thread, yet doesn't beat v3 there. So the "more loads in flight" idea from the v3 section may buy little at M=1. *Inference, not tested:* the remaining ~23% could be fixed costs (launch, ramp-up, wave tail) that a 20 µs kernel can't amortize.
- On 1536² v3 is faster (8.1 vs 9.2 µs), but this shape is dominated by fixed costs (floor 2.9 µs), so it says more about launch overhead than about either kernel.

### Harsha's Tensor Core kernel under the same harness (`results/ours_tc_marlin_20261120_c8/`)

**Why:** by 2026-11-18 Harsha's branch `kernel-ladder-e050` (commit `202bc63`) had its own Tensor Core kernel, `tc` (E044), and `tc_ladder` (E050: E048's 32-row tile at M ≥ 64, `tc` below). Its best results are measured with four cycled matrices from M=4 up. Open question: at M=1, which is faster, v3 or `tc`? The two had only been profiled separately (Nsight: v3 78% DRAM busy, `tc` 75%).

**Harness:** the Marlin harness above, with `w4a16_tc.cu`, `w4a16_tc_common.cuh`, `w4a16_tc_m32.cu` and `w4a16_inblock_vec.cu` copied verbatim from `202bc63` (hashes in `meta.txt`). `tc_ladder` uses E050's dispatch rule. The four-file change to our harness is in `harness.patch`; it is not on this branch, because the kernels aren't. Shapes: our four plus Harsha's 4096×11008 and 11008×4096. M = 1–128, three cache policies, 200 iterations, c8. Series 2 repeats M = 1–4 under the read flush with kernel and shape order reversed. Every output passed the FP64 check before and after timing. Tables: `series*/tables_*.txt` (`tc_tables.py`).

**M=1, read flush, mean µs, series 1 / series 2:**

| K×N | Marlin | `tc` | v3 | v3 vs `tc` |
|---|---|---|---|---|
| 4096×11008 | 61.7 / 62.1 | 64.0 / 65.8 | **59.7 / 59.6** | 7–9% faster |
| 11008×4096 | **59.6 / 59.8** | 69.7 / 70.0 | 60.7 / 61.0 | 13% faster |
| 4096² | 26.3 / 25.7 | 27.1 / 27.6 | **25.8 / 25.4** | 5–8% faster |
| 1536×8960 | 22.5 / 22.5 | 22.5 / 22.5 | 22.5 / 22.4 | tie |
| 8960×1536 | 22.7 / 22.5 | 24.6 / 24.6 | **22.5 / 22.5** | 9% faster |
| 1536² | 9.5 / 8.6 | 8.1 / 8.1 | 8.2 / 8.2 | tie (fixed costs) |

- **At M=1, v3 is the team's fastest kernel.** It beats `tc` by 5–13% (time saved) on four of six shapes and ties on the other two, in both series. Against Marlin it's within ±4% on the five large shapes, at 74–91% of the floor.
- **At M=2, v3 and `tc` tie** (within ~1 µs, either way).
- **From M=4, `tc` wins by ~2×, growing with M.** v3's cost grows linearly with M; `tc`'s barely moves until M=16. Harsha's kernels own M ≥ 4.
- **`tc` loses most on the tall shapes** (11008×4096, 8960×1536: 9–13%). These are the shapes where its grid has the fewest blocks (one block per 32 columns: 128 and 48 blocks for 36 SMs). *Inference, not profiled:* this is the under-filled grid that v1's cross-block split fixes. Harsha's E047 tested split-K on `tc` only on 4096² and rejected it; the tall shapes are where it would be expected to pay. But 4096×11008 (344 blocks) also loses 7–10%, so grid size can't be the whole story.

**Harness cross-check against E050's cyclic numbers (Harsha's series 1, c8):**
- **Rectangles agree.** At 4096×11008 and M=4, our read flush gives `tc` 64.2 vs his 64.4 µs, and Marlin 61.6 vs 61.3. At 11008×4096 and M=4, `tc` 71.2 vs 67.5 and Marlin 60.5 vs 59.2. At M=128, `tc_ladder` and Marlin are 3–4% above his. Two different ways of keeping weights cold land within ~5% on the shapes where four matrices (4 × 22.5 MB) far exceed the 32 MB L2.
- **4096² does not agree.** At M=4 his cycle gives Marlin 16.7 / 15.1 µs and `tc` 18.3 / 19.3 µs; the read flush gives 26.2 and 28.1. His values are **below the 20.4 µs DRAM floor**, so part of the data must come from L2. Four 4096² matrices are 4 × 8.4 MB = 33.6 MB, about the size of L2. This likely explains the 10–13% series-to-series swing in Marlin that E050 reports as unexplained at 4096², M ≤ 16. *Inference:* the swing itself was not tested; the below-floor values were measured.

**Fraction of Marlin's speed, `tc_ladder`, read flush, series 1:** 0.92–0.98 at M=4–16 on 4096×11008 and 4096², 0.85–0.86 on 11008×4096. 0.71–0.89 at M=32, where `tc` still uses 16-row tiles. 0.81–1.03 at M=64–128 (1.03 is one cell: 8960×1536, M=64, one series). This agrees with E050's 0.83–0.94 at M ≥ 64.

