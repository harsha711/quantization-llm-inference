# W4A16 CSC 548 continuation rules

## Every session

- Read `README.md`, `docs/STATUS.md`, `docs/CONTRACT.md`, and relevant `docs/EXPERIMENTS.md` entries before editing. Consult `docs/OPTIMIZATION_MAP.md` and `docs/REFERENCES.md` before selecting/interpreting experiments. Verify the handoff against `git status`, `git log`, code, and raw evidence.
- Implement only the requested experiment. Choose from measured evidence, state hypothesis and success criterion before implementation, and change one main optimization at a time. Propose other optimizations separately and wait for a request.
- Preserve baseline implementations and raw records, especially `results/baseline_*` and `results/ncu/`. Never silently change numerical tolerance, layout, or benchmark conditions. Rerun every series point when executable, flags, kernel, ranks/threads, placement, or topology changes.
- Record every attempt, alternatives, selected design, failures, exact version/commands, validation, raw results, decision, and revisit conditions in `docs/EXPERIMENTS.md`. Untested ideas are not measured failures. Record inconclusive evidence as inconclusive. Do not retry rejected experiments unless their revisit condition is met.
- Update `docs/STATUS.md` and experiment records before ending. Update contract/map/references when affected. Cite hardware assumptions. Distinguish documented facts, observations, calculations, and hypotheses.
- Commit validated checkpoints as work proceeds. No co-author trailers. Do not push unless requested. Preserve unrelated changes.

## Build and numerical checks

- Run `bash scripts/build.sh`, then `ctest --test-dir build --output-on-failure`. CPU-only passes do not validate CUDA. On a GPU run `bash scripts/check_inblock.sh` before measuring. Inspect FP64 and Compute Sanitizer results. Record unavailable checks.
- Compute row-major `C=A*B`, FP16 storage, FP32 accumulation, FP64 CPU reference. Reconstruct `FP16((q-8)*s)` before FP32 accumulation. Preserve packing/scales and group multiples of eight. See `docs/CONTRACT.md` for exact rounding, tails, tolerance, commands, and methodology.
- Never relax `|error| <= 1e-3*|reference| + 1e-3*rms(reference)`. Fail immediately on correctness/launcher errors and preserve diagnostics. No partial measurement claims. E002 requires all output bits to match retained E001 outside timing. E003 selectors `inblock_m2`/`inblock_m4` require full-output bitwise agreement with E002, including rows not sampled by FP64 checks.
- Compare naive/inblock/inblock_specialized/inblock_m2/inblock_m4 in the same executable, seed, flags, node, and cache policy, retaining each defined launch mapping. Sweep M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, at least 100 iterations, both flush and no-flush. E003 additionally measures M=5 with the same conditions as a separate row-tail point. Preserve the fixed two/four-row selectors and incomplete-tile bounds. Store flushing does not prove eviction. Query target metrics and profile separately. `eff_gbps` is modeled, not measured DRAM bandwidth.

## E004 external comparator

- Read E004 before using Marlin. Keep vLLM v0.33.0/source `4ba787ec9959e566d3b587a796dff07673324da5` and its official binary provenance explicit. Use isolated Python 3.11/Torch 2.14, direct dense uint4b8 operator, lossless codes/scales, FP32 global reduction. No serving/model dependencies or silent fallback. Do not change external kernels without a separately requested experiment.
- E004 uses fresh common graph-event comparisons and explicit cuBLAS FP32-reduction math mode 16. Preserve both cache policies, raw samples, all-row FP64 checks, full retained controls and failed attempts. Do not use historical medians as E004 denominators. `scripts/e004/analyze.py` validates recorded evidence. E005 is implemented but stopped at its failed external racecheck gate. Read E005 before any Marlin rerun. Historical stage/resume helpers have fixed evidence paths, so use a distinct output root for reruns and never overwrite records. ARC prun flattens quoted `python -c` commands: use a script file.

## E005 stopped safety gate

- Read E005 and `results/e005_20261106/GATE1.md`. A/B/C share compiled device code and differ only in the gated host policy. Preserve same-build A and untouched upstream source. Occupancy queries predict capacity, not achieved residency.
- E005 memcheck/synccheck and numerical checks pass, but racecheck reports hazards in A/B/C and isolated official wheel. No primary/held-out timings or Nsight counters were released. Never create VALIDATED manually, suppress these reports, or treat passing outputs as race freedom. Revisit only after a separately authorized resolution, then a new matched build and complete validation/series. No next optimization is authorized.

## E006 synchronization correction

- Read E006 and `results/e006_20261106/ACCESS_ANALYSIS.md` before further Marlin work. One source-supported block barrier after the existing stripe-final async drain removes the reported hazard in the tested A/B/C matrix. The original terminal lookahead is unused in the analyzed small-M specialization. No incorrect original output or applicable sanitizer limitation was established.
- Preserve E005 unchanged. E006 control/patched sources, patch, matched binaries/hashes and all failed diagnostics remain separate. The unpatched control still fails racecheck. The corrected body passes full numerical/replay/workspace checks and representative memcheck/racecheck/synccheck on c58. This is bounded evidence, not general validation of other formats/hardware or exact wheel source mapping.
- E006 authorizes no timing. E005's old VALIDATED remains absent. Await a separately requested E005 continuation using the SAME corrected body for A/B/C, with complete validation and all original timing/held-out/profile criteria. Never splice original-body or wheel times into corrected comparisons. No next optimization is authorized.

## E005-R1 corrected-body performance continuation

- The user authorized E005-R1 after E006. Reuse the exact corrected E006 binary and verified validation when source/binary/environment identity holds. Rebuilding or changed conditions require relevant fresh validation. Keep original E005/E006 records immutable and their VALIDATED sentinels absent.
- Use a separate `scripts/e005_r1/` and `results/e005_r1_20261106/`. Compare corrected A36/101376 B, B36/49664 B and C72/49664 B using the same corrected entrypoint. Only corrected A is a performance denominator. Preserve both cache policies, two balanced primary/fallback series, seed-2 held-outs and targeted M8 A/B/C profiles under the original 5% criteria. No new optimization or automatic dispatch.
- E005-R1 is complete and rejected under the primary latency criterion. C increases observed concurrent CTA activity but gives only 0–0.237% M1 and 2.653–2.765% M8 no-flush gains. B gives no consistent primary gain. Held-out rectangular benefits do not override that criterion. Run its analyzer before interpreting results. Preserve failures, samples, profiler reports and corrected A. Await a user request before another optimization or a repeat with changed conditions.

## Authorized continuation while user is away

- The user authorized continued project work after E005-R1. E007 is one bounded source-supported terminal-lookahead experiment, predeclared in EXPERIMENTS.md. This overrides the earlier instruction to wait at the completed E005-R1 checkpoint only for this continuation. Keep E005/E006/E005-R1 immutable. Preserve the E006 barrier, fixed format and host launch policy. Do not turn it into a tuning search.
- E007 is complete and rejected under its5% latency criterion. Participating LDS loads fall 1.931% but issued warp instructions rise 1.757%, with M8 median ties and no reproducible M1 gain. Registers126→124 still allocate128. Preserve measured checker snapshots and corrected static counts: 121 total load opcode sites, 70 permanently off, 51 potentially active in each body. Do not repeat this guard unless its recorded revisit condition changes. Hand off before another optimization.

## E005-R2 rectangular confirmation

- E005-R2 is complete. Read its experiment entry and fresh c10 evidence. Both M=4/seed 2 rectangles pass 5% lower C no-flush median in both reversed series,16 rows/1600 samples. Stronger K=4096/N=11008 gain 13.33%/13.28% has store-flush regression5.48%/6.06%. Corrected E006 A remains default and E005-R1/E007 stay negative.
- One conditional A/C profile pair shows more ready work with higher instruction/L2 overhead and high replay DRAM throughput. Replay duration differs from application timing. No concrete improved partitioner or store-flush cause is established. Run E005-R2 timing/profile analyzers and negative-data check before interpretation. c10's changed driver/runtime received fresh relevant validation. Preserve all old source/binaries/results.
- Stop after this confirmation. No additional measurements, E008, automatic dispatch or partition change is authorized. Await the user's next bounded investigation request. Revalidate c10 allocation 328681, do not assume saved state remains current.

## ARC

- Verify current [ARC documentation](https://arcm.csc.ncsu.edu/~mueller/cluster/arc/) and `scontrol show job` each session. Confirm allocation ownership, RUNNING state, nodes, remaining time, and GPU. Saved IDs/nodes are historical. Do no compute on the gateway.
- Use `sbatch` for unattended jobs, `salloc` interactively, and ARC `prun` inside allocations. No `srun` unless the user explicitly overrides this after confirming a policy change. Use one process on one GPU node. Verify one rank with `prun /bin/hostname`, record OpenMP threads/binding, and leave user-created allocations running.
- Batch scripts are spooled. Export an absolute project path at submission. Never derive it from `BASH_SOURCE` inside batch jobs. Use `SLURM_SUBMIT_DIR` only after validating project files. Helpers executed from real shared-storage paths may locate themselves.
- Inspect a cheap prerequisite using the actual executable, launcher, paths, shared storage, rank/thread layout, topology, and small correctness case before expensive work. Expensive batch jobs must depend on successful prerequisites. Start with the smallest representative smoke test.
- Capture per-node CPU model, ISA, topology, NUMA, affinity, compiler/MPI/GPU/tool versions, source/executable hashes, build flags, commands, and logs. Multi-node CPU generations may differ. Use a portable host ISA. Never distribute `-march=native` without verifying every node's required ISA.
- `scripts/remote.sh` defaults to c7, uses `rsync --delete`, and lacks the allocation environment. Use isolated experiment directories and verified live allocations. Never claim ARC validation or guaranteed script operation from local checks alone.

## Rooflines, if requested

- Recompute all ceilings, ridge points, arithmetic intensities, and performance coordinates from specifications/data. Keep the primary full-machine roof distinct from core-count, ISA, NUMA, and memory-domain sensitivity roofs.
- Keep cores, clock convention, SIMD width, operation count, FMA throughput, and bandwidth scope consistent across equations/plots/captions/classifications. An AVX/FMA roof is not an SSE/non-FMA build's applicable ceiling.
- State finite problem/block traffic models. Label asymptotic approximations/counter proxies and distinguish modeled traffic from measured DRAM. Verify report values, generated labels, packaged plots, and bound classifications agree.
