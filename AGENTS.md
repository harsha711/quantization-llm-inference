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
