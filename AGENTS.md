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
- Never relax `|error| <= 1e-3*|reference| + 1e-3*rms(reference)`. Fail immediately on correctness/launcher errors and preserve diagnostics. No partial measurement claims. E002 additionally requires all output bits to match retained E001 outside timing, including rows not sampled by FP64 checks.
- Compare naive/inblock/inblock_specialized in the same executable, seed, flags, node, and cache policy, retaining each defined launch mapping. Sweep M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, at least 100 iterations, both flush and no-flush. Store flushing does not prove eviction. Query target metrics and profile separately. `eff_gbps` is modeled, not measured DRAM bandwidth.

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
