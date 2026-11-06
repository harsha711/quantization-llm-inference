# W4A16 CSC 548 project instructions

Read `README.md`, `docs/design-decisions.md`, and `docs/experiment-inblock.md` before continuing. Preserve the existing implementations and historical `results/baseline_*` and `results/ncu/` records. Implement only the experiment requested by the user. Propose another optimization separately and wait for a request to implement it.

## Numerical contract and layout

- Compute row-major C = A B. A and C are FP16. Accumulate in FP32. The CPU reference accumulates in FP64.
- Scale is FP16(max(abs(W))/7). Quantization uses nearbyint(W/s), clamped to [-7,7], then stores q=z+8. If the stored scale is zero, z=0. Code 0 is unused for valid weights.
- Reconstruct FP16((q-8)*s) before converting to FP32 for accumulation. Preserve this rounding in every kernel.
- `qweight[k8,n]` has adjacent columns in adjacent uint32 words. Eight consecutive K codes occupy the nibbles in increasing K order. Allocate ceil(K/8)*N words and ceil(K/group)*N scales. Unused tail nibbles are zero and must never be consumed.
- Groups remain multiples of eight. The final K word and quantization group may be partial. N may be arbitrary positive. Do not add a new packed layout as part of the in-block experiment.

## Workflow and evidence

- Check git status first and preserve unrelated changes. Commit validated checkpoints as work proceeds. Do not add co-author trailers to commit messages. Do not push unless requested.
- Keep `docs/design-decisions.md` current with the problem, alternatives considered, chosen design, evidence, failed attempts, and limits. Separate untested alternatives from measured failures. Keep exact reproduction commands and result paths in `docs/experiment-inblock.md`.
- `bash scripts/build.sh` supports CPU-only development. Run `ctest --test-dir build --output-on-failure`. On a GPU, run `bash scripts/check_inblock.sh` before the main experiment. This includes FP64 checks and Compute Sanitizer when available.
- Never relax `|error| <= 1e-3*|reference| + 1e-3*rms(reference)` to accommodate a changed reduction order. Diagnose failures and stop measurements. Preserve logs.
- Compare naive and inblock in the same executable, seed, cache policy, compiler flags, launch configuration, and node. Rerun all M points if any of these change. Use M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, and at least 100 timed iterations.
- Preserve the existing event timing and store-based flush outside the start event. Record both this intended cold policy and `--no-flush-l2`. The store flush does not prove eviction. Nsight `--cache-control all` is a separate method. Model-based `eff_gbps` is not measured DRAM bandwidth.
- Query profiler metrics on the target GPU. Save raw logs, reports, commands, metadata, source/executable hashes, and build flags under a new result directory. Profile separately from latency measurements.

## ARC

- Verify current [ARC documentation](https://arcm.csc.ncsu.edu/~mueller/cluster/arc/) and `scontrol show job` before designing or running launch scripts. Allocation IDs and node names in saved commands are historical. Confirm ownership, RUNNING state, nodes, remaining time, and GPU availability each session.
- Use `sbatch` for unattended jobs, `salloc` for interactive allocations, and ARC `prun` inside allocations. Do not invoke `srun` unless the user explicitly overrides this after confirming a policy change. Do not perform compute on the gateway.
- Batch scripts are spooled. Export an absolute project path at submission. Never derive project paths from `BASH_SOURCE` in a batch script. Use `SLURM_SUBMIT_DIR` only after validating the required files. Repository helper scripts may locate themselves when executed from their real shared-storage paths.
- Before expensive jobs, run a cheap prerequisite using the actual executable, launcher, paths, storage, rank/thread layout, and small correctness case on the intended topology. Inspect the output before release. Make expensive batch jobs depend on successful prerequisite jobs.
- Capture per-node CPU model, ISA, topology, NUMA, affinity, compiler/MPI versions, source hashes, build flags, and raw logs. Use a portable host ISA. Never distribute a `-march=native` binary without checking every allocated node's required ISA.
- For this non-MPI benchmark, launch one process on one allocated GPU node. Verify the rank count with `prun /bin/hostname` before launching benchmarks. Record OpenMP threads and binding. Do not release an allocation created by the user.
- The old `scripts/remote.sh` defaults to c7, uses destructive `rsync --delete`, and does not establish a `prun` job environment. Do not use it blindly. Use a distinct experiment directory and the verified live allocation. Never claim target validation from local tests alone.
