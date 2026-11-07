# E012 source-PC attribution

Predeclared at `83b1e72`. Corrected A and the frozen E011 harness remain unchanged. Read the [design and quality gates](../../docs/EXPERIMENTS.md#e012--bounded-source-pc-attribution-of-corrected-marlin) before releasing measurements.

First inspect `capability_arc.sh` on c52 in one sbatch/prun rank with an exported absolute PROJECT_ROOT. It records allocation/hardware, verifies the complete preserved E011 handoff, queries the source and warp-sampling collections separately, exports existing exact-binary source/SASS correlation, and inspects the installed report API. This metadata/source-import stage does not validate a new GPU workload or authorize real profiles by itself. Inspect its result before runtime restoration, actual-executable correctness prerequisites and any dependent collection.

No old source/results are overwritten. An unsupported metric/replay combination is an inconclusive branch, not a reason to substitute an isolated workload or rebuild the kernel.
