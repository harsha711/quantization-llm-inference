# In-block K-split experiment record

Status: implementation and local CPU checks complete. ARC GPU evaluation is pending. This file will be updated with exact commands, result paths, measurements, and remaining limits.

## Environment established on 2026-11-06

The user supplied `ssh -i ~/username hpuvvad@arc-off-network`, followed by `ssh hpuvvad@c58`. Job 328410 was RUNNING on c58 in `rtx5060ti16g`, with one node, 16 allocated CPUs, and a six-hour limit ending at 20:00 Eastern. Do not reuse this job ID without checking it again. The user owns the interactive allocation. Leave it running.

c58 exposes one RTX 5060 Ti with 16 GB, driver 615.71.09, CUDA toolkit 13.4.92, and Nsight Compute 2026.3.1. Historical c7 records used driver 610.57.04. CPU is a single NUMA domain with 16 logical CPUs. Full topology, affinity, ISA, compiler versions, hashes, and flags belong in the new result metadata.

The remote source directory is `/home/hpuvvad/csc548-w4a16-inblock-20261106`. The original `/home/hpuvvad/csc548-w4a16` is preserved. Do not use historical results as the denominator of a new speedup.

## Local validation

```bash
bash scripts/build.sh
ctest --test-dir build --output-on-failure
git diff --check
bash -n scripts/check_inblock.sh
```

The local machine has no nvcc. The host-only build and CTest passed. CPU tests include K={1,3,7,8,9,40,65,129,137,257}, N=33, group=128, scale/code/reconstruction checks, zero weights, unused tail nibbles, and exhaustive packed-word partition coverage for word counts 1 through 520.

## ARC launch environment

An SSH shell on c58 does not inherit the interactive job's Slurm variables. After verifying the live allocation, the following environment made ARC `prun` launch exactly one process through `mpirun`. This is a launcher inside an existing allocation, not a resource request.

```bash
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1
export SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node
export OMPI_MCA_hwloc_base_binding_policy=none
prun /bin/hostname
```

The expected application output is one `c58` line, alongside the launcher's diagnostic lines. The single process may use the 16 allocated CPU cores for the FP64 reference. No MPI work is performed by the benchmark.

## Measurement plan

Rerun naive, inblock, original-weight cuBLAS, reconstructed-weight cuBLAS, dequantization, and unfused dequantization plus cuBLAS for M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, and 100 timed iterations. Run both the existing intended cold policy and no-flush repeated launches. Median and p90 are CUDA-event latencies. Speedup is naive median divided by inblock median in the same cache policy and shape. Setup, quantization, transfers, and flushing remain outside the timed interval.

On a 32 MiB L2 device the flush writes 64 MiB before the start event. Store traffic can affect subsequent DRAM activity and does not guarantee that every input line was evicted. Warm runs do not flush between launches and do not lock all input data in cache. Nsight profiling will use explicit `--cache-control all`, without the application's flush, and must be described separately.

For M=1 and M=128, profile naive and inblock after querying supported metrics. Record grid/block sizes, registers/thread, shared memory/block, achieved occupancy, scheduler eligibility/issue activity, long-scoreboard and other relevant stalls, SM/memory utilization, and measured DRAM read/write bytes. Keep timing runs separate from profiling. Interpret increased block count as a launch property, not a guaranteed occupancy increase.
