# E015 direct graph replay

Compare E (eager static cache) with G (the same static decode captured by
`torch.cuda.CUDAGraph`). The E014 model, numerical bounds, attention implementation,
workload and request loop are reused unchanged. Both variants stage the current
mask into a persistent buffer. No `torch.compile` or new device kernel is used.
The E015 subclass captures forward+argmax on its first decode call after three
side-stream warmups at the same cache position. Subsequent positions are device
inputs. Prefill and mask/position/input updates stay outside capture.

Read the predeclared E015 entry in `docs/EXPERIMENTS.md`. Historical E014's failed
compiled path is not rerun. Every numerical comparison saves candidate/reference
FP16 logits and error statistics before asserting. Numerical failures also save
indices. Full validation alternates primary/held-out/primary requests, checks
cache resets/progression and stable pointers, and traces actual replay at each
context. These numerical checks are distinct from the unchanged GEMM contract.

Initial ARC root: `/home/hpuvvad/csc548-w4a16-e015-c7-20261109`.
The home-quota-interrupted attempt is preserved there and in `attempt1/` under
the fresh root `/mnt/beegfs/hpuvvad/csc548-w4a16-e015-c9-20261109`.
BeeGFS is available only on compute nodes. Submit `continue_arc.sh` via stdin
from home, exporting the absolute BeeGFS PROJECT_ROOT and MODE=storage,
validation, primary or heldout. Each stage writes its CAPTURED marker, then
waits up to ten minutes for inspected retrieval through the active allocation
and a matching RETRIEVED marker. Subsequent expensive stages depend on the
preceding successfully completed job. The storage prerequisite repeats only
small E/G path/capture checks. Fresh c9 validation repeats the retained-project build/CTest/sanitizer checks
before its full-model cases. No c7 measurements become timing denominators.
Use the absolute exported PROJECT_ROOT and sbatch/prun. Verify current ARC policy,
job ownership/state and actual selected GPU every session. Saved node/job IDs are
historical. The exact E014 checkpoint cache is reused read-only after file hashes;
the new Python environment is `/tmp/e015-hpuvvad-20261109` with the E014 lock.

Stages: `setup_arc.sh` installs pinned dependencies and runs eight-token E/G smoke
on two passages. Inspect it before recording SMOKE_INSPECTED. Dependent
`validate_arc.sh` builds/checks the retained project and runs all four full
variant/context cases. Inspect validation before recording VALIDATION_INSPECTED.
Submit `measure_arc.sh` with `PHASE=primary` only after that gate. It uses two
reversed independent process orders, three request warmups and twenty samples
per cell (160 primary requests). A >=5% decode-median gain in both series per
context releases the identical held-out campaign, with PHASE=heldout. Compare
only fresh matching E/G medians. No passing-subset timing after a validation
failure, no new compiler setting, attention path, precision, or kernel search.

Offline checks (standard-library Python):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e015/check_local.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e015/analyze.py RESULTS --stage validation
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e015/check_analysis.py RESULTS validation
```

Use stage `primary` or `heldout` for released timing records. The checkers write
derived analyses, so use temporary evidence copies when preserving sealed results.
Timing includes common mask/input updates and final synchronization. TTFT and E2E
retain E014's boundaries, with cold graph setup recorded separately. Inspect
memory, other-latency/context regressions and kernel sequence comparisons before
interpreting a gain as launch-overhead reduction.
