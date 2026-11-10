# E014 full-model overhead investigation

**Completed at a failed numerical gate; no primary or held-out timing released.**
Read the [report](../../docs/INFERENCE_OVERHEAD.md) and
[predeclared design](../../docs/EXPERIMENTS.md#e014--single-request-full-model-execution-overhead).
Do not retry unchanged, relax tolerance or release a timing subset.

Pinned Qwen2.5-1.5B model/tokenizer revision
`8faed761d45a263340a0528343f099c05c9a4323`, common FP16 parameters/KV,
batch size 1, 128/2048 input tokens, 64 greedy output tokens. EOS is recorded
and ignored for fixed work. A is eager dynamic KV, B eager static KV, C static
KV plus compiled decode. All use the same enforced memory-efficient SDPA
backend and explicit causal masks.

## Offline reproduction

From the repository root, with standard-library Python and assertions enabled:

```bash
python3 scripts/e014/check_local.py
python3 scripts/e014/analyze.py results/e014_20261108
python3 scripts/e014/check_analysis.py results/e014_20261108
sha256sum -c results/e014_20261108/handoff.sha256
```

The default analyzer checks the **stopped** record, numerical failure,
identities/outputs, actual short-context replay and absence of comparison
samples. The negative checker tests six damaged records in temporary copies.
`check_local.py` tests interval unions, mirrored annotations, launch gaps,
reversed orders, the 5% rule and invalid timing/sample records. No GPU needed.
Host build/CTest checks do not validate CUDA or this model.

Seven raw Chrome traces are losslessly compressed as `timeline_*.json.gz`.
`trace-archive.json` stores original/compressed hashes and sizes; `gzip -dc`
recovers exact original bytes. Both readers support gzip. `reference_*.npz`
stores all-vocabulary FP16 reference logits. Five completed validation JSON
records retain outputs, errors, parameter hashes, warmups, memory and compiler
counters. C2048 log/exit retains the failure; the original fail-fast validator
did **not** retain its failed tensor or partial JSON. That limits further error
characterization. It is not a missing pass.

`pip-freeze.txt`/`requirements.lock` pin the runtime. `model.json` pins file
hashes; the 3 GB checkpoint remains in the isolated ARC cache.
`model-metadata-tokenizer.tar.gz` preserves the other downloaded files,
including tokenizer, config, model card and license. `runtime-source/`,
`runtime-source.json` and `inductor-cache-at-stop.tar.gz` retain relevant
installed source/distribution metadata and generated code.
`executed-tooling-at-stop.tar.gz` preserves staged scripts at failure,
including the original analyzer before its offline missing-key correction.
`diagnostic-source/` preserves the earlier baseline-only harness revision.

`negative-check.log`/`.exit` and job 330286 remain failed: preservation succeeded,
then its checker mishandled intentionally removed compiler counters. Corrected
`local-negative-check.log` passes. The failed ARC log and first local reproduction
remain; no inference was repeated for this correction.

## Historical ARC execution

Shared project `/home/hpuvvad/csc548-w4a16-e014-c4-20261108`; isolated Python
environment `/tmp/e014-hpuvvad-20261108` on c4. These paths/node/job IDs are
historical, not proof of a future allocation. Verify current ARC documentation
and live state before any separately requested continuation. No gateway compute.

Scheduler-recorded exact submissions are in `final-submissions-queue.txt`.
From that absolute shared directory:

```bash
sbatch --parsable --export=ALL,PROJECT_ROOT=/home/hpuvvad/csc548-w4a16-e014-c4-20261108 --output=setup-slurm-%j.log scripts/e014/setup_arc.sh
sbatch --parsable --dependency=afterok:330278 --export=ALL,PROJECT_ROOT=/home/hpuvvad/csc548-w4a16-e014-c4-20261108 --output=diagnose-slurm-%j.log scripts/e014/diagnose_arc.sh
sbatch --parsable --dependency=afterok:330280 --export=ALL,PROJECT_ROOT=/home/hpuvvad/csc548-w4a16-e014-c4-20261108 --output=validation-slurm-%j.log scripts/e014/validate_arc.sh
sbatch --parsable --export=ALL,PROJECT_ROOT=/home/hpuvvad/csc548-w4a16-e014-c4-20261108 --output=seal-slurm-%j.log scripts/e014/seal_arc.sh
```

Each dependent stage was submitted **after inspecting** its prerequisite.
Setup 330278/diagnostic 330280 pass; validation 330282 fails numerically.
Seal 330286 archives, then fails in the original negative checker. Executable
commands are traced in per-stage logs. Scripts use `prun timeout 1800 ...`,
one MPI process/16 host threads on one GPU, and exported absolute PROJECT_ROOT.
Allocations, metadata and accounting retain CPU/ISA/NUMA/topology/affinity,
tools, GPU UUID/driver and state. Final queue is empty; no user allocation cancelled.

`DIAGNOSTIC_CAPTURED`/`DIAGNOSTIC_INSPECTED` record the successful prerequisite
and manual review. `VALIDATION_CAPTURED`/`VALIDATION_INSPECTED` remain absent.
**Never create them manually to bypass failure.** Prepared `measure.py`,
`measure_arc.sh`, `heldout_arc.sh` and nondefault analyzer stages document the
predeclared timing procedure; they were not released or ARC-validated as a
campaign. No primary/held-out samples, medians, TTFT/E2E speedup or 5% result exist.

Transformers supplies Qwen2/caches; PyTorch supplies SDPA, compilation, fusion
and CUDA graph replay. This does not invoke the local W4A16 Marlin kernel or
change its numerical contract.
