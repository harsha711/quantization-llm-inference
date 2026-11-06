# E007 terminal shared-to-register lookahead

E007 is a completed negative latency result. It tests one guard around the unused final A/B register fetch in the analyzed eight-row Marlin specialization. E006's synchronization barrier and the complete async-copy pipeline remain intact. Quantization, G128, scales, arithmetic and host launch policy are unchanged. All previous experiment roots and evidence remain preserved.

Read `AGENTS.md`, `docs/STATUS.md`, the E006/E005-R1/E007 experiment entries and E006's access analysis first. The actual root is `/home/hpuvvad/csc548-w4a16-e007-20261106`. Two matched libraries and complete generated sources remain there. Locally, `matched-source.tar.gz` preserves the source trees, while the `.so` files stay on ARC with recorded hashes. Do not overwrite those libraries or earlier roots.

Run the stdlib-only evidence checker locally:

```bash
python3 scripts/e007/analyze.py results/e007_20261106
```

The analyzer checks matched source/SASS, all 270 regular numerical and 162 sanitizer rows, full-output hash agreement, 56 timing rows/5600 samples, exact-symbol profiles and fixed criteria. Source-generation self-checks model A/B double buffering for stripe lengths 1–86. Negative-data checks reject a corrupted timing sample. All 16 control SASS functions match E006, and 15 candidate fallback functions remain identical.

In the isolated ARC root, historical E006 evidence stays in its original directory. Source `env_arc.sh` and pass `--e006 "$E006_CONTROL/results/e006_20261106"` to the analyzer under `prun "$E005_PYTHON"`. The local default uses the repository's preserved E006 results.

The initial static checker counted only unpredicated LDS/LDSM sites. Corrected accounting includes plain scalar LDS, dotted LDS, LDSM and predicates. Both bodies have 121 static load opcode sites, including 70 permanently predicated-off sites, leaving 51 potentially active sites. Eight become conditional in the candidate. Preserve `check.py.measured.txt`, `analyze.py.measured.txt`, both initial derived JSON files and all raw gate logs. Current analysis resolves original execution-manifest hashes to those snapshots. Current corrected scripts are recorded by the final handoff manifest. No timing input, binary or acceptance criterion changed in this reporting correction.

The tested interactive stage sequence is below. Before any future replay, verify current ARC documentation, actual allocation ownership/state/node/remaining time, GPU UUID/tool/runtime and all fixed control/environment paths in `env_arc.sh` and the reused E005-R1 helper. Use a fresh absolute root and preserve old filenames. A new trial requires a separately recorded hypothesis and authorization, not an unchanged repeat of this rejected guard.

```bash
export PROJECT_ROOT=/absolute/new/shared/project/root
export SLURM_JOB_ID=VERIFIED_JOB SLURM_JOB_NODELIST=VERIFIED_NODE
# Stage unchanged scripts/e004, scripts/e005, scripts/e005_r1, scripts/e006,
# scripts/e007 and current continuation docs. All fixed historical roots must exist.
mkdir -p "$PROJECT_ROOT/results"
bash "$PROJECT_ROOT/scripts/e007/build_arc.sh"
bash "$PROJECT_ROOT/scripts/e007/validate_arc.sh" smoke
# Inspect actual source/SASS, runtime/launcher and full-output smoke evidence.
bash "$PROJECT_ROOT/scripts/e007/validate_arc.sh" full
# Inspect all numerical and sanitizer gates before releasing timing.
# Preserve the exact reporting scripts referenced by the new execution manifests.
cp "$PROJECT_ROOT/scripts/e007/check.py" "$PROJECT_ROOT/results/e007_20261106/check.py.measured.txt"
cp "$PROJECT_ROOT/scripts/e007/analyze.py" "$PROJECT_ROOT/results/e007_20261106/analyze.py.measured.txt"
bash "$PROJECT_ROOT/scripts/e007/measure_arc.sh"
bash "$PROJECT_ROOT/scripts/e007/profile_arc.sh"
```

All GPU/build/runtime launches use `prun` inside the verified allocation. No `srun` or gateway compute occurs. These helpers use the exported absolute PROJECT_ROOT. For unattended jobs use `sbatch`, preserve that absolute path and require successful prerequisites. Never infer project paths from a spooled script's BASH_SOURCE. Historical allocation 328410 is not evidence of future availability. All stages above were executed successfully on ARC, with the later static-reporting correction checked separately.

`prepare_source.py` copies the corrected E006 tree and changes one main-loop fetch call. The guard tests `slice_iters > 1`, since the counter decrements after the inner loop. Its compile-time scope is exactly the analyzed FP16/U4B8/G128 eight-row,256-thread,128Kx128N,four-stage specialization. `bench.py` subclasses the retained E005 `Case` only to hash every output outside timing, and calls the unchanged checker/timer. Each process loads one local body. The wheel supplies lossless repacking only.

Timing compares fixed host A for both bodies: 36 blocks/101376 B dynamic shared. Each point has 10 warmups and 100 common graph-event samples under both operational cache policies. Series 1 runs control primary/fallback before candidate primary/fallback, and series 2 reverses the body order. Held-out seed 2 M2/4 cases follow without retuning. Store flush, capture, allocation, references and hashing remain outside GEMM event intervals. Required kernel reductions/lock work remain inside.

M8 medians tie. M1 changes by −0.355% and 0.000%, missing the fixed 5% criterion. Participating LDS instructions fall 1.931%, while predicate-inclusive load issue remains equal and total warp instructions rise 1.757%. Registers 126→124 still allocate 128. Two M8 profiles use 7 replay passes, cache-control none and unlocked clocks. Profiler duration/traffic is separate from application timing and does not demonstrate complete residency. No further guard, barrier removal, C-policy combination, automatic dispatch, roofline or full LLM serving experiment is included.
