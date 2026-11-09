# E016 full-decode CUDA graph

One bounded comparison of unchanged E015 per-token replay G and one graph over
63 decode steps H. See the predeclaration in `docs/EXPERIMENTS.md`.
Prefill stays eager. Both paths always save full logits into a GPU slab and
record device position/token feedback, including during timing. H restores an
exact post-prefill snapshot after warmup and capture, before the first replay.
No compilation, fusion, attention change or custom device kernel.

Local checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e016/check_local.py
bash scripts/build.sh
ctest --test-dir build --output-on-failure
python3 scripts/e016/analyze.py <copied-evidence> --stage validation
python3 scripts/e016/check_analysis.py <copied-evidence> validation
```

ARC uses the existing pinned E015 Python environment only after exact lock
comparison, an isolated BeeGFS root, and a read-only pinned model cache. The
setup stage checks the installed source, rank/path/storage and an eight-token
G/H primary/held-out/primary sequence. Inspect saved tensors, cache hashes and
actual launches before creating SMOKE_INSPECTED. Full validation must depend
on that successful stage. Inspect it before releasing primary timing. Primary
and conditional held-out campaigns each have eight fresh processes, three
warmups and 20 requests per process, with reversed context/variant orders.
Never create a gate after a failed stage. The EXIT trap preserves the owned
allocation briefly for retrieval even after failure, without more compute.
Submit from home via stdin with an absolute PROJECT_ROOT, MODE, afterok and
an output path whose directory already exists. Stage scripts do not locate
project files from their Slurm-spooled path. Use prun inside allocations.
Historical E014/E015 scripts and evidence remain unchanged. No push.
