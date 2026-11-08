# Project synthesis through E012

This is an offline evidence review, requested on 2026-11-08 from `a78c4a4`. It adds no GPU experiment, kernel change, dispatch, sampler retry, or replacement historical analysis. Existing `results/` experiment directories are inputs only. New reporting outputs live in `results/reporting/project_synthesis/` and `docs/figures/project_synthesis/`.

Scope: read the proposal/course requirements, distinguish original kernels from upstream Marlin, recompute matched comparisons, generate four figures, write `docs/PROJECT_SYNTHESIS.md`, and update the current handoff. Preserve negative/inconclusive decisions. No cross-experiment cumulative speedup or invented confidence interval is permitted.

Verification: check input identities and complete matched row sets, recompute available sample medians, cross-check saved comparison tables, inspect all figures, check document links and quantitative claims, then verify that original sources/results remain unchanged. CPU build/tests are separate from historical CUDA validation. Commit checkpoints without co-author trailers and do not push.

## Reproduce offline figures

From the repository root, use Python 3.10+ with Matplotlib 3.10.7. The exact resolved plotting environment is recorded in [plot-environment.txt](../../results/reporting/project_synthesis/plot-environment.txt). It is a reporting dependency, not a CUDA/runtime dependency. An isolated installation used for this review is:

```bash
UV_CACHE_DIR=/tmp/w4a16-synthesis-uv uv venv --python /usr/bin/python3 /tmp/w4a16-synthesis-venv
UV_CACHE_DIR=/tmp/w4a16-synthesis-uv uv pip install --python /tmp/w4a16-synthesis-venv/bin/python -r results/reporting/project_synthesis/plot-environment.txt
PYTHONDONTWRITEBYTECODE=1 /tmp/w4a16-synthesis-venv/bin/python scripts/reporting/project_synthesis.py
sha256sum -c results/reporting/project_synthesis/figure-inputs.sha256
```

Once the environment exists, regeneration requires no network, GPU, Torch, Nsight installation, original external document, or ARC connection. Run from the repository root for checksum verification. Do not use Python `-O`, which disables the evidence assertions. Only new reporting CSV/JSON/manifests and PDF/PNG figures are regenerated. Historical result directories are opened read-only. PDF timestamps are omitted, so repeated generation in the recorded environment is byte-reproducible.

The script checks complete row sets, duplicate keys, matching comparison settings, identical within-series cyclic operands/storage, numerical pass flags, 100 finite positive samples per timed cyclic row, and exact sample medians. It independently cross-checks E008–E010 retained comparison tables. Its small self-check rejects changed medians, missing/nonfinite samples, and verifies speedup/gain/amortization formulas. It does not replace the historical full correctness/provenance analyzers, which were not rerun because no inconsistency required it.

| Output | Source evidence and statistic |
|---|---|
| [01_kernel_progression.pdf](../../docs/figures/project_synthesis/01_kernel_progression.pdf) | E001/E002 final `timings.csv`, and both E003 `timings.csv` files. These retain recorded medians, not recoverable per-launch samples. Each experiment uses its own same-executable control. E003's first complete timing sweep remains valid despite its subsequent failed profiler import |
| [02_reuse_distance.pdf](../../docs/figures/project_synthesis/02_reuse_distance.pdf) | E008 c10 `series1.jsonl` and `series2.jsonl`, recomputed whole-cycle medians, C gain relative to A |
| [03_cyclic_baselines.pdf](../../docs/figures/project_synthesis/03_cyclic_baselines.pdf) | E009 c4 series JSONL, recomputed complete-cycle medians divided by count. Corrected Marlin, pre-materialized W_hat cuBLAS and timed dequantization+cuBLAS |
| [04_batch_speedup.pdf](../../docs/figures/project_synthesis/04_batch_speedup.pdf) | E010 c4 series JSONL, fresh same-series FP16/Marlin cycle-median ratios at M4/16/64/128 |

Each figure also has a 240 dpi PNG for Markdown. Both series are shown where available, with no confidence intervals. [matched-comparisons.csv](../../results/reporting/project_synthesis/matched-comparisons.csv) contains all 108 plotted comparisons, their source paths and absolute/derived times. [supporting-evidence.json](../../results/reporting/project_synthesis/supporting-evidence.json) recomputes ancillary R1/R2/E007 latency changes, E003 counter reductions, E011 endpoint ratios and E012 attribution bounds/quality failures from saved analysis. These secondary sources retain their original measurement limitations. The 19-input manifest is checked before/after rendering. No input is overwritten.

## Course/proposal review and limitations

[requirements-sources.json](../../results/reporting/project_synthesis/requirements-sources.json) records absolute original document paths and SHA256 values. The three proposal extracts preserve the team PDF, team DOCX and earlier solo DOCX, rather than guessing which was submitted. The course extract preserves PDF pages 5, 6, 10, 18, 19 and 22–25. PDF extraction used `pdftotext -layout INPUT -`, DOCX extraction used Python `zipfile` and `xml.etree.ElementTree` on `word/document.xml` paragraphs. The extracts are included so another checkout can review requirements without access to local files. They establish the saved guidance, not live Moodle submission state or an instructor's interpretation.

The initial default uv cache location was read-only, and the sandboxed package request failed DNS. A `/tmp` cache and approved network installation completed the isolated plotting environment. No project/GPU dependency changed. A missing proposed adapter filename was caught by document-link checking and corrected to the actual E004 `bench.py`. Figure inspection moved legends away from data and added explicit M128 annotations. The staged whitespace check caught CSV's default CRLF endings and an extra blank line at the end of the derived course extract. Emit LF CSV and normalize only that extract's trailing whitespace. These reporting corrections do not change source evidence or numeric values. Local host-only build and CTest pass, with nvcc unavailable. No CUDA build, GPU execution, sanitizer rerun, current scheduler check or new profile was performed or claimed.

## Handoff integrity

`figure-inputs.sha256` hashes the exact 19 saved inputs consumed by the plot/supporting-evidence code. `handoff.sha256` additionally hashes reporting code, final documents, derived tables, figures, extracted requirements and validation logs, excluding itself and its verification log. Historical experiment manifests remain immutable snapshots. Their old documentation hashes should not be rewritten to accommodate this reporting update.
