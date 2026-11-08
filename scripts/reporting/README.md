# Project synthesis through E012

This is an offline evidence review, requested on 2026-11-08 from `a78c4a4`. It adds no GPU experiment, kernel change, dispatch, sampler retry, or replacement historical analysis. Existing `results/` experiment directories are inputs only. New reporting outputs live in `results/reporting/project_synthesis/` and `docs/figures/project_synthesis/`.

Scope: read the proposal/course requirements, distinguish original kernels from upstream Marlin, recompute matched comparisons, generate four figures, write `docs/PROJECT_SYNTHESIS.md`, and update the current handoff. Preserve negative/inconclusive decisions. No cross-experiment cumulative speedup or invented confidence interval is permitted.

Planned verification: check input identities and complete matched row sets, recompute available sample medians, cross-check saved comparison tables, inspect all figures, check document links and quantitative claims, then verify that original sources/results remain unchanged. CPU build/tests are separate from historical CUDA validation. Commit checkpoints without co-author trailers and do not push.
