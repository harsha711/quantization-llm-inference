# Retained analysis inputs

The final E012 handoff includes four unchanged E011 Marlin target-trace JSON files and E011 `profile-analysis.json` as reference inputs. Copies in the isolated E012 tree support its analysis. They do not constitute the entire E011 raw dataset. The complete immutable ARC record remains `/home/hpuvvad/csc548-w4a16-e011-c4-20261107`.

The unchanged E006 `matched-source.tar.gz` and `patched/build-source.sha256` support source verification. No kernel is rebuilt. The original corrected shared library stays in the preserved E006 tree and is identified by its recorded SHA256. See `verified-inputs.sha256`, the full experiment record and source counter checks.

`analysis-initial/snapshot/` preserves the exact three analysis sources from commit `181f6fe`. Its original source hashes verify from that snapshot directory. Final analysis differs only by deterministic ordering of phase-shift dictionary keys. Both CPU runs pass and preserve all eight sampling quality failures.
