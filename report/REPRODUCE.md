# Building the report and reproducing its numbers

## Report PDF
`main.tex` uses the ACM `acmart` class (`sigconf`), `references.bib` and `figs/*.pdf`. Either upload this folder to Overleaf (ACM template compiles out of the box) or, with TeX Live installed: `latexmk -pdf main.tex`.

## Figures and numbers
```bash
uv venv /tmp/plot && uv pip install --python /tmp/plot/bin/python -r results/reporting/project_synthesis/plot-environment.txt
/tmp/plot/bin/python scripts/reporting/report_figures.py results/e041_20261117/full-342501   # writes report/figs/
python3 scripts/reporting/check_report_numbers.py                                           # text matches figs/numbers.json
```
Inputs (read-only): E041 CSVs, E045 `summary.json` (roofline: our `tc`, corrected Marlin, cuBLAS), E027 `analysis.json`, E039/E040/E042 `quality.json`. The checker also reads `results/*/summary.json`, the E001–E003/E008–E010 rows of `results/reporting/project_synthesis/matched-comparisons.csv` and the records it names. Earlier figures: `docs/figures/project_synthesis/` via `scripts/reporting/README.md`.

## Code
- Build and host tests: `bash scripts/build.sh && ctest --test-dir build --output-on-failure` (CUDA targets need `nvcc`; set `CUDA_ARCH=120` for the RTX 5060 Ti).
- GPU gate before measuring: `bash scripts/check_inblock.sh`. Kernel benchmark: `build/w4a16_bench --help`.
- Serial/OpenMP baseline (E041): `OMP_NUM_THREADS=p build/cpu_bench M N K 128 1 10`; ARC batch script: `scripts/e041/submit.sh full | ssh <arc> sbatch`.
- Every experiment's design, commands, failures and raw results: `docs/EXPERIMENTS.md` and `results/eXXX_*`; per-experiment scripts in `scripts/eXXX/`.

## Third-party code
Marlin and vLLM (Apache-2.0) are used unmodified except the one-barrier fix in E006 (patch in `results/e006_20261106/`). GPTQ, AWQ (llm-compressor, Apache-2.0), llama.cpp (MIT) and ONNX Runtime GenAI (MIT) are external comparators.
