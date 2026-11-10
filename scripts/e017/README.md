# E017 corrected Marlin in full-model inference

This is one bounded integration of the preserved corrected E006 A library into the pinned Qwen2.5-1.5B model. Read the E017 predeclaration in [EXPERIMENTS](../../docs/EXPERIMENTS.md) before execution. No Marlin device source, tile, dispatcher, attention implementation, compiler or graph mechanism changes.

`integration.py` replaces the seven transformer-block projections in all 28 blocks. Q uses the exact corrected C ABI and compressed weights. R reconstructs the identical project G128 weights into FP16 and calls ordinary PyTorch linear GEMM. O retains original FP16 weights. All three explicitly add the original q/k/v bias after FP16 GEMM. Embeddings and the tied output head, norms, attention and KV caches remain common. The original checkpoint parameter count is labeled separately from actual resident parameter/tensor counts.

`quantize.py` implements the fixed project arithmetic on one CPU layer at a time. `quant_contract.cpp` only exposes the existing CPU quantizer for independent bit checks. It is not a CUDA kernel. `prerequisite.py` captures first-block real activations, validates one down projection, then prepares one shared quantized artifact set and checks the four distinct shape families. It records full FP64 output references before numerical assertions, repeated calls, output poisoning, guards and module bias behavior. Each sanitizer invocation has a distinct suffix. The initial down-projection sanitizers filter corrected Marlin. Full shape-family memcheck/synccheck are unfiltered. Full shape-family racecheck filters corrected Marlin and covers M1/M128, not complete M2048 or the whole model.

`infer.py` uses one eager static-cache loop for O/R/Q. Both validation and timing retain the full GPU logits slab and position/feedback history. CPU tensor transfers, parameter hashes and cache audits are outside the request timers. GPU cache-reset observations are inside preparation in every path. Initial model/replace storage and runner cache/buffer storage have separate unique-allocation inventories. Allocator measurements are separate and do not describe total device or host memory.

`trace_coverage.py` correlates actual runtime launches to GPU corrected-Marlin symbols within every selected projection marker. At 2048 tokens, gate/up prefill calls split into two existing launches. The expected candidate totals are 196/252 prefill kernels at128/2048 and1372 decode kernels for seven traced steps. These counts supplement numerical validation, not replace it.

ARC execution uses a fresh absolute BeeGFS root and sbatch/prun, one process and one GPU. `bootstrap_body.sh` inspects actual allocation/hardware, installs only the pinned isolated runtime if absent, verifies exact preserved binaries and runs retained project checks plus the real-layer prerequisite. Only after independent evidence inspection may ONE_LAYER_INSPECTED be recorded and an afterok-dependent `stage_arc.sh` be submitted on the verified node. The latter prepares weights and validates operators/full model, then pauses for FULL_VALIDATION_INSPECTED before timing. Never create these markers based on flags or expected behavior. Preserve every failure and stop its dependent branch.

Each timing phase has3 warmups/20 requests per fresh-process cell and two reversed series, with fresh R denominators. The5% decode gate in both series controls release of an identical held-out campaign. All samples and other metrics are retained. Original-versus-reconstructed free-running logits after token divergence also reflect changed conditioning. Two prompts are not a model-quality evaluation.

Local checks:

```bash
python3 -B scripts/e017/check_infer.py
python3 -B scripts/e017/check_trace.py
python3 -B scripts/e017/check_quantize.py  # requires pinned NumPy
c++ -std=c++17 -O2 -fPIC -shared -Iinclude scripts/e017/quant_contract.cpp src/quant.cpp -o /tmp/e017-quant-contract.so
python3 -B scripts/e017/check_contract.py /tmp/e017-quant-contract.so
python3 -B scripts/e017/analyze.py RESULTS --stage prerequisite
python3 -B scripts/e017/analyze.py RESULTS --stage validation
python3 -B scripts/e017/analyze.py RESULTS --stage primary
python3 -B scripts/e017/check_analysis.py RESULTS primary
```

Run analyzers against an evidence copy to preserve historical derived outputs. `audit_tensors.py` independently checks actual saved array payloads and numerical statistics. Source/binary/environment/artifact hashes must hold before releasing a later phase. No script is guaranteed on a future allocation. A pending job or local check is not GPU validation.
