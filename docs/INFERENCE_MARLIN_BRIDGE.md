# E017: corrected Marlin in full-model inference

**The integration stops at the full-model numerical gate.** All declared operator checks pass, but the first corrected-Marlin request exceeds the unchanged elementwise logit bound at 591 of 9,723,904 values. All 64 greedy tokens match the reconstructed-FP16 reference. No primary or held-out timing is released, so E017 establishes no inference speedup. Setup inventories show lower resident tensor storage, reported separately below.

## 1. Comparison and coverage

E017 connects the previously separate Marlin GEMM and FP16 inference paths. R reconstructs selected quantized weights into FP16 and executes PyTorch linear GEMM. Q consumes the identical codes and scales through the unchanged corrected E006 Marlin A library, SHA256 `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`. O retains the original FP16 weights to describe quantization-induced changes. All three explicitly add FP16 bias after GEMM.

The pinned model/tokenizer is `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`, with PyTorch 2.14.0+cu130 and Transformers 4.57.1. Batch size is 1, declared contexts are 128/2048, and each request generates 64 fixed greedy tokens: eager prefill supplies token 1, followed by 63 cached forwards. Efficient SDPA, FP16 KV storage, static cache capacity=context+64, masks, prompts and EOS policy remain common. No kernel, tile, dispatcher, compiler, graph or attention optimization is introduced.

All seven projections in each of 28 transformer blocks are replaced: 196 linear modules. Original weights are transposed from N×K to contiguous K×N before project quantization.

| Selected projections | Count | K | N | Bias |
|---|---:|---:|---:|---|
| q and o | 56 | 1536 | 1536 | q only |
| k and v | 56 | 1536 | 256 | Both |
| gate and up | 56 | 1536 | 8960 | None |
| down | 28 | 8960 | 1536 | None |

Embedding lookup, the tied FP16 output head, RMS norms, RoPE, attention, cache handling, masks and generation remain outside Marlin. Source inspection and actual operator traces establish the existing small-N fallback and two-launch split at M2048/N8960. However, the complete-model coverage trace was scheduled after numerical validation and was never reached. Module replacement and library identity alone do not establish traced execution of every model call. Full-model coverage remains unverified.

Marlin's compressed Tensor Core implementation is upstream work. Our contributions are the synchronization correction, integration, validation and controlled experimental method. No measured result attributes a speedup to the barrier itself.

## 2. Quantization and operator validation

Each selected weight is quantized once under the existing G128 contract: FP16 maxabs/7 scales, FP32 division, nearest-even integer rounding, clipping to −7…7, offset code 8, project packing and FP16 reconstructed products. Both paths consume the same 196 hashed artifacts. Lossless Marlin code/scale permutation and reconstructed-weight bits are checked. R uses PyTorch-selected FP16 GEMM with inherited precision flags; it is not E009's explicit cuBLAS handle or the independent FP64 oracle.

CPU checks match the unchanged native quantizer on four complete cases. Review catches and fixes a signed-zero reconstruction mismatch before GPU integration. The representative actual layer is `model.layers.0.mlp.down_proj`, using captured original-model prefill/decode activations. Its 33 saved FP64 comparisons pass, covering 30,689,280 outputs. Maximum absolute error is 0.0021983644 and relative L2 is at most 0.0002164351. Acceptance uses the unchanged elementwise magnitude/RMS rule, not these maxima alone.

The fresh c7 prerequisite also passes repeated module calls, packing/reconstruction, actual corrected-Marlin traces and eight representative sanitizers. The subsequent integration stage passes all 12 ordinary distinct-shape cases and 32 sanitizer invocations: unfiltered memcheck/synccheck at M1/M128/M2048 and corrected-kernel-filtered racecheck at M1/M128 for each shape family. These are operator checks, not whole-model sanitizer coverage. Retained project CUDA build, CTest and GPU checks pass separately.

## 3. Exact full-model failure

O and R each complete primary/held-out/primary/held-out requests at context128 with deterministic repeated outputs, unchanged weights and accepted cache/reset/position checks. The first Q primary request then fails:

| Q versus R, context128 | Result |
|---|---:|
| Full logits checked | 9,723,904 |
| Nonfinite values | 0 |
| Values outside `abs(delta)<=0.03+0.01*abs(R)` | 591 |
| Maximum absolute difference | 0.0546875 |
| Relative L2, required ≤0.003 | 0.0022366869 |
| Greedy tokens | All 64 equal |

Candidate/reference tensors, exact failed indices, diagnostics, output tokens and weight checks are preserved before the assertion. Independent CPU recomputation reproduces the hashes, failure indices and numerical statistics. Failures occur at 12 generated positions, first at position13 when counting token1 as position1; 443 occur at position19. Position-level details are retained in the independent calculation file.

This request also passes recorded feedback, all28 cache lengths, reset/tail/finite/storage checks and guarded workspace checks. Q repeatability and alternating-prompt behavior are untested because validation stops immediately. All context2048 model cases and separate complete-model traces remain unrun. Passing tokens and aggregate L2 do not override the required elementwise failure. The evidence does not identify a faulty layer or isolate a cause; numerical propagation is an unresolved hypothesis.

O/R quantization differences are descriptive. Primary greedy tokens match despite changed logits; held-out tokens differ at 13 positions, first at generated position47. Later free-running logit differences include changed conditioning. Two passages are not a model-quality evaluation.

## 4. Storage and setup observations

Separate R/Q processes remove selected reconstructed-FP16 GPU copies from Q. Unique model/operator storage, including tied aliases and Q's shared workspace, is:

| Setup inventory | R | Q |
|---|---:|---:|
| Model/operator tensor bytes | 3,087,428,864 | 1,144,967,232 |
| Model/operator MiB | 2944.40 | 1091.93 |
| Additional context128 runner/cache bytes | 24,955,960 | 24,955,960 |

Q's model/operator tensor storage is **62.92% lower**. This is a directly checked setup inventory, not an accepted warmed-memory result or total device/host footprint. Allocator snapshots and initialization peaks are preserved separately. Both paths first load the original FP16 checkpoint before replacement; E017 does not demonstrate compressed-checkpoint startup or fitting a larger model during loading.

Artifact preparation reports 40.337s, including hashing, storage and reuse of the already-quantized prerequisite down projection. Summed layer quantization/hash checks report 11.157s and repacking 0.148s; these are instrumented setup costs, not pure isolated quantizer/kernel benchmarks. R/Q replacement takes 13.764/4.395s and total model initialization 20.566/10.187s. Additional runner construction takes approximately4.70/1.72ms. These single-process observations are not a controlled startup-speed comparison.

The predeclared latency criterion was ≥5% lower Q decode median in both reversed series, using fresh R denominators, three warmups and twenty requests per cell. Neither that campaign nor its conditional held-out campaign runs. No TTFT, decode or complete-request performance claim follows from validation timestamps.

## 5. Evidence and stopping decision

The target is c7, RTX5060Ti16GB/sm120, UUID `GPU-29e9569a-8a7a-cd23-5443-bd0b72ca241a`, driver610.57.04, one prun rank and16 host threads. Prerequisite332886 completes0; complete-model332899 exits1 at the numerical gate. Both are historical. All1328 retrieved paths pass their embedded hash manifest. Independent tensor/artifact, completed-gate, stopped-evidence and corruption checks are retained alongside the failure. No timing-release sentinel is created.

Earlier submission-size, home-cache quota and sanitizer-argument failures remain separate. During slow retrieval the failed job's exit-wait shell is paused solely to retain its owned allocation, then resumed after archive verification; its original failure status remains intact. No GPU work is resumed and no other allocation is changed.

[Experiment record](EXPERIMENTS.md#e017--corrected-marlin-in-the-pinned-full-model), [tooling](../scripts/e017/README.md), [complete failed-stage evidence](../results/e017_20261109/model-arc/), [independent calculations](../results/e017_20261109/independent-model-failure.json) and [accounting](../results/e017_20261109/final-accounting.txt) preserve the comparison. Stop here. The result rejects acceptance under this numerical contract; it does not establish global optimality, general Marlin incompatibility or absence of inference headroom.
