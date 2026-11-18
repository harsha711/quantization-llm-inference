The draft needs revision for the new scope. Its strongest current story is still Marlin integration and model experiments; the authors’ kernel ladder is compressed into three paragraphs, and its comparison with Marlin remains pending.

The checker passes (`report numbers ok: 32`), but I found **two incorrect numeric ranges that it misses**. I changed no files. Report line numbers below refer to the unchanged `main.tex` reviewed.

**1. Claims to fix**

1. **Whole-model execution is presented as necessarily bandwidth-bound.**  
   [Lines 20 and 26](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:20>): “so it is limited by the bytes of weights read” and “so the step is limited by reading weights from DRAM.”  
   Small-\(M\) GEMM intensity does not establish the bottleneck of an entire decode step. E022 finds host-bound projection calls; E025 identifies substantial attention work. **Fix:** describe this as the motivation for compressed, weight-streaming GEMM, conditional on sufficient parallelism and low overhead. Say storage falls approximately fourfold; actual DRAM traffic need not.

2. **The abstract understates the tested AWQ cost.**  
   [Line 20](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:20>): “still 5.2–5.9% with upstream GPTQ or AWQ.”  
   E040’s three AWQ arms cost **5.72%, 5.86%, and 10.21%**. **Fix:** say “5.15% with upstream GPTQ; 5.72–10.21% across tested AWQ recipes,” or explicitly identify the selected recipes.

3. **There is no single correctness check or quantization recipe covering every result.**  
   [Lines 34 and 37](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:34>): “a fixed tolerance used for every result”; “Every result … uses the same 4-bit format and the same correctness check.”  
   Model validation uses distinct gates; perplexity is a quality measurement; llama.cpp/ORT use other formats; E039 includes a 16-level grid; asymmetric AWQ is outside the runtime format. **Fix:** scope the unchanged FP64 tolerance to project GEMM validation, then distinguish model acceptance, quality evaluation, and external formats.

4. **The weight-selection formula is overgeneralized and misses the zero-scale case.**  
   [Lines 41–42](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:41>): “each group … gets … \(\max|w|/7\)” and “each weight is divided by its scale.”  
   This describes plain rounding, not calibrated GPTQ/AWQ. Division is undefined when the stored scale is zero; the implementation sets the signed code to zero. **Fix:** label this the plain-rounding recipe and state `s=0 ⇒ q=8`.

5. **FP64 row coverage is incorrectly stated as universal.**  
   [Line 50](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:50>): “All rows are checked up to \(M=64\), and 64 sampled rows above that.”  
   This applies to the original benchmark and E044, whereas external cyclic experiments check all active rows, including M128. **Fix:** give coverage by harness. Preserve the distinction between sampled FP64 checks and full-output parent bitwise comparisons.

6. **The rung count is inconsistent.**  
   [Line 34](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:34>): “four rungs of our own CUDA kernels.”  
   The report labels three CUDA rungs, grouping E001–E003 together; counting individual optimization steps gives five. **Fix:** use consistent names throughout: CPU baseline, naive CUDA, E001, E002, E003, E043, E044, cyclic comparison, subsequent profile-led changes.

7. **Fast unpacking is not one instruction, and the documented kernel requirements are incomplete.**  
   [Line 59](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:59>): “Two weights are unpacked per instruction.”  
   `unpack2` performs masking, rearrangement, subtraction, and multiplication. **Fix:** “two weights at a time using packed FP16 operations.”  
   [Line 97](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:97>) lists divisibility requirements but omits 16-byte A alignment and the E044 review’s 4-byte C alignment precondition. State these if presenting the supported interface.

8. **The race-fix claim needs the recorded boundary and provenance.**  
   [Lines 20, 30, 61, and 147](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:61>): “found and fixed a shared-memory race”; “the corrected kernel passes … tests.”  
   E006 supports a synchronization correction removing the reported hazard in the tested matrix. It does not establish general validation or exact official-wheel source mapping. **Fix:** identify the pinned vLLM source commit `4ba787ec9959e566d3b587a796dff07673324da5`, distinguish wheel reproduction from the patched local build, and say “removes the reported hazard in the tested configurations.” Retain the existing disclosure that no incorrect original output was established.

9. **The global cuBLAS configuration contradicts E044’s qualification.**  
   [Line 64](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:64>): “cuBLAS with FP32 reduction (math mode 16).”  
   E044 used **default math mode**; line 97 correctly says so. **Fix:** separate original-benchmark/E044 cuBLAS settings from E009/E010/E045’s explicit mode 16. The E044 qualification is recorded in [EXPERIMENTS.md](</home/harsha/Documents/ncsu/sem 3/548/project/proj/docs/EXPERIMENTS.md:97>).

10. **The setup paragraph invents a common measurement and acceptance protocol.**  
    [Line 64](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:64>): “10 warm-ups and 100 timed iterations”; “two passages and two series”; “at least 5% faster in every cell of both series; anything less is a tie.”  
    Exceptions include E004’s 20-sample screen, E001/E002’s single final series, primary-only E020/E023 timings, CPU repetitions, and diagnostics. E001’s criterion was speedup \(>1\); E002/E003 used speedup \(\ge1.05\), which differs from 5% lower latency. Descriptive experiments have no adoption gate. **Fix:** replace this paragraph with a compact protocol table. Also distinguish `sbatch` submission, historical `salloc` allocations, and `prun` launching.

11. **CPU scaling and its cause are stated too strongly.**  
    [Line 87](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:87>): “Scaling is near-ideal to 8 threads”; “The dip at 12 threads is load imbalance, by calculation.”  
    M128 reaches **6.58× at eight threads**, versus 7.79× for M1. The block calculation establishes a 10.67× upper bound at twelve threads, not a complete causal explanation of measured time. **Fix:** qualify near-linear scaling to M1/M4 and say static imbalance imposes that bound; other losses remain unattributed.

12. **Odd-row regressions are overgeneralized.**  
    [Line 93](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:93>): “row reuse slows \(M=1\) and odd row counts.”  
    At M5, two-row reuse improves both policies by about 1.20×; four-row reuse improves store-flush timing and regresses only without flushing. M1 two-row store-flush is inconclusive. **Fix:** name those cases rather than all odd row counts. See [E003](</home/harsha/Documents/ncsu/sem 3/548/project/proj/docs/EXPERIMENTS.md:2271>).

13. **Static instruction sites are mislabeled as dynamic instructions per word.**  
    [Line 95](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:95>): “global load instructions per word fall from 10 to 7 … and from 34 to 6.”  
    The script counts **static LDG sites per kernel function**; it does not calculate executions per packed word. **Fix:** use that unit and retain the existing uncertainty about load-issue causality. The counting command is explicit in [inner.sh:15](</home/harsha/Documents/ncsu/sem 3/548/project/proj/scripts/e043/inner.sh:15>).

14. **The application store-flush policy is described as proven eviction.**  
    [Lines 95 and 97](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:95>): “with L2 flushed”; also line 64’s “an L2 flush.”  
    AGENTS explicitly says store flushing does not prove eviction. **Fix:** consistently say “with the application store-flush policy,” distinguishing it from Nsight’s profiler cache invalidation.

15. **The fast-unpack speedup ranges are wrong.**  
    [Line 97](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:97>): “1.21–1.42× at \(M\le8\) and 1.05–1.12× above.”  
    Recomputed over the five nodes, two series, both policies, and main square cases: **1.10–1.44× at M≤8; 1.00–1.12× above M8**. Concrete counterexamples:
    - c58, M8, no-flush: \(22.080/20.000=1.104×\), in [raw CSV](</home/harsha/Documents/ncsu/sem 3/548/project/proj/results/e044_20261117/full-c58-343107/main-s1.csv:23>).
    - c10, M16, store-flush: both variants take 39.232 µs, so **1.000×**, in [raw CSV](</home/harsha/Documents/ncsu/sem 3/548/project/proj/results/e044_20261117/full-c10-343106/main-s1.csv:27>).  
    **Fix:** report policy-specific ranges or the complete ranges above. The prose experiment summary also needs correction later; its existing range is not authoritative over the CSVs.

16. **The four-matrix paragraph quotes two/four-matrix pooled ranges.**  
    [Line 108](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:108>): “four independent weight matrices … 3.60–4.40× … 7.96–9.33×.”  
    E009’s four-matrix-only ranges are **3.62–4.40×** against reconstructed cuBLAS and **8.05–9.33×** against unfused execution. The lower quoted bounds come from two-matrix cases. **Fix:** use those four-matrix ranges, or explicitly say “two/four matrices.”

17. **Cycling beyond L2 capacity is treated as proof of model-like eviction.**  
    [Line 108](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:108>): “where each layer’s weights are evicted before reuse.”  
    E008–E010 explicitly do not establish eviction or actual per-layer inference behavior. **Fix:** “a synthetic workload with longer weight reuse distance; its active weight set exceeds L2 capacity.”

18. **Roofline coordinates are promoted into a proven bottleneck and zero remaining benefit.**  
    [Line 110](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:110>): “explains both regimes”; “so compression no longer helps.”  
    [Line 151](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:151>): “shrinking to nothing once the product becomes compute-bound.”  
    The transpose retains **15.2–16.8% lower latency at M128**. A logical-byte roofline supports a model-based interpretation, not unique critical-path attribution. **Fix:** say the advantage narrows consistently with the model; preserve both orientation-specific outcomes and the unresolved cause.

19. **Two roofline comparisons need precise mathematical meaning.**  
    [Line 110](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:110>): “the speedup is the intensity ratio … times their efficiencies”; “Marlin at \(M=4\) performs like FP16 at \(M=16\).”  
    **Fix:** write \(S=(I_Q/I_R)(\eta_Q/\eta_R)\) for memory-roof-normalized efficiencies. Specify that the second comparison concerns **achieved FLOP/s**, not latency or equal work. The measured throughput ranges are approximately 5.89–6.09 versus 5.83–6.69 TFLOP/s.

20. **E008 is given a causal conclusion it explicitly avoided.**  
    [Line 110](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:110>): “cache reuse, not the kernel, explained it”; [line 145](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:145>): “A 13% kernel gain disappeared.”  
    E008 establishes reuse dependence. The stronger rectangle retains **1.26–2.39%** gains with four matrices; the transpose regresses **0.43–0.76%**. Weight, output, and workspace reuse effects were not isolated. **Fix:** “the ≥5% advantage did not survive cycling; its precise cache mechanism remains unresolved.”

21. **E003 reduces repeated work; it does not remove the measured traffic.**  
    [Line 113](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:113>): “1.15 GB of L2 traffic … which E003’s two/four-row reuse removes.”  
    L2 traffic becomes approximately **641/384 MB**, reductions of **44.3/66.7%**. DRAM reads remain around 9.7–9.9 MB. **Fix:** report those before/after counters, identify the separate cache-invalidated profiling regime, and avoid attributing all L2 traffic to weight unpacking.

22. **The rounding diagnosis needs its request boundary.**  
    [Line 121](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:121>): “every individual Marlin call passed … so the difference is propagated FP16 rounding.”  
    E018 supports that interpretation for **one context128 primary request**, not every model run. **Fix:** state the 196×64 same-input-call coverage and that bounded interpretation. Keep E017/E019 failed under their original gates.

23. **The whole-step host explanation exceeds E022’s measurement.**  
    [Line 123](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:123>): “because the 196 small launches were 99% host time”; similarly lines 32 and 146.  
    E022 measures a random-input projection sequence, excluding attention, normalization, and cache work. **Fix:** call the whole-step explanation a hypothesis supported at short context by E023; do not use it to explain the unexplained long-context FP16 penalty.

24. **Model disclosures and partial optimization outcomes are incomplete.**  
    [Lines 20, 32, 123, 144, and 151](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:123>) quote model benefits without consistently carrying their required boundaries. Line 121 contains much of the third-gate history, but only scopes it to results *below*; it does not cover abstract/introduction claims.  
    **Fix:** preferably remove numeric model gains from the abstract/introduction under the new scope. In retained background, attach the third-gate disclosure, four known/six fresh cases, missing sanitizer coverage, and plain-rounded blocks’ **+13.1% perplexity**. Also state:
    - E023 passes versus eager Marlin at context128, **misses at 2048**.
    - E026 is **not a robust pass at context128**, below the bar in two of eight cells.
    - E024’s 28-cell robustness concerns graph-Marlin versus FP16, not the incremental attention change.

25. **The external comparison omits material boundaries, and the lesson excludes GEMM incorrectly.**  
    [Line 125](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:125>) correctly says stack against stack, but should add that llama.cpp/ORT quantize the embedding/output matrix and their quality remains unmeasured. State that production vLLM uses unpatched Marlin; identical weights do not mean identical device code or timing boundaries.  
    [Line 146](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:146>): “the remaining gap … is fusion and launch count, not the GEMM.”  
    E030 measures about **378–386 µs of the gap inside Marlin projections**, with per-call causes unisolated. **Fix:** say fewer projection calls and fused operations accompany lower device time; they do not completely isolate the cause.

26. **The calibrated-quality paragraph merges incompatible scopes.**  
    [Line 133](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:133>): “All timed results above use plain round-to-nearest weights”; “runs at the same speed”; “ours matches upstream.”  
    External stacks use their own quantizers. E038 establishes near-equal speed only for its named arms at contexts128/2048 on c10; it did **not** time a calibrated reconstructed-FP16 baseline. E039’s close implementation match is **AT16 versus upstream**, rather than the contract’s AT15 result.  
    **Fix:** scope each statement accordingly. Identify 16-level/asymmetric quality-only arms as outside the runtime contract.

27. **E037’s required selection disclosure is missing.**  
    [Line 133](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:133>) cites the selected +5.62% result without stating that ten variants were scored on test and three were added after seeing a test table.  
    **Fix:** add that disclosure and explicitly say selection used validation only. This also qualifies line 64’s blanket predeclaration statement.

28. **The conclusion associates old speedups with a pooled quality range.**  
    [Line 151](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:151>): “the benefit … costs 5–13% perplexity.”  
    Original model timings carry **+13.1%**. Pairing calibrated quality with measured speed is allowed only for E038’s arms and contexts. **Fix:** keep the original timing/quality pair and present calibrated quality plus E038’s limited timing confirmation separately.

29. **The experiment index changes outcomes.**  
    [Lines 160–169](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:160>):
    - “E004–E006 … Accepted” hides E005’s stopped safety gate.
    - “E005-R1/R2 … Rejected” is wrong: **R2 passed its rectangular confirmation gate**, with cache-policy tradeoffs.
    - “E022–E026 … Accepted” combines diagnostics with E023/E026’s partial passes.
    - “E031–E035 … Faster, changes outputs” combines quality-only experiments with timing; E035 retained all tested tokens and lacked a logit gate.
    - “E036–E040 … All >5% perplexity” must specify **block cost**; calibrated heads passed their quality criteria.  
    **Fix:** separate diagnostic, safety, latency, and quality decisions.

30. **Artifact availability is overstated.**  
    [Line 151](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:151>): “Code, raw results … accompany this report.”  
    The experiment record identifies ten oversized raw paths absent from a GitHub clone and additional BeeGFS-only archives/checkpoints. **Fix:** identify what accompanies the submission, what is hash-recorded elsewhere, and how a grader can obtain missing artifacts.

**2. Structure for the new ladder scope**

The abstract, introduction, results, lessons, and conclusion all need the same central question: **how far did the authors’ kernels close the small-\(M\) gap to corrected Marlin?**

| Current material | Concrete change |
|---|---|
| Abstract, lines 19–20 | Center the authors’ kernels, measured improvements, remaining gap, and roofline. Reduce model work to one motivation sentence. |
| Contributions, line 34 | Lead with kernel design, profile evidence, correctness, matched measurements, and rejected changes. Move model/quantizer work to background. |
| Design/results split, lines 52–113 | Give each optimization its own **baseline → evidence → hypothesis/criterion → one change → correctness → matched result → limitation** sequence. |
| E001–E003 compressed into “rung 1” | Split them. E001 has low-parallelism evidence; E002 has guard/code-generation evidence and 23.1% fewer dynamic instructions; E003 has the clearest profile-to-fix-to-speedup chain. |
| E043/E044 | Explicitly label these historical rungs as lacking pre-change Nsight profiling. Do not retrospectively call them all profile-led. E044’s measured effect belongs to the whole design, not MMA alone. |
| E045 placeholder, line 99 | Supply the actual cyclic results when available: shape/M, all three amortized times, fraction of Marlin speed, both series. Pending evidence cannot support reaching Marlin. |
| Profiling subsection, line 112 | Put evidence beside the rung it motivated. Its present position makes profiling look retrospective. |
| Model/quality, lines 115–140 | Move to a short background section or appendix. Remove the two quality figures from the main ladder narrative. |
| Marlin tuning failures | Retain briefly as comparator history. They are not rejected optimizations of the authors’ kernel. |
| Conclusion, line 151 | End on measured ladder progress and remaining Marlin gap. Preserve the explicit statement that these kernels are outside the model path. |

Add one central ladder figure/table with **absolute latency and matched parent ratios**. Do not multiply historical speedups across experiments or join single-matrix and cyclic timings into one series.

The roofline currently contains only **Marlin and cuBLAS**, so it explains the comparator rather than locating the authors’ kernels. Add their measured points with clearly separated benchmark regimes. State the finite byte formulas, useful operation count, excluded workspace traffic, ceiling source, and clock convention. Keep the full-machine Tensor ceiling distinct from an applicable scalar FP32 ceiling.

The new profile-led framing is prospective: E045 remains pending in the reviewed report, and E046 was being predeclared during this review. Show pending work honestly rather than filling its evidence chain with hypotheses.

**3. Risky numbers the checker does not cover**

The checker checks required substrings and selected summary assertions; it does not verify every occurrence, provenance, or qualification.

| Report lines | Uncovered or insufficiently covered numbers |
|---|---|
| 20, 93 | E001–E003’s 3.85×, 2.89×, 1.31–1.33×, 1.55–1.75×, and 1.3× regression. |
| 87 | 4.3 GB/s, 8.65 MB, 64 blocks, twelve-thread 10.7× bound, and 13–53× GPU/CPU ratios. |
| 95 | Static LDG counts and their incorrectly stated unit. |
| 97 | **Incorrect fast-unpack ranges**; exhaustive unpack coverage and alignment requirements. |
| 108 | **Incorrect four-matrix lower bounds**, unfused comparison, and M128 percentage ranges. |
| 105, 110 | 448 GB/s provenance, 75–94%/85–88%/82–99% roof fractions, larger-\(M\) intensities, throughput equivalence, and plotted performance coordinates. |
| 121–125 | Logit failure counts, gate-case disclosures, all model timings/gains, memory reduction, trace counts, and head gains. |
| 130, 133 | 307,050 tokens, E038’s 0.3%, AWQ’s 10.21%, and calibrated-speed applicability. |

I independently regenerated E043/E044 summary cells from their CSVs and checked all **92 timing-row medians** in E008–E010 against raw samples.

The roof arithmetic is internally consistent **conditional on its assumptions**:  
\(36\times2.572\,\mathrm{GHz}\times512=47.407104\) TFLOP/s; ridge \(=105.8194\) FLOP/byte. All 32 stored intensities match the finite byte model. However, the generator describes the compute ceiling as coming from a third-party listing, and `main.tex` supplies neither that provenance nor the nominal-boost assumption.

**4. LaTeX/source problems without compiling**

- [Lines 14 and 16](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:14>): author placeholders remain.
- [Line 99](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:99>): the bold E045 placeholder is unfinished submission content.
- [Lines 73–85](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:73>): six unconstrained `r` columns, long kernel names, and headers are likely to overflow one ACM column. Shorten labels or use a two-column table.
- [Lines 68–70](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:68>): the two-panel scaling PDF is about **425 pt wide**, then reduced to one roughly 241-pt column. Its nominal 8-pt text becomes approximately **4.5 pt**. Use `figure*` or stack the panels.
- [Lines 155–174](</home/harsha/Documents/ncsu/sem 3/548/project/proj/report/main.tex:155>): the appendix’s narrow first column contains long experiment identifiers; check overfull boxes. `[h]` also does not ensure placement immediately there.
- The bibliography contains `roofline`, but the report never cites it. The PTX fragment-layout source and hardware-ceiling specifications also need citations.
- I found **no missing citation keys, undefined/duplicate labels, missing figure files, or mismatched environments**. `acmart` already provides graphics support; an additional `graphicx` import is unnecessary.
- Verify the course’s **four-page minimum excluding references** after compilation; source inspection cannot establish compliance.

Not checked: LaTeX compilation, final pagination, GPU reruns, or complete historical analyzer campaigns.  
Risk: the passing number checker currently permits incorrect ranges, missing disclosures, and unsupported causal conclusions.
