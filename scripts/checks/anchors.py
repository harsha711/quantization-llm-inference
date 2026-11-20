"""Presentation anchor numbers from result files (2026-11-20).

Kernel timings (E050, E051) are recomputed from raw samples/CSVs through floorcheck.
Model-level anchors read each experiment's analyzer outputs (E052 medians, E054
summary.json, E055 ppl, E056 group totals), not raw per-request samples.

Rules fixed before running:
- Kernel timings go through floorcheck; only OK (cyclic) or OK_FLUSHED rows are used.
- Model comparisons use every E052 job that timed both arms at a context, all four
  cells (primary/heldout x series 1/2): pct_less = 100*(1 - a/b), pct_more = 100*(a/b - 1).
- Nothing here is a new measurement; every value names its source.

Usage: python3 scripts/checks/anchors.py      Tests: python3 -m unittest scripts/checks/test_anchors.py
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floorcheck as fc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "results")
CELLS = ("primary|s1", "primary|s2", "heldout|s1", "heldout|s2")
# explicit, so a later smoke/prereq analysis is never pooled silently
E052_JOBS = ("core-c10-343363", "long-c10-343364", "head-c10-343365", "ext-c10-343366", "ext-c5-343367")


def _j(p):
    with open(os.path.join(R, p)) as fh:
        return json.load(fh)


def problem():
    K, N = 4096, 11008
    w4 = fc.bytes_model(K, N, 1)
    f16 = fc.bytes_model(K, N, 1, kind="fp16")
    flop = 2 * K * N
    return dict(shape="4096x11008 M1", w4_bytes_m1=w4, fp16_bytes_m1=f16,
                w4_flop_per_byte_m1=flop / w4, fp16_flop_per_byte_m1=flop / f16,
                ridge_measured=fc.PEAK_FLOP_PER_S / fc.BW_BYTES_PER_S,
                source="calculation; roofs E057")


def _e052_medians():
    out = []
    for job in E052_JOBS:
        out.append((job, _j(f"e052_20261118/{job}/analysis.json")["medians"]))
    return out


def _cmp(a, b, metric, sign, jobs=None):
    res = {}
    for job, med in _e052_medians():
        ctxs = {int(k.split("|")[1]) for k in med}
        for ctx in ctxs:
            for c in CELLS:
                ka, kb = f"{a}|{ctx}|{c}", f"{b}|{ctx}|{c}"
                if ka in med and kb in med:
                    x, y = med[ka][metric], med[kb][metric]
                    v = 100 * (1 - x / y) if sign == "less" else 100 * (x / y - 1)
                    res.setdefault(ctx, []).append(v)
                    if jobs is not None and job not in jobs.setdefault(ctx, []):
                        jobs[ctx].append(job)
    return dict(sorted(res.items()))


def model():
    jobs = {}
    d = dict(gqa_less_time_than_gra_pct=_cmp("GQA", "GRA", "decode_ms", "less"),
             gta_less_time_than_gra_pct=_cmp("GTA", "GRA", "decode_ms", "less"),
             gta_less_time_than_gqa_pct=_cmp("GTA", "GQA", "decode_ms", "less", jobs),
             gta_more_prefill_than_gqa_pct=_cmp("GTA", "GQA", "prefill_ms", "more"),
             source="E052 analyzer medians; GTA vs GQA pools c10 and c5 (both RTX 5060 Ti), see jobs")
    d["jobs"] = dict(sorted(jobs.items()))
    return d


def cpu():
    t = {}
    with open(os.path.join(R, "e041_20261117/full-342501/strong.csv")) as fh:
        for r in csv.DictReader(fh):
            t[(int(r["M"]), int(r["threads"]))] = float(r["median_ms"])
    e = _j("e057_20261120/full-c8-344144/analysis.json")
    return dict(t1_ms={M: t[(M, 1)] for M in (1, 4, 128)},
                sp16={M: t[(M, 1)] / t[(M, 16)] for M in (1, 4, 128)},
                triad_gbps={int(p): v["Triad"]["median"] for p, v in e["A3"].items()},
                clock_ghz_p1=e["A2"]["1"]["1"]["clock_GHz"],
                clock_ghz_p16={int(M): e["A2"][M]["16"]["clock_GHz"] for M in e["A2"]},
                source="E041 strong.csv (c58); E057 analysis.json (c8, not c58)")


def climb():
    """E051 (a) flushed single 4096^2 matrix, series 1 and 2 (c8). NOT for CPU ratios:
    E041's CPU ran on c58; use processor_ratios() (E051 c, same node)."""
    rows = fc.e051_a_rows(os.path.join(R, "e051_20261118/full-c8-343361"))
    out = {}
    for src in ("a-main-s1.csv", "a-main-s2.csv"):
        for r in rows:
            if r["source"] == src and r["flushed"] and r["K"] == r["N"] == 4096 and r["M"] in (1, 128):
                if r["status"] != fc.OK_FLUSHED:
                    raise ValueError(f"climb row failed floor check: {r}")
                d = out.setdefault(r["M"], {}).setdefault(r["config"], dict(us=[], status=r["status"]))
                d["us"].append(r["time_us"])
    return out


def processor_ratios():
    """E051 (c): c8 CPU T1 / flushed tc_ladder GPU time, same node. Processor ratios,
    never parallel speedups."""
    d = os.path.join(R, "e051_20261118/full-c8-343361")
    cpu_t1, gpu = {}, {}
    with open(os.path.join(d, "c-cpu.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["threads"] == "1":
                cpu_t1[int(r["M"])] = float(r["median_ms"]) * 1e3
    for r in fc.e051_c_rows(os.path.join(d, "analysis.txt")):
        if r["config"] == "tc_ladder" and r["flushed"]:
            if r["status"] != fc.OK_FLUSHED:
                raise ValueError(f"ratio row failed floor check: {r}")
            gpu[r["M"]] = r["time_us"]
    return {M: cpu_t1[M] / gpu[M] for M in sorted(gpu)}


def e050():
    rows = fc.e050_rows(os.path.join(R, "e050_20261118/full-c8-343308"))
    rect = [r for r in rows if (r["K"], r["N"]) != (4096, 4096)]
    t = {(r["K"], r["N"], r["M"], r["config"], r["series"]): r["time_us"] for r in rect}
    small, large, ladder = [], [], []
    for (K, N, M, cfg, s), v in t.items():
        if cfg != "tc_ladder":
            continue
        frac = t[(K, N, M, "marlin", s)] / v
        (small if M in (4, 16) else large).append(frac)
        if M in (64, 128):
            ladder.append(v / t[(K, N, M, "tc", s)])
    return dict(frac_small_m=small, frac_large_m=large, ladder_over_tc_large_m=ladder,
                statuses_used=[r["status"] for r in rect],
                source="E050 series*.jsonl, rectangles only (4096^2 invalid)")


def roofline():
    e = _j("e057_20261120/full-c8-344144/analysis.json")["B"]
    rows = fc.e050_rows(os.path.join(R, "e050_20261118/full-c8-343308"))
    pts = {}
    for r in rows:
        if (r["K"], r["N"], r["series"]) == (4096, 11008, 1) and r["config"] in ("tc_ladder", "marlin"):
            if r["status"] != fc.OK:
                raise ValueError(f"roofline row failed floor check: {r}")
            M, t = r["M"], r["time_us"]
            gbps = fc.bytes_model(4096, 11008, M) / t / 1e3
            tflops = 2 * M * 4096 * 11008 / t / 1e6
            ai = 2 * M * 4096 * 11008 / fc.bytes_model(4096, 11008, M)
            mem_bound = ai < fc.PEAK_FLOP_PER_S / fc.BW_BYTES_PER_S
            # only the roof a point sits under is an efficiency; the other is an off-roof coordinate
            pts[(r["config"], M)] = dict(us=t, gbps=gbps, tflops=tflops, flop_per_byte=ai,
                                         pct_bw=100 * gbps / (fc.BW_BYTES_PER_S / 1e9) if mem_bound else None,
                                         pct_peak=None if mem_bound else 100 * tflops / (fc.PEAK_FLOP_PER_S / 1e12))
    return dict(read_gbps=e["read_best"]["GBps"], copy_gbps=e["copy"]["GBps"],
                mma_tflops=e["mma_confirm"]["TFLOPs"], points=pts,
                source="E057 B; E050 series 1, 4096x11008")


def e054():
    d = _j("e054_20261119/summary.json")
    dec = {g: [x for row in v["rows"] for x in row["decode_pct_ours_below_marlin"]] for g, v in d.items()}
    rows = [row for v in d.values() for row in v["rows"]]
    holds = sum(1 for row in rows if row["R2"])
    pre2048 = {g: [x for row in v["rows"] if row["context"] == 2048 for x in row["prefill_pct_ours_below_marlin"]]
               for g, v in d.items()}
    return dict(decode_less_time_pct=dec, n_gpu_models=len(d),
                n_nodes=len({n for v in d.values() for n in v["nodes"]}),
                direction_holds=(holds, len(rows)),
                prefill_2048_more_time_pct={g: [-x for x in v] for g, v in pre2048.items()},
                source="E054 summary.json")


def quality():
    q = _j("e055_20261119/quality-c8-344064/analysis.json")
    p = {k: v["perplexity"] for k, v in q["variants"].items()}
    c = q["comparisons"]["QT_vs_QM"]
    return dict(ppl=p, blocks_cost_pct=100 * (p["R"] / p["O"] - 1),
                qt_vs_qm_pct=c["ppl_pct"], top1_agreement=c["top1_agreement"],
                source="E055 analysis.json (c8)")


def standing():
    more = _cmp("GTA", "V", "decode_ms", "more")
    d = _j("e056_20261119/full-c10-344130/analysis.json")["contexts"]
    e56 = {}
    for ctx, v in d.items():
        T, V = v["arms"]["GTA"], v["arms"]["V"]
        gap = T["total_us"] - V["total_us"]
        e56[int(ctx)] = dict(GTA_us=T["total_us"], V_us=V["total_us"],
                             gemm_share_pct=100 * (T["gemm"]["us"] - V["gemm"]["us"]) / gap,
                             other_share_pct=100 * (T["other"]["us"] - V["other"]["us"]) / gap,
                             attention_share_pct=100 * (T["attention"]["us"] - V["attention"]["us"]) / gap,
                             other_extra_launches=T["other"]["calls"] - V["other"]["calls"],
                             total_extra_launches=T["launches"] - V["launches"])
    return dict(more_time_than_vllm_pct=more, e056=e56,
                source="E052 ext medians; E056 analysis.json arms (profiler device time)")


def compute_all():
    return dict(problem=problem(), model=model(), cpu=cpu(), climb=climb(),
                processor_ratios=processor_ratios(), e050=e050(),
                roofline=roofline(), e054=e054(), quality=quality(), standing=standing())


def _keys(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _keys(v)
                for k, v in o.items()}
    if isinstance(o, list):
        return [_keys(v) for v in o]
    return o


if __name__ == "__main__":
    print(json.dumps(_keys(compute_all()), indent=1))
