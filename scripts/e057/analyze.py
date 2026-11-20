#!/usr/bin/env python3
"""E057 analyzer: python3 analyze.py RUN_DIR --mode prereq|full > analysis.json 2> analysis.txt

Measurements are parsed from the run directory; everything derived here (per-call deltas, per-multiply-add
rates, clocks, fractions of specification values) is a CALCULATION from those measurements.
Exit status 1 if a completeness/correctness criterion fails (list in "failures")."""
import json, os, re, statistics, sys
from datetime import datetime, timedelta, timezone

S = sys.argv[1]
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "full"
FULL = MODE == "full"
K = N = 4096
MS = [1, 4, 128] if FULL else [4]
PS = [1, 2, 4, 8, 12, 16] if FULL else [1, 16]
SPS = PS
SPEC_BW = 448.0          # GB/s: 128-bit GDDR7 at 28 Gb/s (RTX 5060 Ti specification)
SPEC_TF = 36 * 2.572e9 * 512 / 1e12   # 47.41 TFLOP/s nominal used on the slides
E041 = {  # job 342501, c58: median ms and output hash at p = 1 / 16
    1: (26.565784, 2.024104, "fc781097bdda75a0"),
    4: (31.286131, 2.535952, "4035c1298a59e5bf"),
    128: (241.244821, 19.654114, "df892185f2df8f2a")}
fail = []
out = {"run": os.path.basename(os.path.normpath(S)), "mode": MODE}
txt = []


def say(*a):
    txt.append(" ".join(str(x) for x in a))


def hi(M, p):
    if not FULL:
        return 4
    return 2 + (5 * p if M == 128 else 20 * p)


def bench_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        f = line.strip().split(",")
        if len(f) == 15 and f[0].isdigit():
            rows.append(dict(threads=int(f[0]), M=int(f[1]), reps=int(f[6]), median_ms=float(f[7]),
                             fp64_pass=int(f[10]), rel_l2=float(f[12]), hash=f[14]))
    return rows


def check_bench(rows, where):
    for r in rows:
        if r["fp64_pass"] != 1:
            fail.append("FP64 check failed: %s %s" % (where, r))
        if r["hash"] != E041[r["M"]][2]:
            fail.append("output hash differs from E041: %s %s" % (where, r))


def perf_csv(path):
    vals = {}
    if not os.path.exists(path):
        fail.append("missing " + path)
        return vals
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.rstrip("\n").split(",")
        if len(f) < 3:
            continue
        ev = f[2].replace(":u", "")
        try:
            vals[ev] = float(f[0])
        except ValueError:
            vals[ev] = f[0]          # "<not supported>" / "<not counted>"
    return vals


# ---------------- A4: E041 re-time ----------------
a4 = bench_rows(os.path.join(S, "a4-retime.csv"))
check_bench(a4, "A4")
if len(a4) != 2 * len(MS):
    fail.append("A4 rows %d != %d" % (len(a4), 2 * len(MS)))
out["A4"] = []
say("A4 re-time (median ms, 10 reps in full) vs E041 c58 job 342501:")
for r in a4:
    ref = E041[r["M"]][0 if r["threads"] == 1 else 1]
    d = dict(M=r["M"], threads=r["threads"], median_ms=r["median_ms"], e041_ms=ref,
             ratio_to_e041=r["median_ms"] / ref, hash_equal_e041=r["hash"] == E041[r["M"]][2])
    out["A4"].append(d)
    say("  M%-3d p%-2d %9.3f ms  E041 %9.3f  ratio %.3f" % (r["M"], r["threads"], r["median_ms"], ref, d["ratio_to_e041"]))

# ---------------- perf ----------------
perf_ok = open(os.path.join(S, "perf", "perf_ok.txt")).read().strip() == "1"
out["perf_available"] = perf_ok
if not perf_ok:
    say("perf UNAVAILABLE:", open(os.path.join(S, "perf", "UNAVAILABLE.txt")).read().strip()[:600])

GROUPS = {1: ["cycles", "instructions", "task-clock"], 2: ["L1-dcache-loads", "L1-dcache-load-misses"],
          3: ["cache-references", "cache-misses"], 4: ["LLC-loads", "LLC-load-misses"],
          5: ["stalled-cycles-frontend", "stalled-cycles-backend"]}


def delta(tag_lo, tag_hi, ev, calls):
    lo, hi_ = perf_csv(os.path.join(S, "perf", tag_lo + ".csv")), perf_csv(os.path.join(S, "perf", tag_hi + ".csv"))
    a, b = lo.get(ev), hi_.get(ev)
    if isinstance(a, float) and isinstance(b, float):
        return (b - a) / calls
    return "unsupported: %s / %s" % (a, b)


def sel_bench(tag):
    rows = bench_rows(os.path.join(S, "perf", tag + ".out"))
    check_bench(rows, tag)
    return rows


if perf_ok:
    out["A1"] = {}
    say("\nA1 perf stat, T1, per GEMM call = (counts at R=hi - counts at R=2) / (hi - 2), user mode, mean of 3 runs each:")
    for M in MS:
        H = hi(M, 1)
        d = {"calls_hi": H, "calls_lo": 2, "macs_per_call": M * N * K}
        for g, evs in GROUPS.items():
            for ev in evs:
                d[ev] = delta("a1-M%d-p1-R2-g%d" % (M, g), "a1-M%d-p1-R%d-g%d" % (M, H, g), ev, H - 2)
            sel_bench("a1-M%d-p1-R2-g%d" % (M, g)); sel_bench("a1-M%d-p1-R%d-g%d" % (M, H, g))
        for ev in ["cycles", "instructions", "task-clock"]:
            if not isinstance(d[ev], float):
                fail.append("A1 M%d %s not measured: %s" % (M, ev, d[ev]))
        macs = float(M * N * K)
        per = {}
        for ev in [e for evs in GROUPS.values() for e in evs if e != "task-clock"]:
            if isinstance(d[ev], float):
                per[ev] = d[ev] / macs
        d["per_mac"] = per
        if isinstance(d["cycles"], float) and isinstance(d["instructions"], float):
            d["IPC"] = d["instructions"] / d["cycles"]
            d["clock_GHz"] = d["cycles"] / (d["task-clock"] * 1e6)
            d["cycles_per_weight"] = d["cycles"] / (N * K)
        for a, b, name in [("L1-dcache-load-misses", "L1-dcache-loads", "L1_miss_ratio"),
                           ("cache-misses", "cache-references", "cache_miss_ratio"),
                           ("LLC-load-misses", "LLC-loads", "LLC_miss_ratio")]:
            if isinstance(d.get(a), float) and isinstance(d.get(b), float) and d[b] > 0:
                d[name] = d[a] / d[b]
        out["A1"][M] = d
        say("  M%-3d task-clock/call %.3f ms, cycles %.4g, instr %.4g, IPC %s, clock %s GHz" % (
            M, d["task-clock"] if isinstance(d["task-clock"], float) else float("nan"),
            d["cycles"] if isinstance(d["cycles"], float) else float("nan"),
            d["instructions"] if isinstance(d["instructions"], float) else float("nan"),
            "%.3f" % d["IPC"] if "IPC" in d else "-", "%.3f" % d["clock_GHz"] if "clock_GHz" in d else "-"))
        say("       per multiply-add: " + ", ".join("%s %.4g" % (k, v) for k, v in per.items()))
        say("       ratios: " + ", ".join("%s %.4f" % (k, d[k]) for k in ["L1_miss_ratio", "cache_miss_ratio", "LLC_miss_ratio"] if k in d)
            + "; unsupported: " + ", ".join(k for k in d if isinstance(d[k], str)))

    # hotspots (-g twin, perf record cycles:u)
    out["hotspot"] = {}
    sym_re = re.compile(r"^\s+([\d.]+)%\s+(\S+)\s+\[\.\]\s+(.*\S)\s*$")

    def group_line(src):
        m = re.match(r"cpu_gemm\.cpp:(\d+)", src)
        if m:
            ln = int(m.group(1))
            if ln == 41: return "FMA loop c[n] += av*w[n] (line 41)"
            if 37 <= ln <= 40: return "activation convert / loop control (lines 37-40)"
            if 29 <= ln <= 33: return "4-bit unpack + LUT gather into tile (lines 29-33)"
            if 24 <= ln <= 28: return "per-group LUT build (lines 24-28, non-inlined part)"
            return "GEMM other lines (init/store/loops)"
        if src.startswith("fp16_host.h"):
            return "FP16<->FP32 software conversion (fp16_host.h, inlined at lines 26/28/39/47)"
        return "GEMM, other source (" + src.split(":")[0] + ")"

    for M in MS:
        h = {"top_symbols": [], "gemm_groups": {}}
        p = os.path.join(S, "perf", "rec-M%d-symbol.txt" % M)
        if not os.path.exists(p):
            fail.append("missing " + p); continue
        for line in open(p):
            m = sym_re.match(line)
            if m:
                h["top_symbols"].append((float(m.group(1)), m.group(3)))
        h["top_symbols"] = h["top_symbols"][:10]
        gemm_total = 0.0
        p2 = os.path.join(S, "perf", "rec-M%d-symsrc.txt" % M)
        if os.path.exists(p2):
            for line in open(p2):
                f = [x.strip() for x in line.split("|")]
                if len(f) >= 3 and f[0].endswith("%") and "cpu_w4a16_gemm" in f[1]:
                    v = float(f[0].rstrip("%")); gemm_total += v
                    k = group_line(f[-1]); h["gemm_groups"][k] = h["gemm_groups"].get(k, 0.0) + v
        h["gemm_share_of_samples"] = gemm_total
        h["gemm_groups_share_of_gemm"] = {k: v / gemm_total for k, v in h["gemm_groups"].items()} if gemm_total else {}
        out["hotspot"][M] = h
        say("\nhotspot M%d (cycles:u samples, -g twin, whole process incl. setup/check):" % M)
        for v, s in h["top_symbols"][:6]:
            say("   %6.2f%%  %s" % (v, s[:110]))
        say("   GEMM function %.2f%% of samples; within it:" % gemm_total)
        for k, v in sorted(h["gemm_groups_share_of_gemm"].items(), key=lambda kv: -kv[1]):
            say("     %5.1f%%  %s" % (100 * v, k))

    # A2 clocks
    out["A2"] = {}
    say("\nA2 average clock while computing = delta cycles:u / delta task-clock (GHz), and effective CPUs = task-clock / wall:")
    for M in MS:
        row = {}
        for p in PS:
            H = hi(M, p)
            c = delta("a2-M%d-p%d-R2" % (M, p), "a2-M%d-p%d-R%d" % (M, p, H), "cycles", H - 2)
            t = delta("a2-M%d-p%d-R2" % (M, p), "a2-M%d-p%d-R%d" % (M, p, H), "task-clock", H - 2)
            ins = delta("a2-M%d-p%d-R2" % (M, p), "a2-M%d-p%d-R%d" % (M, p, H), "instructions", H - 2)
            b = sel_bench("a2-M%d-p%d-R%d" % (M, p, H))
            wall = statistics.median([r["median_ms"] for r in b]) if b else None
            if not (isinstance(c, float) and isinstance(t, float)):
                fail.append("A2 M%d p%d not measured" % (M, p)); continue
            row[p] = dict(clock_GHz=c / (t * 1e6), task_clock_ms_per_call=t, cycles_per_call=c,
                          instructions_per_call=ins, wall_median_ms=wall,
                          effective_cpus=(t / wall) if wall else None)
        out["A2"][M] = row
        say("  M%-3d " % M + "  ".join("p%d %.3f GHz (cpus %.1f)" % (p, v["clock_GHz"], v["effective_cpus"] or 0) for p, v in row.items()))

# ---------------- A3 STREAM ----------------
out["A3"] = {}
say("\nA3 STREAM (midterm source, 64M doubles = 512 MB per array; per run best of 10; median over runs), GB/s:")
for p in SPS:
    res = {"Copy": [], "Scale": [], "Add": [], "Triad": []}
    reps = 5 if FULL else 1
    for r in range(1, reps + 1):
        f = os.path.join(S, "stream", "t%d-r%d.txt" % (p, r))
        if not os.path.exists(f):
            fail.append("missing " + f); continue
        got = 0
        for line in open(f):
            m = re.match(r"^(Copy|Scale|Add|Triad)\s+([\d.]+) GB/s", line)
            if m:
                res[m.group(1)].append(float(m.group(2))); got += 1
        if got != 4:
            fail.append("STREAM output incomplete " + f)
    out["A3"][p] = {k: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in res.items() if v}
    say("  p%-2d " % p + "  ".join("%s %.1f [%.1f-%.1f]" % (k, v["median"], v["min"], v["max"]) for k, v in out["A3"][p].items()))

# ---------------- B GPU ----------------
B = {"copy": None, "read": [], "mma": [], "mma_confirm": None, "phases": {}}
p = os.path.join(S, "b-gpu.txt")
for line in open(p):
    line = line.strip()
    if line.startswith("PHASE"):
        _, name, what, ms = line.split()
        B["phases"].setdefault(name, {})[what] = int(ms)
        continue
    f = line.split(",")
    kind = f[0]
    kv = {}
    for x in f[1:]:
        if "=" in x:
            k, v = x.split("=", 1)
            try:
                kv[k] = float(v)
            except ValueError:
                kv[k] = v
    if kind == "device":
        B["device"] = line
    elif kind == "copy":
        B["copy"] = kv
    elif kind == "read":
        B["read"].append(kv)
    elif kind == "mma_tile":
        B["mma_tile"] = kv
    elif kind == "mma":
        B["mma"].append(kv)
    elif kind == "mma_confirm":
        B["mma_confirm"] = kv
    elif kind == "failures":
        B["failures"] = int(f[1])
if B.get("failures") != 0:
    fail.append("GPU microbench failures=%s" % B.get("failures"))
if not B["copy"] or B["copy"]["mismatches"] != 0:
    fail.append("copy check")
if len(B["read"]) != 12 or any(r["sum_ok"] != 1 for r in B["read"]):
    fail.append("read rows/sums")
if B.get("mma_tile", {}).get("pass") != 1:
    fail.append("mma tile check")
if len(B["mma"]) != 32 or any(r["bad_outputs"] != 0 for r in B["mma"]) or not B["mma_confirm"] or B["mma_confirm"]["bad_outputs"] != 0:
    fail.append("mma rows/outputs")
hm = open(os.path.join(S, "sass-hmma.txt")).read().split()
B["sass_hmma"] = " ".join(hm)
if not any(x.startswith("HMMA.16816.F32") for x in hm):
    fail.append("no HMMA.16816.F32 in SASS")

# clocks during phases (nvidia-smi -lms 100 samples), node-local timestamps with the recorded UTC offset
tz = open(os.path.join(S, "tz.txt")).read().strip()
off = timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * (1 if tz[0] == "+" else -1)
samples = []
for line in open(os.path.join(S, "b-clk-during.csv")):
    f = [x.strip() for x in line.split(",")]
    if len(f) < 6 or not re.match(r"\d{4}/", f[0]):
        continue
    t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone(off))
    num = lambda s: float(s.split()[0]) if s and s.split()[0].replace(".", "").isdigit() else None
    samples.append((t.timestamp() * 1000, f[1], num(f[2]), num(f[3]), num(f[4]), num(f[5])))
B["clock_samples"] = len(samples)
for name, ph in B["phases"].items():
    sel = [s for s in samples if ph.get("begin", 0) <= s[0] <= ph.get("end", 0)]
    if sel:
        sm = [s[2] for s in sel if s[2] is not None]; mem = [s[3] for s in sel if s[3] is not None]
        pw = [s[5] for s in sel if s[5] is not None]
        ph["n_samples"] = len(sel)
        ph["sm_MHz"] = dict(median=statistics.median(sm), min=min(sm), max=max(sm)) if sm else None
        ph["mem_MHz"] = dict(median=statistics.median(mem), min=min(mem), max=max(mem)) if mem else None
        ph["power_W"] = dict(median=statistics.median(pw), max=max(pw)) if pw else None
        ph["pstates"] = sorted(set(s[1] for s in sel))

say("\nB GPU (%s)" % B.get("device", "?"))
if B["copy"]:
    c = B["copy"]
    B["copy_pct_spec"] = 100 * c["GBps"] / SPEC_BW
    say("  D2D cudaMemcpy %d B (read+write counted): median %.3f ms -> %.1f GB/s = %.1f%% of %.0f GB/s spec" % (
        c["bytes_per_copy"], c["median_ms"], c["GBps"], B["copy_pct_spec"], SPEC_BW))
if B["read"]:
    br = max(B["read"], key=lambda r: r["GBps"])
    B["read_best"] = br
    B["read_best_pct_spec"] = 100 * br["GBps"] / SPEC_BW
    say("  read-only kernel, best of %d configs (max of medians): %.1f GB/s = %.1f%% of spec (unroll %d, %d blocks/SM); range %.1f-%.1f" % (
        len(B["read"]), br["GBps"], B["read_best_pct_spec"], br["unroll"], br["blocks_per_sm"],
        min(r["GBps"] for r in B["read"]), max(r["GBps"] for r in B["read"])))
if B["mma"]:
    bm = max(B["mma"], key=lambda r: r["TFLOPs"])
    B["mma_best"] = bm
    say("  mma.m16n8k16 f32.f16.f16.f32 best of %d configs: %.2f TFLOP/s = %.1f%% of %.1f nominal (nacc %d, %d warps/block, %d blocks/SM)" % (
        len(B["mma"]), bm["TFLOPs"], 100 * bm["TFLOPs"] / SPEC_TF, SPEC_TF, bm["nacc"], bm["warps_per_block"], bm["blocks_per_sm"]))
    if B["mma_confirm"]:
        cf = B["mma_confirm"]
        B["mma_confirm_pct_nominal"] = 100 * cf["TFLOPs"] / SPEC_TF
        say("  confirmation run (%d launches): %.2f TFLOP/s = %.1f%% of nominal" % (cf["reps"], cf["TFLOPs"], B["mma_confirm_pct_nominal"]))
        ph = B["phases"].get("mma_confirm", {})
        if ph.get("sm_MHz"):
            clk = ph["sm_MHz"]["median"] * 1e6
            B["mma_confirm_flop_per_clk_per_sm"] = cf["TFLOPs"] * 1e12 / (36 * clk)
            B["nominal_at_observed_clock_TF"] = 36 * clk * 512 / 1e12
            say("  SM clock during confirmation: median %.0f MHz (min %.0f, max %.0f, %d samples) -> %.0f FLOP/clk/SM; 512 FLOP/clk/SM at that clock = %.2f TFLOP/s" % (
                ph["sm_MHz"]["median"], ph["sm_MHz"]["min"], ph["sm_MHz"]["max"], ph["n_samples"],
                B["mma_confirm_flop_per_clk_per_sm"], B["nominal_at_observed_clock_TF"]))
for name, ph in B["phases"].items():
    if ph.get("sm_MHz"):
        say("  clocks during %-11s SM median %.0f [%.0f-%.0f] MHz, mem median %.0f MHz, power median %s W, pstate %s, n=%d" % (
            name, ph["sm_MHz"]["median"], ph["sm_MHz"]["min"], ph["sm_MHz"]["max"],
            ph["mem_MHz"]["median"] if ph.get("mem_MHz") else float("nan"),
            ph["power_W"]["median"] if ph.get("power_W") else "?", ",".join(ph["pstates"]), ph["n_samples"]))
    else:
        say("  clocks during %-11s no samples in window" % name)
out["B"] = B
say("  SASS HMMA forms: " + B["sass_hmma"])

out["failures"] = fail
say("\nfailures: %d" % len(fail))
for f in fail:
    say("  " + f)
print(json.dumps(out, indent=1, default=str))
sys.stderr.write("\n".join(txt) + "\n")
sys.exit(1 if fail else 0)
