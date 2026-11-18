#!/usr/bin/env python3
"""E046 analyzer: per-case counter table from the ncu raw/source CSV exports.

analyze.py RUN_DIR  -> writes RUN_DIR/../analysis.txt and summary.json next to the run directory.
Uses the second (timed) GEMM launch of each case; the first is a repeatability check. Setup kernels
(dequant_w4a16_kernel) are excluded by name. Durations are kernel-replay values, not production latency.
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1]); ncu = run / "ncu"
M = {
    "dur_us": "gpu__time_duration.sum", "dram_rd": "dram__bytes_op_read.sum",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed", "l2_bytes": "lts__t_bytes.sum",
    "l2_tex_rd_sectors": "lts__t_sectors_srcunit_tex_op_read.sum", "l2_hit": "lts__t_sector_hit_rate.pct",
    "ld_req": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "ld_sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "bytes_per_sector_pct": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "occ_ach": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theo": "sm__maximum_warps_per_active_cycle_pct", "regs": "launch__registers_per_thread",
    "smem_static": "launch__shared_mem_per_block_static", "grid": "launch__grid_size",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "hmma_pct": "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "lsu_pct": "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "alu_pct": "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "fma_pct": "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "inst": "smsp__inst_executed.sum",
}

def num(x):
    try: return float(x.replace(",", ""))
    except (ValueError, AttributeError): return None

def raw_rows(f):
    r = list(csv.reader(open(f)))
    return [dict(zip(r[0], row)) for row in r[2:] if "dequant_w4a16_kernel" not in row[r[0].index("Kernel Name")]]

def source_blocks(f):
    """Per-kernel blocks of the SASS source page: list of (kernel name, header, rows)."""
    blocks, cur = [], None
    for row in csv.reader(open(f)):
        if row and row[0] == "Kernel Name": cur = [row[1], None, []]; blocks.append(cur)
        elif cur is not None and cur[1] is None: cur[1] = row
        elif cur is not None and row: cur[2].append(dict(zip(cur[1], row)))
    return [b for b in blocks if "dequant_w4a16_kernel" not in b[0]]

def opcode(src):
    t = src.split()
    t = t[1:] if t and t[0].startswith("@") else t
    return t[0] if t else "?"

def source_summary(rows):
    stalls = Counter(); ops = Counter(); ldg = defaultdict(lambda: Counter())
    for d in rows:
        op = opcode(d["Source"]); ex = num(d["Instructions Executed"]) or 0
        ops[op.split(".")[0]] += ex
        for k, v in d.items():
            if k.startswith("stall_") and "(Not Issued)" not in k: stalls[k[6:]] += num(v) or 0
        if op.startswith("LDG"):
            g = ldg[op]
            g["count"] += 1; g["exec"] += ex; g["samples"] += num(d["# Samples"]) or 0
            g["long_sb"] += 0  # LDG stalls are charged to consumers; kept for completeness
            g["l2_sec"] += num(d["L2 Theoretical Sectors Global"]) or 0
            g["l2_ideal"] += num(d["L2 Theoretical Sectors Global Ideal"]) or 0
            g["l2_excess"] += num(d["L2 Theoretical Sectors Global Excessive"]) or 0
            g["req"] += num(d["L1 Tag Requests Global"]) or 0
    tot = sum(stalls.values()) or 1
    return ({k: round(100 * v / tot, 1) for k, v in stalls.most_common() if v}, dict(ops), {k: dict(v) for k, v in ldg.items()})

summary = {}
for log in sorted(ncu.glob("*.log")):
    case = log.stem; rows = raw_rows(ncu / f"{case}.raw.csv")
    rows = [r for r in rows if r["Kernel Name"] == rows[-1]["Kernel Name"]]  # inblock_m4_vec also runs its bitwise parent
    assert len(rows) == 2, (case, len(rows))
    assert rows[0]["Kernel Name"] == rows[1]["Kernel Name"], case
    bench = list(csv.DictReader(open(ncu / f"{case}.bench.csv")))
    assert all(b["pass"] in ("1", "yes", "true") for b in bench) if bench and "pass" in bench[0] else True, case
    d = rows[1]; c = {k: num(d.get(v)) for k, v in M.items()}
    c["kernel"] = d["Kernel Name"][:70]; c["dur_first_us"] = num(rows[0][M["dur_us"]])
    for k in ("dur_us", "dur_first_us"): c[k] /= 1e3
    c["l2_to_dram"] = c["l2_bytes"] / c["dram_rd"] if c["dram_rd"] else None
    c["sectors_per_req"] = c["ld_sec"] / c["ld_req"] if c["ld_req"] else None
    c["dram_gbps"] = c["dram_rd"] / (c["dur_us"] * 1e-6) / 1e9
    blocks = source_blocks(ncu / f"{case}.source.csv")
    assert blocks[-1][0].split("(")[0].split("<")[0].split("::")[-1] in d["Kernel Name"], case
    c["stalls_pct"], ops, c["ldg"] = source_summary(blocks[-1][2])
    c["ops"] = {k: int(ops.get(k, 0)) for k in ("LDG", "HMMA", "HFMA2", "HADD2", "HMUL2", "LOP3", "PRMT", "LDS", "STS", "IMAD", "FFMA")}
    summary[case] = c

out = []
hdr = "case|dur us (1st)|DRAM rd MB|DRAM %pk|DRAM GB/s|L2 MB|L2/DRAM|L2 hit%|occ ach/theo %|regs|smem B|issue %|HMMA pipe %|LSU %|ALU %|FMA %|sect/req|B/sect %"
out.append(hdr); out.append("|".join("---" for _ in hdr.split("|")))
for k, c in summary.items():
    f = lambda v, p=1: "n/a" if v is None else f"{v:.{p}f}"
    out.append("|".join([k, f"{f(c['dur_us'])} ({f(c['dur_first_us'])})", f(c['dram_rd']/1e6, 2), f(c['dram_pct']), f(c['dram_gbps'], 0),
                         f(c['l2_bytes']/1e6, 1), f(c['l2_to_dram'], 2), f(c['l2_hit']), f"{f(c['occ_ach'])}/{f(c['occ_theo'])}",
                         f(c['regs'], 0), f(c['smem_static'], 0), f(c['issue_pct']), f(c['hmma_pct']), f(c['lsu_pct']), f(c['alu_pct']),
                         f(c['fma_pct']), f(c['sectors_per_req'], 2), f(c['bytes_per_sector_pct'])]))
out.append("")
for k, c in summary.items():
    top = ", ".join(f"{s} {v}%" for s, v in list(c["stalls_pct"].items())[:6])
    out.append(f"{k}: stalls (share of sampled warp states) {top}")
    out.append(f"    ops executed (warp-level) {c['ops']}")
    for op, g in c["ldg"].items():
        out.append(f"    {op}: {g['count']:.0f} sites, {g['exec']:.0f} exec, req {g['req']:.0f}, L2 sectors {g['l2_sec']:.0f} (ideal {g['l2_ideal']:.0f}, excess {g['l2_excess']:.0f})")
(run.parent / "analysis.txt").write_text("\n".join(out) + "\n")
json.dump(summary, open(run.parent / "summary.json", "w"), indent=1)
print("\n".join(out))
