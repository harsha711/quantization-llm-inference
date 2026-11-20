#!/usr/bin/env python3
"""Tables for results/ours_tc_marlin_*: mean us per kernel, % of the 427.4 GB/s floor, and
time relative to Marlin (Marlin time / kernel time = fraction of Marlin's speed)."""
import json
import sys
from collections import defaultdict

PEAK = 427.4e9  # measured read ceiling, w4a16_bwprobe on c8 (docs/OURS.md)
rows = [json.loads(l) for l in open(sys.argv[1])]
policy = sys.argv[2] if len(sys.argv) > 2 else "read_flush"
stat = sys.argv[3] if len(sys.argv) > 3 else "mean_us"
t = defaultdict(dict)
for r in rows:
    if r["policy"] == policy:
        t[(r["K"], r["N"], r["M"])][r["kernel"]] = r[stat]

def floor_us(k, n, m, kern):
    w = 2 * k * n if kern == "cublas" else k * n // 2 + 2 * (k // 128) * n
    return (w + 2 * m * k + 2 * m * n) / PEAK * 1e6

ks = ["marlin", "tc_ladder", "tc", "v3", "v2", "v1", "cublas"]
print(f"policy={policy} stat={stat}")
shapes = sorted({(k, n) for k, n, _ in t}, key=lambda x: (-x[0] * x[1], x))
for k, n in shapes:
    print(f"\nK={k} N={n}   (floor at M=1: {floor_us(k, n, 1, 'marlin'):.1f} us)")
    print(f"{'M':>4} " + " ".join(f"{x:>9}" for x in ks) + " | " + " ".join(f"{x+'/Mar':>9}" for x in ("tc_ladder", "v3")) + " | %floor mar/tcl/v3")
    for m in sorted(m for kk, nn, m in t if (kk, nn) == (k, n)):
        d = t[(k, n, m)]
        cells = " ".join(f"{d[x]:9.1f}" if x in d else f"{'-':>9}" for x in ks)
        frac = " ".join(f"{d['marlin']/d[x]:9.2f}" if x in d and "marlin" in d else f"{'-':>9}" for x in ("tc_ladder", "v3"))
        pf = "/".join(f"{100*floor_us(k,n,m,x)/d[x]:.0f}" for x in ("marlin", "tc_ladder", "v3") if x in d)
        print(f"{m:>4} {cells} | {frac} | {pf}")
