#!/usr/bin/env python3
"""E044 analyzer: analyze.py RESULTS_DIR -> per-node cells, decision, summary.json.

RESULTS_DIR holds one full-<host>-<job>/ directory per node (main-s{1,2}.csv, held-s{1,2}.csv).
"""
import csv, glob, json, os, sys

ROOF = 448.0  # GB/s, RTX 5060 Ti specification (modelled bytes, not measured DRAM)
PAIRS = {"tc": "inblock_m4_vec", "tc_cvt": "inblock_m4_vec"}
PRIMARY = {("tc", 8), ("tc", 32), ("tc", 128)}
KERNELS = {"inblock_m4_vec", "tc_cvt", "tc", "cublas_dq"}


def load(path, expect):
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == expect, f"{path}: {len(rows)} rows, expected {expect}"
    assert all(r["pass"] == "1" for r in rows), f"{path}: a row failed FP64"
    assert {r["kernel"] for r in rows} == KERNELS
    return {(r["kernel"], int(r["M"]), int(r["K"]), int(r["N"]), r["flush_l2"]): r for r in rows}


def cells(rows, series, split):
    out = []
    for (k, M, K, N, fl), r in rows.items():
        if k not in PAIRS:
            continue
        p = rows[(PAIRS[k], M, K, N, fl)]
        c, q = float(r["median_us"]), float(p["median_us"])
        cb = float(rows[("cublas_dq", M, K, N, fl)]["median_us"])
        out.append(dict(split=split, series=series, kernel=k, parent=PAIRS[k], M=M, K=K, N=N,
                        policy="flush" if fl == "1" else "noflush", cand_us=c, parent_us=q,
                        gain_pct=100 * (1 - c / q), roof_frac=float(r["eff_gbps"]) / ROOF,
                        parent_roof_frac=float(p["eff_gbps"]) / ROOF, vs_cublas_dq=c / cb,
                        primary=split == "main" and (k, M) in PRIMARY))
    return out


def main(root):
    nodes = {}
    for d in sorted(glob.glob(os.path.join(root, "full-*"))):
        host = os.path.basename(d).split("-")[1]
        try:
            cs = []
            for s in (1, 2):
                cs += cells(load(f"{d}/main-s{s}.csv", 56), s, "main")
                cs += cells(load(f"{d}/held-s{s}.csv", 48), s, "held")
            bit = open(f"{d}/inner.log").read()
            assert "INNER-OK" in bit, "inner run incomplete"
            nodes[host] = dict(dir=os.path.basename(d), cells=cs,
                               primary_pass=all(c["gain_pct"] >= 5 for c in cs if c["primary"]))
            assert sum(c["primary"] for c in cs) == 12
        except (AssertionError, FileNotFoundError) as e:
            nodes[host] = dict(dir=os.path.basename(d), error=str(e))
    ok = [n for n in nodes.values() if "error" not in n]
    decision = "accepted" if ok and len(ok) == len(nodes) and all(n["primary_pass"] for n in ok) else "rejected"
    json.dump(dict(decision=decision, nodes=nodes), open(os.path.join(root, "summary.json"), "w"), indent=1)
    for h, n in nodes.items():
        if "error" in n:
            print(h, "ERROR", n["error"]); continue
        print(f"== {h} primary_pass={n['primary_pass']}")
        for c in sorted(n["cells"], key=lambda c: (c["split"] != "main", c["kernel"], c["K"], c["N"], c["M"], c["policy"], c["series"])):
            print(f"{'*' if c['primary'] else ' '} {c['split']:4} s{c['series']} {c['kernel']:14} M{c['M']:<3} {c['K']}x{c['N']:<5} {c['policy']:7} "
                  f"{c['parent_us']:8.2f} -> {c['cand_us']:8.2f} us  gain {c['gain_pct']:6.2f}%  roof {c['parent_roof_frac']:.2f}->{c['roof_frac']:.2f}  x cublas_dq {c['vs_cublas_dq']:.2f}")
    print("decision:", decision)


if __name__ == "__main__":
    main(sys.argv[1])
