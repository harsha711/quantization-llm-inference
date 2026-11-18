#!/usr/bin/env python3
"""E048 analyzer: analyze.py RESULTS_DIR -> per-node cells, decision, summary.json.

RESULTS_DIR holds one full-<host>-<job>/ directory per node (main-s{1,2}.csv/.log, held-s{1,2}.csv/.log).
"""
import csv, glob, json, os, re, sys

ROOF = 448.0  # GB/s, RTX 5060 Ti specification (modelled bytes, not measured DRAM)
CAND, PARENT = "tc_m32", "tc"
PRIMARY_M = {32, 64, 128}
KERNELS = {"tc", "tc_m32", "cublas_dq"}


def load(path, expect):
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == expect, f"{path}: {len(rows)} rows, expected {expect}"
    assert all(r["pass"] == "1" for r in rows), f"{path}: a row failed FP64"
    assert all(int(r["check_rows"]) >= int(r["M"]) for r in rows), f"{path}: not every row checked"
    assert {r["kernel"] for r in rows} == KERNELS
    return {(r["kernel"], int(r["M"]), int(r["K"]), int(r["N"]), r["flush_l2"]): r for r in rows}


def paths(log):
    """(M, K, N, flush) -> 1 if the bench ran the 32-row kernel (M >= 32), else 0; every output must equal tc."""
    out, key = {}, None
    for line in open(log):
        m = re.match(r"shape: M=(\d+) N=(\d+) K=(\d+) .* flush_l2=(\d)", line)
        if m:
            key = (int(m[1]), int(m[3]), int(m[2]), m[4])
        m = re.match(r"tc_m32 vs tc: (\d+) / (\d+) FP16 outputs differ", line)
        if m:
            assert m[1] == "0", f"{log}: tc_m32 differs from tc at {key}"
            out[key] = int(key[0] >= 32)
    return out


def cells(rows, split_of, series, split):
    out = []
    for (k, M, K, N, fl), r in rows.items():
        if k != CAND:
            continue
        p = rows[(PARENT, M, K, N, fl)]
        c, q = float(r["median_us"]), float(p["median_us"])
        cb = float(rows[("cublas_dq", M, K, N, fl)]["median_us"])
        out.append(dict(split=split, series=series, M=M, K=K, N=N, S=split_of[(M, K, N, fl)],
                        policy="flush" if fl == "1" else "noflush", cand_us=c, parent_us=q,
                        gain_pct=100 * (1 - c / q), roof_frac=float(r["eff_gbps"]) / ROOF,
                        parent_roof_frac=float(p["eff_gbps"]) / ROOF, cand_vs_cublas_dq=c / cb,
                        parent_vs_cublas_dq=q / cb,
                        primary=split == "main" and M in PRIMARY_M))
    return out


def main(root):
    nodes = {}
    for d in sorted(glob.glob(os.path.join(root, "full-*"))):
        host = os.path.basename(d).split("-")[1]
        try:
            cs = []
            for s in (1, 2):
                cs += cells(load(f"{d}/main-s{s}.csv", 42), paths(f"{d}/main-s{s}.log"), s, "main")
                cs += cells(load(f"{d}/held-s{s}.csv", 48), paths(f"{d}/held-s{s}.log"), s, "held")
            assert "INNER-OK" in open(f"{d}/inner.log").read(), "inner run incomplete"
            assert "CHECK-OK" in open(f"{d}/check-e048.log").read(), "E048 checks incomplete"
            assert sum(c["primary"] for c in cs) == 12
            nodes[host] = dict(dir=os.path.basename(d), cells=cs,
                               primary_pass=all(c["gain_pct"] >= 5 for c in cs if c["primary"]))
        except (AssertionError, FileNotFoundError, KeyError) as e:
            nodes[host] = dict(dir=os.path.basename(d), error=str(e))
    ok = [n for n in nodes.values() if "error" not in n]
    decision = "accepted" if ok and len(ok) == len(nodes) and all(n["primary_pass"] for n in ok) else "rejected"
    json.dump(dict(decision=decision, nodes=nodes), open(os.path.join(root, "summary.json"), "w"), indent=1)
    for h, n in nodes.items():
        if "error" in n:
            print(h, "ERROR", n["error"]); continue
        print(f"== {h} primary_pass={n['primary_pass']}")
        for c in sorted(n["cells"], key=lambda c: (c["split"] != "main", c["K"], c["N"], c["M"], c["policy"], c["series"])):
            print(f"{'*' if c['primary'] else ' '} {c['split']:4} s{c['series']} M{c['M']:<3} {c['K']}x{c['N']:<5} m32={c['S']} {c['policy']:7} "
                  f"{c['parent_us']:8.2f} -> {c['cand_us']:8.2f} us  gain {c['gain_pct']:6.2f}%  roof {c['parent_roof_frac']:.2f}->{c['roof_frac']:.2f}  "
                  f"x cublas_dq {c['parent_vs_cublas_dq']:.2f}->{c['cand_vs_cublas_dq']:.2f}")
    print("decision:", decision)


if __name__ == "__main__":
    main(sys.argv[1])
