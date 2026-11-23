# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for the roofline series (docs/OURS.md). Runs on the Mac, no GPU needed.

    uv run scripts/ours/plot.py results/ours_<timestamp>/sweep.csv

Writes next to the CSV:
  fig_pct_floor_m1.png  how close each kernel gets to its own bandwidth floor at M=1
  fig_latency_vs_m.png  median latency vs M, one panel per shape

pct_floor uses each kernel's own byte model: cuBLAS FP16 reads 4x the weight
bytes, so it can sit near 100% of *its* floor while still being ~4x slower.
The latency figure is the apples-to-apples comparison.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["naive", "inblock_specialized", "v0", "v1_runtime_tail", "v1_interleaved", "v1", "v2", "v3", "unfused", "cublas"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pct_floor" not in rows[0]:
        sys.exit(f"{path}: no pct_floor column (run the sweep with --peak-gbps / sweep.sh)")
    return rows


def ordered(kernels):
    return sorted(kernels, key=lambda k: (ORDER.index(k) if k in ORDER else len(ORDER), k))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    rows = load(path)
    shapes = list(dict.fromkeys((int(r["K"]), int(r["N"])) for r in rows))
    kernels = ordered({r["kernel"] for r in rows})
    peak = float(rows[0]["peak_gbps"])

    # Figure 1: % of floor at M=1, grouped by shape.
    m1 = {(int(r["K"]), int(r["N"]), r["kernel"]): float(r["pct_floor"]) for r in rows if r["M"] == "1"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(kernels)
    for i, k in enumerate(kernels):
        xs = [j + i * width for j in range(len(shapes))]
        ys = [m1.get((K, N, k), 0.0) for K, N in shapes]
        ax.bar(xs, ys, width, label=k)
    ax.axhline(100, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(shapes))])
    ax.set_xticklabels([f"K={K}\nN={N}" for K, N in shapes])
    ax.set_ylabel(f"% of bandwidth floor (ceiling {peak:.0f} GB/s)")
    ax.set_title("M=1: how close each kernel gets to reading its bytes at full speed")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    out1 = path.with_name("fig_pct_floor_m1.png")
    fig.savefig(out1, dpi=150)

    # Figure 2: latency vs M per shape.
    series = defaultdict(list)
    for r in rows:
        series[(int(r["K"]), int(r["N"]), r["kernel"])].append((int(r["M"]), float(r["median_us"])))
    fig, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 3.6), squeeze=False)
    for ax, (K, N) in zip(axes[0], shapes):
        for k in kernels:
            pts = sorted(series.get((K, N, k), []))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=k)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("M (tokens per multiply)")
        ax.set_title(f"K={K} N={N}")
    axes[0][0].set_ylabel("median latency (µs)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out2 = path.with_name("fig_latency_vs_m.png")
    fig.savefig(out2, dpi=150)
    print(f"wrote {out1}\nwrote {out2}")


if __name__ == "__main__":
    main()
