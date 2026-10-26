#!/usr/bin/env python3
"""Render the README benchmark figure (res/img/bench_10m_m3_{light,dark}.png).

Usage: python3 benchmark/plot_readme_chart.py benchmark/bench_<ver>_<os>_<arch>_<date>.json [--version 0.7.3]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

CODECS = ["none", "snappy", "zstd", "lz4"]
LIBS = ["carquet", "arrow_cpp", "pyarrow"]
THEMES = {
    "light": dict(bg="#fcfcfb", grid="#e6e5df", text="#1a1a19", muted="#8a897f", tick="#5d5c55",
                  bars=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(bg="#1a1a19", grid="#33332f", text="#f0efec", muted="#8f8e85", tick="#c3c2b7",
                 bars=["#3987e5", "#d95926", "#199e70"]),
}


def label(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def render(data, version, theme, out):
    t = THEMES[theme]
    rows = {(r["library"], r["compression"]): r for r in data["results"] if r["rows"] == 10_000_000}
    names = [f"carquet {version}", f"Arrow C++ {data['arrow_cpp_version']}", f"PyArrow {data['pyarrow_version']}"]
    iters = data["config"]["bench_iterations"]["large"]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(10.5, 4.7), dpi=200, facecolor=t["bg"])
    fig.text(0.06, 0.925, f"Parquet at 10 million rows (INT64 + DOUBLE + INT32), {data['system']['cpu']}, {data['timestamp']}",
             fontsize=12, fontweight="bold", color=t["text"], va="baseline")
    fig.text(0.06, 0.87, f"Trimmed median of {iters} runs, page cache purged between write and read. "
             "Uncompressed reads use carquet's mmap zero-copy path.", fontsize=8.5, color=t["muted"], va="baseline")
    fig.legend(handles=[Patch(color=c) for c in t["bars"]], labels=names, loc="upper left",
               bbox_to_anchor=(0.055, 0.855), ncol=3, frameon=False, fontsize=9.5, labelcolor=t["text"],
               handlelength=1.3, handleheight=1.0, columnspacing=1.6, handletextpad=0.9)

    for i, (key, title, log) in enumerate([("write_ms", "Write time, ms (lower is better)", False),
                                           ("read_ms", "Read time, ms, log scale (lower is better)", True)]):
        ax = fig.add_axes([0.06 + i * 0.5105, 0.1207, 0.4185, 0.565], facecolor=t["bg"])
        top = 0
        for j, lib in enumerate(LIBS):
            vals = [rows[(lib, c)][key] for c in CODECS]
            top = max(top, *vals)
            xs = [k + (j - 1) * 0.244 for k in range(len(CODECS))]
            ax.bar(xs, vals, width=0.22, color=t["bars"][j], zorder=3)
            for x, v in zip(xs, vals):
                ax.annotate(label(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.5, color=t["tick"])
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.1, max(top * 1.6, 120))
            ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(0, top * 1.12)
            ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim(-0.55, len(CODECS) - 0.45)
        ax.set_xticks(range(len(CODECS)), CODECS)
        ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
        ax.tick_params(length=0, colors=t["tick"], labelsize=10, pad=5)
        ax.tick_params(axis="x", labelsize=10.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=t["text"], pad=12)

    fig.savefig(out, dpi=200, facecolor=t["bg"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json")
    ap.add_argument("--version", help="carquet version shown in the legend (default: from the JSON)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "res", "img"))
    args = ap.parse_args()
    data = json.load(open(args.json))
    for theme in THEMES:
        out = os.path.join(args.out_dir, f"bench_10m_m3_{theme}.png")
        render(data, args.version or data["carquet_version"], theme, out)
        print(out)


if __name__ == "__main__":
    main()
