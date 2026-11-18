import argparse
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 10, "lines.linewidth": 1.8, "lines.markersize": 5})

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALPHA = 0.05
NTS = [40, 50, 60, 70]
DELTAS = [0.25, 0.5, 0.75, 1.0]
WORKERS = [4, 16]
COLORS = {"PSI-PT": "#D9A05B", "PSI-PT-oc": "#E57B54", "DS": "#88A78D", "Bonferroni": "#365D96",
          "Naive": "tab:gray", "PPL-SI": "#365D96"}
# (method of the result files, p-value field, legend label), in legend order
CURVES = [("si", "p_value", "PSI-PT"), ("si", "p_value_oc", "PSI-PT-oc"), ("split", "p_value", "DS"),
          ("si", "p_value_bonf", "Bonferroni"), ("si", "p_value_naive", "Naive")]
N0_LABEL = r"# target instances $n_0$"


def p_value(r, field, p):
    # Bonferroni: the naive p-value corrected for all 2^p models that could be selected
    if field == "p_value_bonf":
        return min(r["p_value_naive"] * float(2 ** p), 1.0)
    return r[field]


def pooled_p_values(pattern, field):
    # p-values of every result file matching results/pattern, pooled per (n0, delta); delta is None for FPR
    pooled = {}
    for path in glob.glob(os.path.join(RESULTS, pattern)):
        m = re.search(r"_delta([0-9.]+)_seed", os.path.basename(path))
        delta = float(m.group(1)) if m else None
        d = json.load(open(path))
        for nt, entry in d["results"].items():
            pooled.setdefault((int(nt), delta), []).extend(
                p_value(r, field, d["config"]["p"]) for r in entry["reps"]
                if r["p_value"] is not None and not r["invalid"])
    return pooled


def draw_rates(ax, pattern, curves, keys, x_of, name):
    # rejection rate at level ALPHA of every curve at the (n0, delta) keys
    for method, field, label in curves:
        pooled = pooled_p_values(pattern.format(method=method), field)
        keys_m = [k for k in keys if k in pooled]
        rates = [np.mean(np.array(pooled[k]) <= ALPHA) for k in keys_m]
        for k, rate in zip(keys_m, rates):
            n = len(pooled[k])
            print(f"{label:10s} n0={k[0]} delta={k[1]}  {name}={rate:.3f} +- {np.sqrt(rate * (1 - rate) / n):.3f}"
                  f"  ({n} p-values)")
        ax.plot([x_of(k) for k in keys_m], rates, "o-", color=COLORS[label], label=label, markersize=4)


def save(fig, name):
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out}.pdf and {out}.png")


def fpr():
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_rates(ax, "fpr/fpr_{method}_h10_ng_nt*_seed*.json", CURVES, [(nt, None) for nt in NTS],
               lambda k: k[0], "FPR")
    ax.axhline(ALPHA, color="darkgray", linestyle="--", linewidth=0.8, alpha=0.6, label=rf"$\alpha = {ALPHA}$")
    ax.set_xticks(NTS)
    ax.set_ylim(-0.02, 0.27)
    ax.set_xlabel(N0_LABEL)
    ax.set_ylabel("FPR")
    ax.legend(loc="upper left")
    save(fig, "fpr")


def tpr():
    # TPR against n0 (delta = 0.5) and against delta (n0 = 50); the naive test is not shown (no FPR control)
    pattern = "tpr/tpr_{method}_s10_h10_ng_nt*_delta*_seed*.json"
    for name, keys, x_of, xticks, xlabel in [
            ("tpr_nt", [(nt, 0.5) for nt in NTS], lambda k: k[0], NTS, N0_LABEL),
            ("tpr_delta_h10_nt50", [(50, d) for d in DELTAS], lambda k: k[1], DELTAS, r"signal strength $\delta$")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_rates(ax, pattern, CURVES[:4], keys, x_of, "TPR")
        ax.set_xticks(xticks)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TPR")
        ax.legend(loc="upper left")
        save(fig, name)


def timing():
    # median time per p-value of the z-segment (PPL-SI) and per-fit (PSI-PT) parallelism,
    # against the number of workers and against p
    for x, csv_name, name in [("n_jobs", "timing_workers.csv", "timing_workers_median"),
                              ("p", "timing_p.csv", "timing_p_median")]:
        times = {}
        for r in csv.DictReader(open(os.path.join(RESULTS, "timing", csv_name))):
            times.setdefault(r["method"], {}).setdefault(int(r[x]), []).append(float(r["seconds"]))
        xs = [v for v in sorted(times["fit"]) if x == "p" or v in WORKERS]
        fig, ax = plt.subplots(figsize=(6, 4))
        if x == "n_jobs":
            # workers drawn equally spaced; one worker (one process) is the common starting point
            seq = np.median(times["seq"][1])
            ticks, pos = [1] + xs, list(range(len(xs) + 1))
            print(f"{'no parallelism':14s} n_jobs=1  median time={seq:.1f}s")
        else:
            ticks, pos = xs, xs
        for m, label in [("seg", "PPL-SI"), ("fit", "PSI-PT")]:
            level = [np.median(times[m][v]) for v in xs]
            for v, s in zip(xs, level):
                print(f"{label:14s} {x}={v}  median time={s:.1f}s  ({len(times[m][v])} p-values)")
            ax.plot(pos, ([seq] if x == "n_jobs" else []) + level, "o-", color=COLORS[label], label=label)
        if x == "n_jobs":
            ax.plot([0], [seq], marker="o", linestyle="none", color="dimgray", markersize=7, zorder=3,
                    label="no parallelism (1 worker)")
        ax.set_xticks(pos)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_ylim(0, 1.1 * ax.get_ylim()[1])  # room above the top curve
        ax.set_xlabel("# workers" if x == "n_jobs" else r"# features $p$")
        ax.set_ylabel("median time per p-value (s)")
        ax.legend(loc="upper right" if x == "n_jobs" else "upper left")
        save(fig, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="figures of the paper from the result files in results/")
    parser.add_argument("figures", nargs="+", choices=["fpr", "tpr", "timing"])
    for figure in parser.parse_args().figures:
        {"fpr": fpr, "tpr": tpr, "timing": timing}[figure]()
