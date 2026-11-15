"""Figures 2 and 3 from the result files written by emerging.py and synthetic_benchmark.py."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_pkl, out
from feae import stream

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def fig_emerging(scenarios=(("ToN", "scanning", 42, "ToN-IoT: unseen scanning attacks"),
                            ("DDoS", "DrDoS_NTP", 42, "CIC-DDoS2019: unseen DrDoS-NTP attacks"))):
    fig, ax = plt.subplots(4, 2, figsize=(7.2, 6.6), sharex="col")
    for j, (ds, fam, seed, title) in enumerate(scenarios):
        P = load_pkl(f"{ds}_novel_{fam}_{seed}")
        r = stream(P, seed=seed)
        tr = r["trace"]
        x = np.arange(len(tr)) + 1
        cpw = P["cp"] / 512
        al = [i + 1 for i, ch in enumerate(r["src"]) if ch != "."]
        H = np.array([np.nan if s is None else s[0] for _, _, s in tr])
        F = np.array([np.nan if s is None else s[1] for _, _, s in tr])
        wts = np.array([w for w, _, _ in tr])
        ax[0, j].set_title(title, fontsize=8, fontweight="bold", loc="left")
        for a, (yv, lab, col) in zip(ax[:2, j], [(H, "Entropy $H_t$", "#1f4e79"), (F, "PFD$_t$", "#2e7d32")]):
            a.plot(x, yv, color=col, lw=1.3, marker="o", ms=2.5)
            a.set_ylabel(lab)
            for w in al:
                a.axvline(w, color="#c62828", lw=1.2)
                a.axvspan(w + 0.5, w + 4.5, color="0.92", zorder=0)
            a.axvline(cpw, color="k", lw=1, ls="--")
        a = ax[2, j]
        for i, (lab, col) in enumerate([("LR", "#1f4e79"), ("CNB", "#e67e22"), ("ET", "#2e7d32")]):
            a.plot(x, wts[:, i], label=lab, color=col)
        a.set_ylabel("Weight $w_i$")
        a.axvline(cpw, color="k", lw=1, ls="--")
        if j == 1:
            a.legend(frameon=False, loc="center right", fontsize=7)
        a = ax[3, j]
        a.plot(x, 100 * np.array([acc for _, acc, _ in tr]), color="#6a1b9a", lw=1.2)
        a.set_ylabel("Window acc. (%)")
        a.set_xlabel("Window index (512 flows per window)")
        a.axvline(cpw, color="k", lw=1, ls="--")
        for k in (2, 3):
            for w in al:
                ax[k, j].axvline(w, color="#c62828", lw=1.2)
    ax[0, 1].plot([], [], color="k", ls="--", lw=1, label="attack onset")
    ax[0, 1].plot([], [], color="#c62828", lw=1.2, label="FE-AE alarm")
    ax[0, 1].legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out("fig_emerging.png"), dpi=200)


def fig_comparison():
    s = pd.read_csv(out("synthetic_summary.csv"))
    cov = s[s.stream.isin(["MeanShift", "ScaleShift", "ShapeShift", "GradualMean"])].groupby("det").det20.sum()
    names = ["KSWIN", "DDM", "HDDM_A", "ADWIN", "PageHinkley", "EDDM", "FE-AE, features only"]
    e = pd.read_csv(out("emerging_raw.csv")).groupby("var").acc.mean() * 100
    order = ["CNB alone", "LR alone", "ET alone", "Static ensemble", "Adaptive, beta=1", "FE-AE"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for a, vals, labs, yl in ((ax[0], [cov[n] for n in names], names, "Detected covariate changes (of 80)"),
                             (ax[1], [e[n] for n in order], order, "Accuracy after attack onset (%)")):
        cols = ["#9aa7b5"] * (len(vals) - 1) + ["#c62828"]
        b = a.bar(range(len(vals)), vals, color=cols)
        a.bar_label(b, fmt="%.4g", fontsize=7)
        a.set_xticks(range(len(vals)))
        a.set_xticklabels([l.replace(", features only", "\n(label-free)").replace(" alone", "").replace(" ensemble", "\nensemble").replace(", beta=1", "\n$\\beta$=1") for l in labs], fontsize=6.5)
        a.set_ylabel(yl)
    ax[1].set_ylim(70, 93)
    ax[0].set_title("(a) Drift detection, four covariate-drift streams", fontsize=8, loc="left")
    ax[1].set_title("(b) Emerging attacks, 15 scenarios on five datasets", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_comparison.png"), dpi=200)


if __name__ == "__main__":
    which = sys.argv[1:] or ["emerging", "comparison"]
    if "emerging" in which:
        fig_emerging()
    if "comparison" in which:
        fig_comparison()
