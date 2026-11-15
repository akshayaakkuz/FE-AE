"""Emerging-attack streams: onset detection (Table 9) and accuracy after the onset (Table 10)."""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import C, DATASETS, acc_after, load_pkl, out, score_onset, single_preds
from feae import stream

VARIANTS = {"Static ensemble": dict(adapt=False), "Adaptive, beta=1": dict(trigger="none", beta=1),
            "Adaptive, no trigger": dict(trigger="none"), "FE-AE, features only": dict(use_err=False), "FE-AE": dict()}


def main(names):
    rows = []
    for name in names:
        for f in sorted(glob.glob(out(f"{name}_novel_*.pkl"))):
            key = os.path.basename(f)[:-4]
            fam, seed = key[len(name) + 7:].rsplit("_", 1)
            P = load_pkl(key)
            cp = P["cp"]
            for k, nm in (("pA", "LR alone"), ("pB", "CNB alone"), ("pC", "ET alone")):
                rows.append(dict(ds=name, fam=fam, seed=int(seed), var=nm, acc=acc_after(single_preds(P, k), P["yte"], cp)))
            for nm, o in VARIANTS.items():
                r = stream(P, seed=int(seed), **o)
                det, fa, delay = score_onset(r["src"], cp)
                rows.append(dict(ds=name, fam=fam, seed=int(seed), var=nm, acc=acc_after(r["preds"], P["yte"], cp),
                                 det=det, fa=fa, delay=delay))
    df = pd.DataFrame(rows)
    df.to_csv(out("emerging_raw.csv"), index=False)
    t9 = df[df["var"].isin(["FE-AE, features only", "FE-AE"])].groupby(["ds", "fam", "var"], sort=False).agg(
        detected=("det", "sum"), false_alarms=("fa", "sum"), delay=("delay", "mean")).round(0)
    print("\nTable 9: onset detection (3 seeds per scenario)\n", t9.to_string())
    for v in ("FE-AE, features only", "FE-AE"):
        g = df[df["var"] == v]
        print(v, "total: %d/%d detected, %d false alarms, mean delay %.0f" % (g.det.sum(), len(g), g.fa.sum(), g.delay.mean()))
    t10 = (100 * df.pivot_table(index="var", columns=["ds", "fam"], values="acc", aggfunc="mean", sort=False)).round(1)
    t10["Mean"] = (100 * df.groupby("var", sort=False).acc.mean()).round(1)
    print("\nTable 10: accuracy (%) after the onset\n", t10.to_string())
    fe = df[df["var"] == "FE-AE"].sort_values(["ds", "fam", "seed"]).acc.values
    for base in ("ET alone", "Static ensemble", "Adaptive, beta=1"):
        b = df[df["var"] == base].sort_values(["ds", "fam", "seed"]).acc.values
        print("FE-AE minus %s: %+.2f pp, paired t p=%.4f, Wilcoxon p=%.4f" % (
            base, 100 * (fe - b).mean(), stats.ttest_rel(fe, b).pvalue, stats.wilcoxon(fe, b).pvalue))


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
