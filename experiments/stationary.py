"""Stationary streams: baselines (Table 11), paired t-tests (Table 12), sensitivity and label delay (Table 14),
false alarms per run and alarms on the class-ordered streams (Section 5.3)."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import C, DATASETS, load_pkl, out
from feae import reorder, single, stream

VARIANTS = {
    "Static ensemble": dict(adapt=False),
    "Adaptive, beta=1": dict(trigger="none", beta=1),
    "Adaptive, no trigger": dict(trigger="none"),
    "FE-AE, features only": dict(use_err=False),
    "FE-AE": dict(),
}
SENS = {"beta=1": dict(beta=1), "beta=8": dict(beta=8), "beta=30": dict(), "beta=50": dict(beta=50),
        "alpha=0.5": dict(alpha=0.5), "alpha=0.9": dict(alpha=0.9), "W=256": dict(W=256), "W=1024": dict(W=1024),
        "K=3": dict(K=3), "K=5": dict(K=5), "d=1": dict(delay=1), "d=2": dict(delay=2), "d=4": dict(delay=4)}


def main(names):
    rows, sens, co = [], [], []
    for name in names:
        for seed in C.SEEDS:
            P = load_pkl(f"{name}_{seed}")
            for k, nm in (("pA", "LR alone"), ("pB", "CNB alone"), ("pC", "ET alone")):
                rows.append(dict(ds=name, seed=seed, var=nm, acc=single(P, k), f1=np.nan, flags=0))
            for nm, o in VARIANTS.items():
                r = stream(P, seed=seed, **o)
                rows.append(dict(ds=name, seed=seed, var=nm, acc=r["acc"], f1=r["f1"], flags=r["flags"]))
            for nm, o in SENS.items():
                r = stream(P, seed=seed, **o)
                sens.append(dict(ds=name, seed=seed, setting=nm, acc=r["acc"], flags=r["flags"]))
            if P.get("order") is not None:                     # class-ordered stream: sort the test part by family
                perm = np.argsort(P["order"], kind="stable")
                fam = P["order"][perm]
                r = stream(reorder(P, perm), seed=seed)
                bounds = [i for i in range(1, len(fam)) if fam[i] != fam[i - 1]]
                al = [(i + 1) * 512 for i, ch in enumerate(r["src"]) if ch != "."]
                near = sum(any(b <= a <= b + C.TOL for b in bounds) for a in al)
                co.append(dict(ds=name, seed=seed, alarms=len(al), after_boundary=near))
    df, sf = pd.DataFrame(rows), pd.DataFrame(sens)
    df.to_csv(out("stationary_raw.csv"), index=False)
    sf.to_csv(out("sensitivity_raw.csv"), index=False)
    piv = lambda d, v: (100 * d.pivot_table(index=v, columns="ds", values="acc", aggfunc="mean", sort=False)).round(2)[list(names)]
    print("\nTable 11: accuracy (%), mean over seeds\n", piv(df, "var").to_string())
    fe = df[df["var"] == "FE-AE"]
    print("FE-AE std (pp):", (100 * fe.groupby("ds", sort=False).acc.std()).round(2).to_dict())
    print("FE-AE macro-F1:", (100 * fe.groupby("ds", sort=False).f1.mean()).round(2).to_dict())
    print("False alarms per run:", df[df["var"].isin(["FE-AE", "FE-AE, features only"])].groupby(["var", "ds"], sort=False).flags.mean().round(2).to_dict())
    print("\nTable 12: paired t-tests, FE-AE minus baseline in pp (p)")
    for name in names:
        a = fe[fe.ds == name].sort_values("seed").acc.values
        line = []
        for base in ("Static ensemble", "Adaptive, beta=1", "ET alone"):
            b = df[(df.ds == name) & (df["var"] == base)].sort_values("seed").acc.values
            line.append("%s %+.2f (%.3f)" % (base, 100 * (a - b).mean(), stats.ttest_rel(a, b).pvalue))
        print(name, " | ".join(line))
    print("\nTable 14: sensitivity, accuracy (%)\n", piv(sf, "setting").to_string())
    print("False alarms per run by K:", sf[sf.setting.isin(["K=3", "beta=30", "K=5"])].groupby("setting", sort=False).flags.mean().round(2).to_dict())
    if co:
        print("\nClass-ordered streams (alarms over all seeds, alarms within 3 windows after a family boundary)\n",
              pd.DataFrame(co).groupby("ds", sort=False)[["alarms", "after_boundary"]].sum().to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
