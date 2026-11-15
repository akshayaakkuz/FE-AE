"""Robustness (Table 15): Gaussian noise on the second half of the test stream and flipped training labels."""
import sys

import numpy as np
import pandas as pd

from common import C, DATASETS, acc_after, load_pkl, out, score_onset, single_preds
from feae import single, stream


def main(names):
    rows = []
    for name in names:
        for seed in C.SEEDS3:
            for kind in ("noise0.5", "noise1.0", "poison10", "poison20"):
                P = load_pkl(f"{name}_{kind}_{seed}")
                cp = P.get("cp", 0)                            # noise: accuracy on the perturbed part only
                rows.append(dict(ds=name, kind=kind, seed=seed, var="ET alone", acc=acc_after(single_preds(P, "pC"), P["yte"], cp)))
                for nm, o in (("Static ensemble", dict(adapt=False)), ("FE-AE, features only", dict(use_err=False)), ("FE-AE", dict())):
                    r = stream(P, seed=seed, **o)
                    row = dict(ds=name, kind=kind, seed=seed, var=nm, acc=acc_after(r["preds"], P["yte"], cp))
                    if kind.startswith("noise"):
                        row["det"], row["fa"], row["delay"] = score_onset(r["src"], cp)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out("robustness_raw.csv"), index=False)
    t = (100 * df[df["var"] != "FE-AE, features only"].pivot_table(index=["kind", "var"], columns="ds", values="acc", aggfunc="mean", sort=False)).round(1)
    print("\nTable 15: accuracy (%)\n", t[list(names)].to_string())
    n = df[df.kind.str.startswith("noise") & df["var"].str.startswith("FE-AE")]
    print("\nOnset of the perturbation\n", n.groupby(["kind", "var"], sort=False).agg(detected=("det", "sum"), runs=("det", "size"), false_alarms=("fa", "sum"), delay=("delay", "mean")).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
