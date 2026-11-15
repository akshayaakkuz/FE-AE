"""Fit the pipeline once per dataset / seed / scenario and cache the test-stream predictions.

    python experiments/prepare.py                # all datasets
    python experiments/prepare.py NSL UNSW       # selected datasets
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

from common import C, DATASETS, families, pkl
from feae import datasets
from feae.pipeline import prepare, split_novel, split_random


def save(name, make):
    if os.path.exists(pkl(name)):
        return
    P = make()
    with open(pkl(name), "wb") as f:
        pickle.dump(P, f)
    print(name, "features/pca/dfs", P["n_in"], P["n_pca"], P["n_dfs"], "fit %.0fs" % P["fit_time"], flush=True)


def main(names):
    for name in names:
        d, label, fam = datasets.load(name)
        y = pd.factorize(d[label])[0]
        order = d[fam].astype(str).values
        ex = C.EXCLUDE[name]
        print(name, d.shape, flush=True)
        for seed in C.SEEDS:                                   # stationary streams
            tr, te = split_random(y, seed)
            save(f"{name}_{seed}", lambda: prepare(d, label, ex, tr, te, seed, order))
        for f in families(order):                              # emerging-attack streams
            for seed in C.SEEDS3:
                tr, te, h = split_novel(y, order, f, seed)
                save(f"{name}_novel_{f}_{seed}", lambda: {**prepare(d, label, ex, tr, te, seed, order), "cp": h})
        for seed in C.SEEDS3:                                  # robustness: input noise and label poisoning
            tr, te = split_random(y, seed)
            for s in (0.5, 1.0):
                save(f"{name}_noise{s}_{seed}", lambda: {**prepare(d, label, ex, tr, te, seed, noise=s), "cp": len(te) // 2})
            for p in (0.1, 0.2):
                save(f"{name}_poison{int(p * 100)}_{seed}", lambda: prepare(d, label, ex, tr, te, seed, poison=p))


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
