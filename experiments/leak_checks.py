"""Leakage checks on NSL-KDD: duplicate rows between train and test, permuted-label control,
and absence of the unseen family from the training data of an emerging-attack split."""
import numpy as np
import pandas as pd

from common import C
from feae import datasets, single, stream
from feae.pipeline import prepare, split_novel, split_random


def main(seed=42):
    d, label, fam = datasets.load("NSL")
    y = d[label].values
    tr, te = split_random(y, seed)
    X = d.drop(columns=C.EXCLUDE["NSL"] + [label])
    h = pd.util.hash_pandas_object(X, index=False).values
    print("test rows with an identical row in the training data: %.1f %%" % (100 * np.isin(h[te], h[tr]).mean()))
    d2 = d.copy()
    d2[label] = np.random.default_rng(0).permutation(d2[label].values)
    P = prepare(d2, label, C.EXCLUDE["NSL"], tr, te, seed)
    print("permuted labels: FE-AE %.3f, ET alone %.3f (chance level expected)" % (stream(P, seed=seed)["acc"], single(P, "pC")))
    tr2, te2, cp = split_novel(y, d[fam].values, "satan", seed)
    print("rows of the unseen family in the training data:", int((d[fam].values[tr2] == "satan").sum()),
          "| rows shared by train and test:", len(set(tr2) & set(te2)))


if __name__ == "__main__":
    main()
