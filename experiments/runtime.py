"""Model size, accuracy and inference time of Extra Trees with 300 / 100 / 50 trees (Table 16) and cost of the signal."""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from common import C, DATASETS, out
from feae import Reference, datasets
from feae.pipeline import prepare, split_random


def main(names, seed=42):
    rows = []
    for name in names:
        d, label, fam = datasets.load(name)
        y = pd.factorize(d[label])[0]
        tr, te = split_random(y, seed)
        for n in (300, 100, 50):
            P = prepare(d, label, C.EXCLUDE[name], tr, te, seed, n_trees=n, keep_models=True)
            et = P["et"]
            et.set_params(n_jobs=1)                            # single-thread inference
            win = P["Fte"][:512]
            t0 = time.perf_counter()
            for _ in range(20):
                et.predict_proba(win)
            ms = (time.perf_counter() - t0) / 20 * 1000
            acc = (P["classes"][np.argmax(P["pC"], 1)] == P["yte"]).mean()
            rows.append(dict(ds=name, trees=n, size_mb=len(pickle.dumps(et)) / 2 ** 20, acc=acc, ms_per_window=ms))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out("runtime.csv"), index=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2048, 3))
    t0 = time.perf_counter()
    ref = Reference(X, rng, 512)
    cal = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in range(500):
        ref.stats(X[:512])
    print("drift signal: %.2f ms per window, calibration of a reference %.0f ms" % ((time.perf_counter() - t0) / 500 * 1000, cal))


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
