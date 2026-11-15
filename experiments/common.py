"""Helpers shared by the experiment scripts."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

DATASETS = ("ToN", "NSL", "UNSW", "DDoS", "IoT")
W = 512


def out(name):
    os.makedirs(C.OUT_DIR, exist_ok=True)
    return os.path.join(C.OUT_DIR, name)


def pkl(name):
    return out(name.replace(" ", "_").replace("/", "_") + ".pkl")


def load_pkl(name):
    with open(pkl(name), "rb") as f:
        return pickle.load(f)


def families(order, k=3, min_rows=800):
    """The k largest attack families (at least min_rows flows in the sample)."""
    vc = pd.Series(order).value_counts()
    return [v for v in vc.index if str(v).lower() not in ("normal", "benign", "nan", "") and vc[v] >= min_rows][:k]


def score_onset(src, cp, tol=C.TOL):
    """Match the alarms of an alarm string against one change point cp (in samples).
    Returns (detected 0/1, false alarms, delay in samples or nan)."""
    alarms = [(i + 1) * W for i, ch in enumerate(src) if ch != "."]
    hit = [a for a in alarms if cp <= a <= cp + tol]
    return int(bool(hit)), len(alarms) - (1 if hit else 0), (hit[0] - cp if hit else np.nan)


def acc_after(preds, y, cp):
    return float((np.asarray(preds)[cp:] == np.asarray(y)[cp:]).mean())


def single_preds(P, k):
    return P["classes"][np.argmax(P[k], 1)]
