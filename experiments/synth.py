"""Synthetic drift streams, prequential errors of a Hoeffding tree, River detectors and scoring."""
import math

import numpy as np
from river import drift, tree
from river.datasets import synth

from common import C, W

SEG, NSEG = 5120, 5
N = SEG * NSEG                                   # 25,600 samples, change points at multiples of 5,120
CHANGES = [SEG * k for k in range(1, NSEG)]
STREAMS = ("MeanShift", "ScaleShift", "ShapeShift", "GradualMean", "SEA", "Agrawal", "Hyperplane")
COVARIATE = STREAMS[:4]


def gen(name, seed, drift_on=True):
    """Return (X, y). With drift_on=False the matched stream without drift is generated."""
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 5))
    y = np.zeros(N, int)

    def seg(k):
        return k if drift_on else 0

    if name in ("SEA", "Agrawal"):
        xs, ys = [], []
        for k in range(NSEG):
            kk = seg(k) % 2
            g = (synth.SEA(variant=[0, 3][kk], seed=seed + k) if name == "SEA"
                 else synth.Agrawal(classification_function=[0, 2][kk], seed=seed + k))
            for x, yy in g.take(SEG):
                xs.append(list(x.values()))
                ys.append(int(yy))
        return np.array(xs, float), np.array(ys)
    for k in range(NSEG):
        sl = slice(k * SEG, (k + 1) * SEG)
        kk = seg(k)
        if name == "Hyperplane":
            xs = rng.uniform(0, 1, (SEG, 5))
            wv = np.random.default_rng(seed * 100 + (kk % 2)).uniform(-1, 1, 5)
            yy = ((xs - 0.5) @ wv > 0).astype(int)
        elif name == "MeanShift":
            mu = 1.5 * (kk % 2)
            xs = rng.normal(mu, 1, (SEG, 5))
            yy = ((xs[:, :3] - mu).sum(1) > 0).astype(int)
        elif name == "ScaleShift":
            sd = 1.0 if kk % 2 == 0 else 2.0
            xs = rng.normal(0, sd, (SEG, 5))
            yy = (xs[:, :3].sum(1) > 0).astype(int)
        elif name == "ShapeShift":
            xs = rng.normal(0, 1, (SEG, 5)) if kk % 2 == 0 else rng.uniform(-math.sqrt(3), math.sqrt(3), (SEG, 5))
            yy = (xs[:, :3].sum(1) > 0).astype(int)
        elif name == "GradualMean":
            mu0 = 1.5 * ((kk - 1) % 2) if kk > 0 else 0.0
            mu1 = 1.5 * (kk % 2)
            a = np.clip(np.arange(SEG) / 1024, 0, 1) if kk > 0 else np.ones(SEG)
            mu = (mu0 + (mu1 - mu0) * a)[:, None]
            xs = rng.normal(0, 1, (SEG, 5)) + mu
            yy = ((xs[:, :3] - mu).sum(1) > 0).astype(int)
        flip = rng.random(SEG) < 0.05
        X[sl] = xs
        y[sl] = np.where(flip, 1 - yy, yy)
    return X, y


def errors(X, y):
    """Prequential 0/1 errors of a Hoeffding tree (test-then-train)."""
    m = tree.HoeffdingTreeClassifier()
    e = np.zeros(len(y), int)
    for i in range(len(y)):
        x = {j: float(v) for j, v in enumerate(X[i])}
        p = m.predict_one(x)
        e[i] = int(p != y[i]) if p is not None else 1
        m.learn_one(x, int(y[i]))
    return e


DETECTORS = {
    "ADWIN": lambda: drift.ADWIN(),
    "DDM": lambda: drift.binary.DDM(),
    "EDDM": lambda: drift.binary.EDDM(),
    "HDDM_A": lambda: getattr(drift.binary, "HDDM_A", getattr(drift.binary, "HDDMA", None))(),
    "PageHinkley": lambda: drift.PageHinkley(),
    "KSWIN": lambda: drift.KSWIN(seed=42),          # KSWIN samples its window at random
}


def run_detector(d, e):
    al = []
    for i, v in enumerate(e):
        d.update(int(v))
        if d.drift_detected:
            al.append(i)
    return al


def alarms(src):
    """Alarm string of feae.detect -> alarm positions (end of the window)."""
    return [(i + 1) * W for i, ch in enumerate(src) if ch != "."]


def score(al, changes=CHANGES, tol=C.TOL):
    """Returns (detected changes, mean delay, unmatched alarms)."""
    used, delays = set(), []
    for c in changes:
        for i, a in enumerate(al):
            if i not in used and c <= a <= c + tol:
                used.add(i)
                delays.append(a - c)
                break
    return len(delays), (np.mean(delays) if delays else np.nan), len(al) - len(delays)
