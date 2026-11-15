"""Label-free comparison with a Kolmogorov-Smirnov test, type of change from the statistic that fires (Table 8)
and the detectability bound of Eq. (3)."""
import math
import time

import numpy as np
from scipy import stats
from scipy.stats import norm

from common import C, W
from feae import Reference
from synth import CHANGES, N, SEG, score

K, REFW = 4.0, 4


def gen(kind, seed, drift=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 5))
    for k in range(1, 5):
        sl = slice(k * SEG, (k + 1) * SEG)
        if not drift or k % 2 == 0:
            continue
        if kind == "mean 1.5":
            X[sl] += 1.5
        elif kind == "mean 0.5":
            X[sl] += 0.5
        elif kind == "mean 0.25":
            X[sl] += 0.25
        elif kind == "scale x2":
            X[sl] *= 2.0
        elif kind == "scale x1.25":
            X[sl] *= 1.25
        elif kind == "shape":
            X[sl] = rng.uniform(-math.sqrt(3), math.sqrt(3), (SEG, 5))
    return X[:, :3]


def _loop(X, make_ref, test):
    out, ref, buf, skip = [], None, [], 0
    for t in range(len(X) // W):
        w = X[t * W:(t + 1) * W]
        if skip > 0:
            skip -= 1
            buf.append(w)
            if skip == 0:
                ref, buf = make_ref(np.vstack(buf)), []
            continue
        if ref is None:
            buf.append(w)
            if len(buf) == REFW:
                ref, buf = make_ref(np.vstack(buf)), []
            continue
        r = test(ref, w)
        if r is not None:
            out.append(((t + 1) * W,) + r)
            skip, buf = REFW, []
    return out


def feae_run(X, seed):
    rng = np.random.default_rng(seed)

    def test(ref, w):
        z = np.abs(ref.stats(w) - ref.m) / ref.sd
        return (z[0], z[1]) if (z[0] > K or z[1] > K) else None

    return _loop(X, lambda R: Reference(R, rng, W), test)


def ks_run(X, alpha=1e-3):
    def test(ref, w):
        p = min(stats.ks_2samp(ref[:, j], w[:, j]).pvalue for j in range(3))
        return () if p < alpha / 3 else None

    return [a[0] for a in _loop(X, lambda R: R, test)]


def main():
    print("change | FE-AE detected/unmatched | alarm raised by: PFD, entropy only | KS detected/unmatched | no-drift alarms FE-AE, KS")
    for kind in ("mean 1.5", "mean 0.5", "mean 0.25", "scale x2", "scale x1.25", "shape"):
        d = u = kd = ku = pf = en = fa = kfa = 0
        for s in C.SEEDS:
            X = gen(kind, s)
            r = feae_run(X, s)
            a, _, b = score([x[0] for x in r])
            d, u = d + a, u + b
            for tt, zh, zf in r:
                if any(c <= tt <= c + C.TOL for c in CHANGES):
                    pf, en = pf + (zf > K), en + (zf <= K)
            a, _, b = score(ks_run(X))
            kd, ku = kd + a, ku + b
            X0 = gen(kind, s, False)
            fa, kfa = fa + len(feae_run(X0, s)), kfa + len(ks_run(X0))
        print(f"{kind:12s} | {d}/{u} | PFD {pf}, entropy only {en} | {kd}/{ku} | {fa}, {kfa}")
    n, m = W, 3
    print("\nEq. (3): mean shift delta (reference std. dev.) -> expected z of the crossing count (alarm if > %g)" % K)
    for dlt in (0.25, 0.3, 0.35, 0.41, 0.5):
        q = norm.cdf(dlt)
        print("  delta %.2f  z = %.2f" % (dlt, (n - 1) * (0.5 - 2 * q * (1 - q)) / math.sqrt((n - 1) / (4 * m))))
    X = np.random.default_rng(0).normal(size=(2048, 3))
    ref = Reference(X, np.random.default_rng(0), W)
    t = time.perf_counter()
    for _ in range(500):
        ref.stats(X[:W])
    a = (time.perf_counter() - t) / 500 * 1000
    t = time.perf_counter()
    for _ in range(500):
        [stats.ks_2samp(X[:, j], X[:W, j]) for j in range(3)]
    print("\nms per window: FE-AE statistics %.2f, KS tests %.2f" % (a, (time.perf_counter() - t) / 500 * 1000))


if __name__ == "__main__":
    main()
