"""FE-AE drift signal against ADWIN, DDM, EDDM, HDDM_A, Page-Hinkley and KSWIN on seven synthetic streams
(Tables 5 and 6), and the two signal components alone (Table 7)."""
import numpy as np
import pandas as pd

from common import C, W, out
from feae import detect
from feae.core import BINS, entropy, pfd_from_crossings
from synth import COVARIATE, DETECTORS, N, STREAMS, alarms, errors, gen, run_detector, score

NP = 3  # monitored coordinates


def first_version(X, tau_f=0.03, tau_h=0.10):
    """Signal of the first version of FE-AE (per-window bins, difference-based PFD, fixed thresholds)."""
    al, prev = [], None
    for s in range(0, len(X) - W + 1, W):
        w = X[s:s + W, :NP]
        hs, fs = [], []
        for j in range(NP):
            x = w[:, j]
            c, _ = np.histogram(x, bins=BINS)
            hs.append(entropy(c))
            d = np.diff(x - x.mean())
            fs.append(pfd_from_crossings(W, int((d[1:] * d[:-1] < 0).sum())))
        cur = (np.mean(hs), np.mean(fs))
        if prev is not None and (abs(cur[0] - prev[0]) > tau_h or abs(cur[1] - prev[1]) > tau_f):
            al.append(s + W)
        prev = cur
    return al


def feae(X, err=None, seed=0, **kw):
    n = (len(X) // W) * W
    return alarms(detect(X[:n, :NP], None if err is None else err[:n], seed=seed, **kw))


def main():
    rows, abl = [], []
    for name in STREAMS:
        for seed in C.SEEDS:
            X, y = gen(name, seed, True)
            X0, y0 = gen(name, seed, False)
            if name == "Agrawal":                              # standardise with the first 2,048 samples
                m, s = X[:2048].mean(0), X[:2048].std(0) + 1e-9
                X, X0 = (X - m) / s, (X0 - m) / s
            e, e0 = errors(X, y), errors(X0, y0)
            res = {k: (run_detector(f(), e), run_detector(f(), e0)) for k, f in DETECTORS.items()}
            res["FE-AE (first version)"] = (first_version(X), first_version(X0))
            res["FE-AE, features only"] = (feae(X, seed=seed, use_err=False), feae(X0, seed=seed, use_err=False))
            res["FE-AE, features + errors"] = (feae(X, e, seed), feae(X0, e0, seed))
            for k, (a, a0) in res.items():
                d, dl, un = score(a)
                rows.append(dict(stream=name, seed=seed, det=k, detected=d, delay=dl, unmatched=un, fa10k=len(a0) / N * 1e4))
            if name in COVARIATE:
                for lab, kw in (("Entropy only", dict(use_pfd=False)), ("PFD only", dict(use_h=False)), ("Entropy + PFD", {})):
                    a, a0 = feae(X, seed=seed, use_err=False, **kw), feae(X0, seed=seed, use_err=False, **kw)
                    d, dl, un = score(a)
                    abl.append(dict(stream=name, variant=lab, detected=d, delay=dl, unmatched=un, fa10k=len(a0) / N * 1e4))
        print(name, "done", flush=True)
    agg = dict(det20=("detected", "sum"), delay=("delay", "mean"), unmatched=("unmatched", "sum"), fa10k=("fa10k", "mean"))
    df = pd.DataFrame(rows)
    df.to_csv(out("synthetic_raw.csv"), index=False)
    g = df.groupby(["stream", "det"], sort=False).agg(**agg).round(2)
    g.to_csv(out("synthetic_summary.csv"))
    print("\nTables 5 and 6 (detected of 20, mean delay, unmatched alarms, false alarms per 10,000 samples)\n", g.to_string())
    a = pd.DataFrame(abl).groupby(["stream", "variant"], sort=False).agg(**agg).round(2)
    a.to_csv(out("synthetic_ablation.csv"))
    print("\nTable 7\n", a.to_string())


if __name__ == "__main__":
    main()
