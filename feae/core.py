"""FE-AE core: reference-anchored fractal-entropy drift signal and adaptive ensemble weighting.

Notation follows the paper:
  W      window length (flows)                      R (refw)  windows forming the reference set
  m      monitored latent coordinates (3)           B (BINS)  histogram bins per coordinate
  N_b    bootstrap windows for calibration (200)    K         alarm threshold (bootstrap std. dev.)
  alpha  EWMA smoothing of the weights              beta      sharpening exponent of the target weights
  d      label delay in windows                     delta     minimum rise of the error rate (error channel)
"""
import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

BINS = 20
N_BOOT = 200

DEFAULTS = dict(W=512, alpha=0.7, beta=30.0, K=4.0, refw=4, trigger="feae", use_h=True, use_pfd=True,
                use_err=True, adapt=True, delay=0, seed=0, dmin=0.01)


def pfd_from_crossings(n, n_delta):
    """Petrosian fractal dimension (Eq. 2) of a sequence of length n with n_delta sign changes."""
    return math.log10(n) / (math.log10(n) + math.log10(n / (n + 0.4 * n_delta)))


def entropy(counts):
    """Shannon entropy (natural logarithm) of a histogram; empty bins contribute zero."""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


class Reference:
    """Reference set: fixed bins, reference mean and bootstrap-calibrated mean / std of both statistics."""

    def __init__(self, ref, rng, W, boot=N_BOOT):
        self.lo = ref.min(0)
        self.hi = ref.max(0)
        self.mu = ref.mean(0)
        st = np.array([self.stats(ref[rng.integers(0, len(ref), W)]) for _ in range(boot)])
        self.m = st.mean(0)
        self.sd = st.std(0) + 1e-9

    def stats(self, w):
        """Return (H_t, PFD_t) of window w, averaged over the monitored coordinates (Eqs. 1 and 2)."""
        hs, fs = [], []
        for j in range(w.shape[1]):
            x = np.clip(w[:, j], self.lo[j], self.hi[j])
            c, _ = np.histogram(x, bins=BINS, range=(self.lo[j], self.hi[j] + 1e-12))
            hs.append(entropy(c))
            b = w[:, j] > self.mu[j]
            fs.append(pfd_from_crossings(len(w), int((b[1:] != b[:-1]).sum())))
        return np.array([np.mean(hs), np.mean(fs)])


def error_stats(e):
    """The same two statistics on a 0/1 error sequence: binary entropy of the error rate and PFD of e."""
    p = min(max(e.mean(), 1e-6), 1 - 1e-6)
    h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return np.array([h, pfd_from_crossings(len(e), int((e[1:] != e[:-1]).sum()))]), p


def _error_reference(ebuf, rng, W):
    p = min(max(float(np.concatenate(ebuf).mean()), 1e-4), 1 - 1e-4)
    sim = np.array([error_stats((rng.random(W) < p).astype(int))[0] for _ in range(N_BOOT)])
    return sim.mean(0), sim.std(0) + 1e-9, p


def stream(P, **over):
    """Replay a prepared test stream window by window (Algorithm 1).

    P holds the class probabilities of the three learners (pA: LR, pB: CNB, pC: ET), the labels `yte`,
    the class array `classes` and the monitored latent coordinates `pil`.
    Returns accuracy, macro-F1, the alarm string `src` (one character per window: '.', 'H' entropy,
    'F' PFD, 'E' error channel), the predictions and a per-window trace.
    """
    c = {**DEFAULTS, **over}
    W = c["W"]
    cl = P["classes"]
    pA, pB, pC, yte, pil = P["pA"], P["pB"], P["pC"], P["yte"], P["pil"]
    N = len(yte)
    rng = np.random.default_rng(c["seed"])
    w = np.ones(3) / 3
    preds = np.zeros(N, int)
    hist, errh = [], []
    ref, buf, skip = None, [], 0
    eref, ebuf = None, []
    flags, trace = [], []
    for t in range(math.ceil(N / W)):
        idx = np.arange(t * W, min((t + 1) * W, N))
        full = len(idx) == W
        fired, st = None, None
        if c["trigger"] == "feae" and full:
            win = pil[idx]
            if skip > 0:                       # collecting the new reference after an alarm
                skip -= 1
                buf.append(win)
                if skip == 0:
                    ref = Reference(np.vstack(buf), rng, W)
                    buf = []
            elif ref is None:                  # collecting the first reference
                buf.append(win)
                if len(buf) == c["refw"]:
                    ref = Reference(np.vstack(buf), rng, W)
                    buf = []
            else:
                st = ref.stats(win)
                z = np.abs(st - ref.m) / ref.sd
                if c["use_h"] and z[0] > c["K"]:
                    fired = "H"
                elif c["use_pfd"] and z[1] > c["K"]:
                    fired = "F"
                if fired is None and c["use_err"] and len(errh) > c["delay"]:
                    e = errh[-1 - c["delay"]]
                    if len(e) == W:
                        if eref is None:
                            ebuf.append(e)
                            if len(ebuf) == c["refw"]:
                                eref = _error_reference(ebuf, rng, W)
                                ebuf = []
                        else:
                            se, pw = error_stats(e)
                            ze = (se - eref[0]) / eref[1]
                            if (ze[0] > c["K"] or ze[1] > c["K"]) and pw - eref[2] > c["dmin"]:
                                fired = "E"
                if fired:
                    skip = c["refw"]
                    buf = []
                    eref, ebuf = None, []
        drift = bool(fired) and c["adapt"]
        wt = w.copy()
        pa = hist[-1 - c["delay"]] if len(hist) > c["delay"] else None   # most recent labelled accuracies
        if drift and pa is not None:           # weight reset on an alarm
            q = pa ** c["beta"]
            wt = q / q.sum()
        pf = wt[0] * pA[idx] + wt[1] * pB[idx] + wt[2] * pC[idx]
        preds[idx] = cl[np.argmax(pf, 1)]      # prediction before label feedback (Eq. 7)
        acc = np.array([(cl[np.argmax(p[idx], 1)] == yte[idx]).mean() for p in (pA, pB, pC)])
        hist.append(acc)
        errh.append((preds[idx] != yte[idx]).astype(int))
        if c["adapt"] and pa is not None:      # Eqs. 5 and 6
            q = pa ** c["beta"]
            w = c["alpha"] * wt + (1 - c["alpha"]) * q / (q.sum() + 1e-12)
            w /= w.sum()
        flags.append(fired or "")
        trace.append((wt.copy(), (preds[idx] == yte[idx]).mean(), None if st is None else st.copy()))
    return dict(acc=accuracy_score(yte, preds), f1=f1_score(yte, preds, average="macro"),
                flags=sum(1 for f in flags if f), src="".join(f or "." for f in flags), trace=trace, preds=preds)


def detect(pil, err=None, **over):
    """Drift signal only (no ensemble): `pil` are the monitored coordinates, `err` optional 0/1 errors
    of any classifier. Returns one character per full window ('.', 'H', 'F' or 'E')."""
    c = {**DEFAULTS, **over}
    W = c["W"]
    rng = np.random.default_rng(c["seed"])
    ref, buf, skip = None, [], 0
    eref, ebuf = None, []
    out = []
    for t in range(len(pil) // W):
        win = pil[t * W:(t + 1) * W]
        fired = None
        if skip > 0:
            skip -= 1
            buf.append(win)
            if skip == 0:
                ref = Reference(np.vstack(buf), rng, W)
                buf = []
        elif ref is None:
            buf.append(win)
            if len(buf) == c["refw"]:
                ref = Reference(np.vstack(buf), rng, W)
                buf = []
        else:
            z = np.abs(ref.stats(win) - ref.m) / ref.sd
            if c["use_h"] and z[0] > c["K"]:
                fired = "H"
            elif c["use_pfd"] and z[1] > c["K"]:
                fired = "F"
            if fired is None and c["use_err"] and err is not None and t > c["delay"]:
                e = err[(t - 1 - c["delay"]) * W:(t - c["delay"]) * W]
                if eref is None:
                    ebuf.append(e)
                    if len(ebuf) == c["refw"]:
                        eref = _error_reference(ebuf, rng, W)
                        ebuf = []
                else:
                    se, pw = error_stats(e)
                    ze = (se - eref[0]) / eref[1]
                    if (ze[0] > c["K"] or ze[1] > c["K"]) and pw - eref[2] > c["dmin"]:
                        fired = "E"
            if fired:
                skip = c["refw"]
                buf = []
                eref, ebuf = None, []
        out.append(fired or ".")
    return "".join(out)


def reorder(P, perm):
    return {**P, **{k: P[k][perm] for k in ("pA", "pB", "pC", "yte", "pil")}}


def single(P, k):
    """Accuracy of one base learner alone (k = 'pA', 'pB' or 'pC')."""
    return accuracy_score(P["yte"], P["classes"][np.argmax(P[k], 1)])
