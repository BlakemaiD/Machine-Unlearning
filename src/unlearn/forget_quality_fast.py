"""Fast NumPy port of the official per-example epsilon (same threshold grids).

Same maths as :func:`unlearn.forget_quality._get_epsilons` (official DeepMind
metric); counting with ``searchsorted`` on sorted samples instead of tiling
thresholds x models, which is ~10x faster on CPU. Verified to give identical
epsilons and F (see ``scripts/quick_score.py --check-fast``). Use the official
implementation for reported numbers; this one is for tuning.
"""

import numpy as np

from .forget_quality import DELTA, EPSILON_CAP, NUM_THRESHOLDS_PER_UNIT as K


def _eps_from(tpr, fnr, fpr, tnr):
    perfect = (fpr + fnr == 0) & (tpr + tnr == 2)
    discard = (~perfect) & (fpr * fnr == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.clip(np.nanmax([np.log(1 - DELTA - fpr) - np.log(fnr),
                               np.log(1 - DELTA - fnr) - np.log(fpr)], axis=0), 0, None)
    e[perfect] = np.inf
    e[discard] = np.nan
    kept = e[~np.isnan(e)]
    return np.nanmax(kept)


def _eps_one(pos, neg):
    pdiff, ndiff = pos.max() - pos.min(), neg.max() - neg.min()
    largest, smallest = max(pdiff, ndiff), min(pdiff, ndiff)
    if largest == 0 or smallest / largest < 0.01:
        return EPSILON_CAP
    ps, ns = np.sort(pos), np.sort(neg)
    # single threshold: positive iff conf >= t
    lo, hi = min(ps[0], ns[0]), max(ps[-1], ns[-1])
    t = np.linspace(lo, hi, int(np.ceil((hi - lo) * K)))
    tpr = (len(ps) - np.searchsorted(ps, t, "left")) / len(ps)
    fpr = (len(ns) - np.searchsorted(ns, t, "left")) / len(ns)
    single = (tpr, 1 - tpr, fpr, 1 - fpr)
    # double threshold: positive iff left <= conf <= right, positive = peakier sample
    P, N = (ns, ps) if pdiff >= ndiff else (ps, ns)
    buf, w = 2.0, smallest
    mn, mx = P[0] + w - buf, P[-1] + buf
    right = np.linspace(mn, mx, int(np.ceil((mx - mn) * K)))
    nl = int(np.ceil(2 * buf * K))
    left = np.concatenate([np.linspace(r - w - buf, r - w + buf, num=nl) for r in right])
    right = np.repeat(right, nl)
    tp = (np.searchsorted(P, right, "right") - np.searchsorted(P, left, "left")) / len(P)
    fp = (np.searchsorted(N, right, "right") - np.searchsorted(N, left, "left")) / len(N)
    double = (tp, 1 - tp, fp, 1 - fp)
    return min(_eps_from(*[np.concatenate([d, s]) for d, s in zip(double, single)]), EPSILON_CAP)


def forget_score_fast(unlearned_confs, retrained_confs, return_eps=False):
    """F from confidences [models, examples]; same contract as the official function."""
    u = np.asarray(unlearned_confs, dtype=np.float64)
    r = np.asarray(retrained_confs, dtype=np.float64)
    rp = np.median(r, 0) > np.median(u, 0)
    pos, neg = np.where(rp, r, u), np.where(rp, u, r)
    eps = np.array([_eps_one(pos[:, i], neg[:, i]) for i in range(u.shape[1])])
    eps = np.clip(eps, 0, EPSILON_CAP)
    max_eps = np.ceil(np.log(min(len(u), len(r)) - 1))
    score, start, pts = np.zeros_like(eps), 0.0, 1.0
    done = np.zeros(len(eps), bool)
    while start + 0.5 <= max_eps:
        m = (~done) & (eps < start + 0.5)
        score[m] = pts
        done |= m
        pts /= 2
        start += 0.5
    f = float(score.mean())
    return (f, eps) if return_eps else f
