#!/usr/bin/env python
"""Quick comparison of method settings on the first n runs each (for tuning).

Scores each <method>/<tag> run folder with the competition metric using its
first ``--n`` runs against disjoint groups of ``--n`` retrained models (the
metric needs equal counts) and averages F over the groups, so settings and
baselines are compared at the same number of models.

    python scripts/quick_score.py --orig own --n 16 lash/s01 lash/s02 neggrad_plus/default
"""

import argparse
import sys

import numpy as np
import pandas as pd

from common import add_common_args, exp_dir
from unlearn.evaluation import load_runs
from unlearn import metrics as M
from unlearn.forget_quality import compute_forget_score_from_confs, compute_logit_scaled_confidence
from unlearn.forget_quality_fast import forget_score_fast
from sklearn.metrics import roc_auc_score


def acc(runs, split):
    return float(np.mean([(r[f"{split}_logits"].argmax(1) == r[f"{split}_targets"]).mean() for r in runs]))


def mia(runs):
    """Mean (forget-vs-test loss MIA, MIA-efficacy) over runs; see unlearn.metrics."""
    a, e = [], []
    for i, r in enumerate(runs):
        a.append(M.mia_from_losses(M.losses_from_logits(r["forget_logits"], r["forget_targets"]),
                                   M.losses_from_logits(r["test_logits"], r["test_targets"]), seed=i))
        e.append(M.mia_efficacy(r["retain_logits"], r["retain_targets"], r["test_logits"],
                                r["test_targets"], r["forget_logits"], r["forget_targets"], seed=i))
    return float(np.mean(a)), float(np.mean(e))


def ulira_auc(runs, ret_f, ret_t):
    """Offline U-LiRA, per model: each forget/test image gets N(mu_i, sd_i) from the
    retrained models (OUT shadows); attack score z^2 (two-sided: catches under- and
    over-forgetting); AUC of forget vs test, averaged over models. 0.5 = no leak.
    ret_f / ret_t: [n_ref, n_ex] confidences of retrained models not being scored."""
    if len(ret_f) < 2:  # n = 64 uses every retrained model as reference: no shadows left
        return float("nan")
    mf, sf = ret_f.mean(0), ret_f.std(0) + 1e-3
    mt, st = ret_t.mean(0), ret_t.std(0) + 1e-3
    out = []
    for r in runs:
        zf = (compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1] - mf) / sf
        zt = (compute_logit_scaled_confidence(r["test_logits"], r["test_targets"])[1] - mt) / st
        out.append(roc_auc_score(np.r_[np.ones(len(zf)), np.zeros(len(zt))], np.r_[zf ** 2, zt ** 2]))
    return float(np.mean(out))


def tconfs(runs):
    return np.stack([compute_logit_scaled_confidence(r["test_logits"], r["test_targets"])[1]
                     for r in runs])


def confs(runs):
    return np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                     for r in runs])


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--orig", default="own")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--csv", default=None)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else None)
    p.add_argument("--no-f", action="store_true", help="skip the (slow) forgetting score F")
    p.add_argument("--fast", action="store_true", help="fast NumPy port of F (same maths) for tuning")
    p.add_argument("--check-fast", type=int, default=0, metavar="K",
                   help="compare official vs fast F on the first K images of each setting")
    p.add_argument("settings", nargs="+", help="method/tag")
    a = p.parse_args()
    base = exp_dir(a.out, a.dataset, a.arch, a.version)
    ret = load_runs(base / "reference" / "retrain", "seed_*.npz")
    rc = confs(ret)
    ra_r, ta_r = acc(ret, "retain"), acc(ret, "test")
    mia_r, eff_r = mia(ret[: a.n])
    # U-LiRA reference: retrained models 16..63 as shadows; the first n retrained models
    # (scored against the same shadows) give the no-leak reference
    rt = tconfs(ret)
    sh_f, sh_t = rc[a.n:], rt[a.n:]
    ul_r = ulira_auc(ret[: a.n], sh_f, sh_t)
    rows = [{"setting": "retrained (reference)", "n": a.n, "score": np.nan, "F": np.nan, "F_sd": np.nan,
             "RA": ra_r, "FA": acc(ret, "forget"), "TA": ta_r, "MIA": mia_r, "MIA_eff": eff_r, "ULiRA_AUC": ul_r}]
    for s in a.settings:
        runs = load_runs(base / a.orig / "runs" / s)[: a.n]
        if not runs:
            print(f"skip {s}: no runs")
            continue
        uc = confs(runs)
        groups = [rc[i:i + len(runs)] for i in range(0, len(rc) - len(runs) + 1, len(runs))]
        if a.check_fast:
            k = a.check_fast
            fo = compute_forget_score_from_confs(uc[:, :k], groups[0][:, :k], device=a.device)
            ff = forget_score_fast(uc[:, :k], groups[0][:, :k])
            print(f"check {s}: official {fo:.6f} fast {ff:.6f} on {k} images")
        if a.no_f:
            fs = [np.nan]
        elif a.fast:
            fs = [forget_score_fast(uc, g) for g in groups]
        else:
            fs = [compute_forget_score_from_confs(uc, g, device=a.device) for g in groups]
        f = float(np.mean(fs))
        ra, fa, ta = acc(runs, "retain"), acc(runs, "forget"), acc(runs, "test")
        m, e = mia(runs)
        rows.append({"setting": s, "n": len(runs), "score": f * ra / ra_r * ta / ta_r, "F": f, "F_sd": float(np.std(fs)),
                     "RA": ra, "FA": fa, "TA": ta, "MIA": m, "MIA_eff": e, "ULiRA_AUC": ulira_auc(runs, sh_f, sh_t)})
    df = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="first")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
