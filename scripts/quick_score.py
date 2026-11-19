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
from unlearn.forget_quality import compute_forget_score_from_confs, compute_logit_scaled_confidence


def acc(runs, split):
    return float(np.mean([(r[f"{split}_logits"].argmax(1) == r[f"{split}_targets"]).mean() for r in runs]))


def confs(runs):
    return np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                     for r in runs])


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--orig", default="own")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--csv", default=None)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else None)
    p.add_argument("settings", nargs="+", help="method/tag")
    a = p.parse_args()
    base = exp_dir(a.out, a.dataset, a.arch, a.version)
    ret = load_runs(base / "reference" / "retrain", "seed_*.npz")
    rc = confs(ret)
    ra_r, ta_r = acc(ret, "retain"), acc(ret, "test")
    rows = []
    for s in a.settings:
        runs = load_runs(base / a.orig / "runs" / s)[: a.n]
        if not runs:
            print(f"skip {s}: no runs")
            continue
        uc = confs(runs)
        groups = [rc[i:i + len(runs)] for i in range(0, len(rc) - len(runs) + 1, len(runs))]
        fs = [compute_forget_score_from_confs(uc, g, device=a.device) for g in groups]
        f = float(np.mean(fs))
        ra, fa, ta = acc(runs, "retain"), acc(runs, "forget"), acc(runs, "test")
        rows.append({"setting": s, "n": len(runs), "score": f * ra / ra_r * ta / ta_r, "F": f, "F_sd": float(np.std(fs)),
                     "RA": ra, "FA": fa, "TA": ta})
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
