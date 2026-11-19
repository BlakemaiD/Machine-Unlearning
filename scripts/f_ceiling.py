#!/usr/bin/env python
"""Reference ceiling for the forgetting score F at a given number of models.

Splits the N retrained oracle models into two halves and scores one half
against the other: two sets of genuinely retrained models are as
indistinguishable as anything can be, so this is the best F achievable with
N/2 models per side. Selected methods are scored the same way (first N/2
unlearned runs vs the same retrained half) for an apples-to-apples comparison.

    python scripts/f_ceiling.py --orig own --methods neggrad_plus sebastian_4 --csv out.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from common import add_common_args, exp_dir
from unlearn.evaluation import load_runs
from unlearn.forget_quality import (compute_forget_score_from_confs,
                                    compute_logit_scaled_confidence)


def confs(runs):
    return np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                     for r in runs])


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--orig", default="own", choices=["own", "challenge"])
    p.add_argument("--methods", nargs="*", default=[])
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    base = exp_dir(a.out, a.dataset, a.arch, a.version)
    r = confs(load_runs(base / "reference" / "retrain", "seed_*.npz"))
    half_a, half_b = r[0::2], r[1::2]
    n = len(half_b)
    ceiling = compute_forget_score_from_confs(half_a, half_b)
    rows = [{"name": "retrain vs retrain (ceiling)", "n_per_side": n, "F": ceiling,
             "fraction_of_ceiling": 1.0}]
    for m in a.methods:
        u = confs(load_runs(base / a.orig / "runs" / m / "default"))[:n]
        f = compute_forget_score_from_confs(u, half_b)
        rows.append({"name": m, "n_per_side": n, "F": f, "fraction_of_ceiling": f / ceiling})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
