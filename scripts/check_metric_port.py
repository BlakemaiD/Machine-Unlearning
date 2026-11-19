#!/usr/bin/env python
"""Check that unlearn.forget_quality reproduces the official metric.py exactly.

Imports the untouched organiser code from references/code/unlearning_evaluation
(it hard-codes .cuda(), so this needs a GPU) and compares forgetting scores on
synthetic confidence distributions ranging from identical to fully separated.
"""

import sys
import types
import warnings

import numpy as np

from common import REPO
from unlearn.forget_quality import (compute_forget_score_from_confs,
                                    compute_logit_scaled_confidence)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # official log(0) warnings
    # The published metric.py contains `assert thr_eps, ...` on a NumPy array,
    # which raises "truth value of an array ... is ambiguous" whenever more than
    # one threshold survives, i.e. always. Patch only that line (in memory; the
    # reference copy stays untouched) so the rest of the official code can run.
    src = (REPO / "references/code/unlearning_evaluation/metric.py").read_text()
    bug = 'assert thr_eps, "Something went wrong'
    assert src.count(bug) == 1, "official metric.py changed; review the patch"
    src = src.replace(bug, 'assert len(thr_eps), "Something went wrong')
    official = types.ModuleType("official_metric")
    exec(compile(src, "official_metric.py", "exec"), official.__dict__)

    rng = np.random.default_rng(0)
    n_models, n_examples = 32, 60
    ok = True
    for shift in [0.0, 0.3, 1.0, 3.0, 10.0]:
        r = rng.normal(2.0, 1.0, (n_models, n_examples))
        u = rng.normal(2.0 + shift, 1.0, (n_models, n_examples))
        u[:, :5] = 7.0  # a few point-mass examples (epsilon cap path)
        ours = compute_forget_score_from_confs(u, r)
        ref = official.compute_forget_score_from_confs(u, r)
        match = abs(ours - ref) < 1e-9
        ok &= match
        print(f"shift={shift:5.1f}  ours={ours:.6f}  official={ref:.6f}  {'OK' if match else 'MISMATCH'}")

    logits = rng.normal(size=(100, 10)).astype(np.float32)
    targets = rng.integers(0, 10, 100)
    a = compute_logit_scaled_confidence(logits.copy(), targets)[1]
    b = official.compute_logit_scaled_confidence(logits.copy(), targets)[1]
    conf_ok = np.allclose(a, b)
    ok &= conf_ok
    print("logit-scaled confidence:", "OK" if conf_ok else "MISMATCH")
    print("METRIC PORT", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
