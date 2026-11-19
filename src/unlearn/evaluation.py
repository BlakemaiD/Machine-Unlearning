"""Run artifacts and aggregate evaluation (unlearned runs vs. retrained oracle runs).

Each run (one unlearned model, or one retrained-from-scratch model) is stored as
an ``.npz`` file holding the model's logits on the fixed-order forget, retain,
test and validation sets, plus cost information. All metrics are computed from
these files, so evaluation can be repeated or extended without re-running any
training.
"""

import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .forget_quality import (competition_score, compute_forget_score_from_confs,
                             compute_logit_scaled_confidence)

SPLITS = ("forget", "retain", "test", "val")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def collect_outputs(net, loaders):
    """Logits/targets of ``net`` on every split, using the fixed-order loaders."""
    names = {"forget": "forget_eval", "retain": "retain_eval", "test": "test", "val": "val"}
    out = {}
    for split, key in names.items():
        logits, targets = M.predict(net, loaders[key])
        out[f"{split}_logits"] = logits.astype(np.float16)
        out[f"{split}_targets"] = targets.astype(np.uint8)
    return out


def save_run(path, outputs, meta):
    """Write one run: arrays to ``<path>.npz``, metadata (hparams, cost...) inside as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **outputs)


def load_run(path):
    with np.load(path, allow_pickle=False) as z:
        run = {k: z[k] for k in z.files if k != "meta"}
        run["meta"] = json.loads(str(z["meta"]))
    # A diverged run (NaN/inf weights) is kept and scored, not dropped: its
    # non-finite logits become 0 (a uniform prediction) and it is flagged.
    run["diverged"] = False
    for split in SPLITS:
        logits = run[f"{split}_logits"].astype(np.float32)
        if not np.isfinite(logits).all():
            run["diverged"] = True
            logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        run[f"{split}_logits"] = logits
        run[f"{split}_targets"] = run[f"{split}_targets"].astype(np.int64)
    return run


def load_runs(directory, pattern="run_*.npz"):
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} files in {directory}")
    return [load_run(f) for f in files]


# ---------------------------------------------------------------------------
# Per-run and aggregate metrics
# ---------------------------------------------------------------------------

def run_metrics(run, seed=0):
    """Scalar metrics for one model that do not need the retrained oracle."""
    r = {}
    for split in SPLITS:
        r[f"{split}_acc"] = M.acc_from_logits(run[f"{split}_logits"], run[f"{split}_targets"])
    f_loss = M.losses_from_logits(run["forget_logits"], run["forget_targets"])
    t_loss = M.losses_from_logits(run["test_logits"], run["test_targets"])
    r["forget_loss"] = float(f_loss.mean())
    r["test_loss"] = float(t_loss.mean())
    r["mia_forget_vs_test"] = M.mia_from_losses(f_loss, t_loss, seed=seed)
    r["mia_efficacy"] = M.mia_efficacy(
        run["retain_logits"], run["retain_targets"], run["test_logits"], run["test_targets"],
        run["forget_logits"], run["forget_targets"], seed=seed)
    meta = run.get("meta", {})
    r["seconds"] = float(meta.get("seconds", np.nan))
    r["peak_mem_mb"] = float(meta.get("peak_mem_mb", np.nan))
    return r


def _mean_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(v)), float(np.nanstd(v))


def summarize(unlearned_runs, retrained_runs, retrained_per_run=None):
    """Aggregate metrics for one method.

    Args:
        unlearned_runs: list of run dicts (N unlearned models).
        retrained_runs: list of run dicts (N retrained-from-scratch models).
        retrained_per_run: optional precomputed ``run_metrics`` for the
            retrained runs (avoids recomputing them for every method).
    Returns:
        (summary dict, per-example epsilon array)
    """
    u_per = [run_metrics(r, seed=i) for i, r in enumerate(unlearned_runs)]
    r_per = retrained_per_run or [run_metrics(r, seed=i) for i, r in enumerate(retrained_runs)]

    s = {"n_unlearned": len(unlearned_runs), "n_retrained": len(retrained_runs),
         "n_diverged": int(sum(bool(r.get("diverged", False)) for r in unlearned_runs))}
    keys = ["retain_acc", "forget_acc", "test_acc", "val_acc", "forget_loss", "test_loss",
            "mia_forget_vs_test", "mia_efficacy", "seconds", "peak_mem_mb"]
    for k in keys:
        s[k], s[k + "_std"] = _mean_std([p[k] for p in u_per])
        s["retrain_" + k], _ = _mean_std([p[k] for p in r_per])

    # --- gaps to the oracle (lower is better) --------------------------------
    for k in ["retain_acc", "forget_acc", "test_acc", "mia_forget_vs_test", "mia_efficacy"]:
        s["gap_" + k] = abs(s[k] - s["retrain_" + k])
    s["avg_gap"] = float(np.mean([s["gap_retain_acc"], s["gap_forget_acc"],
                                  s["gap_test_acc"], s["gap_mia_efficacy"]]))

    # --- official competition metric --------------------------------------
    u_conf = np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                       for r in unlearned_runs])
    r_conf = np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                       for r in retrained_runs])
    n = min(len(u_conf), len(r_conf))
    if n >= 3:
        f_score, eps = compute_forget_score_from_confs(u_conf[:n], r_conf[:n], return_eps=True)
        s["forget_quality_F"] = f_score
        s["competition_score"] = competition_score(
            f_score, [p["retain_acc"] for p in u_per], [p["test_acc"] for p in u_per],
            [p["retain_acc"] for p in r_per], [p["test_acc"] for p in r_per])
        s["eps_median"] = float(np.median(eps))
        s["eps_frac_below_0.5"] = float((eps < 0.5).mean())
    else:
        eps = np.full(u_conf.shape[1], np.nan)
        s["forget_quality_F"] = s["competition_score"] = np.nan

    # --- output-distribution match to retrained models (paired i <-> i mod N_r) --
    for split in ("forget", "test"):
        js, kl, agree, act = [], [], [], []
        for i, u in enumerate(unlearned_runs):
            r = retrained_runs[i % len(retrained_runs)]
            js.append(M.js_divergence(u[f"{split}_logits"], r[f"{split}_logits"]))
            kl.append(M.kl_divergence(r[f"{split}_logits"], u[f"{split}_logits"]))
            agree.append(M.prediction_agreement(u[f"{split}_logits"], r[f"{split}_logits"]))
            act.append(M.activation_distance(u[f"{split}_logits"], r[f"{split}_logits"]))
        s[f"js_{split}"] = float(np.mean(js))
        s[f"kl_retrain_to_unlearned_{split}"] = float(np.mean(kl))
        s[f"agreement_{split}"] = float(np.mean(agree))
        s[f"activation_dist_{split}"] = float(np.mean(act))

    # --- efficiency ---------------------------------------------------------
    s["rte_speedup_vs_retrain"] = (s["retrain_seconds"] / s["seconds"]
                                   if s["seconds"] and not np.isnan(s["seconds"]) else np.nan)
    return s, eps
