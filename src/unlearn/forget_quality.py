"""Official NeurIPS 2023 Machine Unlearning forgetting-quality metric.

Port of ``metric.py`` from https://github.com/google-deepmind/unlearning_evaluation
(Apache-2.0; copy kept in ``references/code/unlearning_evaluation``). The maths
and constants are unchanged. Local changes:

* runs on :data:`unlearn.device.DEVICE` instead of hard-coded ``.cuda()``;
* :func:`compute_forget_score_from_confs` can also return the per-example
  epsilons (useful for plots / per-example analysis);
* :func:`competition_score` wraps the utility adjustment from ``main.py``.

How it works (Triantafillou et al. 2024, arXiv:2406.09073, Sec. 3):
for every forget example, the logit-scaled confidence under N unlearned models
and N retrained-from-scratch models forms two empirical distributions. A family
of single- and double-threshold attacks tries to tell them apart; the best
attack's FPR/FNR give an (epsilon, delta=1e-5) estimate. Each example scores
``2 / 2**floor(eps / 0.5)`` points (1 point if eps < 0.5, 0.5 if < 1.0, ...);
the forgetting quality F is the mean over the forget set. The final competition
score is ``F * (RA_u / RA_r) * (TA_u / TA_r)``.
"""

import numpy as np
import torch

from .device import DEVICE

DELTA = 1e-05
EPSILON_CAP = 50
NUM_THRESHOLDS_PER_UNIT = 100


def compute_logit_scaled_confidence(logits, targets):
    """Transform logits into logit-scaled confidences: log p_y - log(1 - p_y).

    Numerically-stable version adapted (by the organisers) from the LiRA codebase.
    Returns ``(prob_correct, conf)``.
    """
    logits = np.array(logits, dtype=np.float64)
    targets = np.asarray(targets)
    count = logits.shape[0]
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    prob_correct = probs[np.arange(count), targets[:count]].copy()
    probs[np.arange(count), targets[:count]] = 0
    prob_wrong = np.sum(probs, axis=1)
    conf = np.log(prob_correct + 1e-45) - np.log(prob_wrong + 1e-45)
    return prob_correct, conf


def _get_double_threshold_rates(pos_confs_, neg_confs_):
    """tpr, fnr, fpr, tnr for a two-threshold rule (inside -> peakier distribution)."""
    dev = pos_confs_.device
    pos_diff = pos_confs_.max() - pos_confs_.min()
    neg_diff = neg_confs_.max() - neg_confs_.min()
    smallest = neg_diff if pos_diff > neg_diff else pos_diff

    # For simplicity, define 'positive' as the peakiest distribution.
    if pos_diff >= neg_diff:
        pos_confs_, neg_confs_ = neg_confs_, pos_confs_

    buffer = 2.0
    width = smallest
    min_ = (pos_confs_.min() + width - buffer).detach().cpu().numpy()
    max_ = (pos_confs_.max() + buffer).detach().cpu().numpy()
    num_right_thresholds = int(np.ceil((max_ - min_) * NUM_THRESHOLDS_PER_UNIT))
    right_thresholds_ = np.linspace(min_, max_, num=num_right_thresholds)
    width = width.detach().cpu().numpy()

    num_left_thresholds = int(np.ceil(2 * buffer * NUM_THRESHOLDS_PER_UNIT))
    thr_left_ = np.concatenate(
        [np.linspace(tr - width - buffer, tr - width + buffer, num=num_left_thresholds)
         for tr in right_thresholds_],
        axis=0,
    )
    thr_right_ = np.tile(
        np.reshape(right_thresholds_, (-1, 1)), (1, num_left_thresholds)
    ).reshape((-1))
    total_num_thresholds = len(thr_left_)

    num_models = pos_confs_.shape[0]
    thr_right_ = torch.from_numpy(np.tile(thr_right_[None, :], (num_models, 1))).to(dev)
    thr_left_ = torch.from_numpy(np.tile(thr_left_[None, :], (num_models, 1))).to(dev)
    pos_confs_ = torch.tile(pos_confs_[:, None], (1, total_num_thresholds))
    neg_confs_ = torch.tile(neg_confs_[:, None], (1, total_num_thresholds))

    def _is_pp(c):
        return torch.logical_and(torch.le(thr_left_, c), torch.le(c, thr_right_)).float()

    tpr = torch.mean(_is_pp(pos_confs_), 0)
    fnr = 1.0 - tpr
    fpr = torch.mean(_is_pp(neg_confs_), 0)
    tnr = 1.0 - fpr
    return tpr, fnr, fpr, tnr


def _get_single_threshold_rates(pos_confs_, neg_confs_):
    """tpr, fnr, fpr, tnr for a one-threshold rule (right -> larger-median class)."""
    dev = pos_confs_.device
    all_confs = torch.cat([pos_confs_, neg_confs_])
    min_val, max_val = all_confs.min(), all_confs.max()
    num_thresholds = int(torch.ceil((max_val - min_val) * NUM_THRESHOLDS_PER_UNIT))
    thresholds_ = torch.linspace(min_val.item(), max_val.item(), num_thresholds,
                                 dtype=pos_confs_.dtype, device=dev)
    num_models = pos_confs_.shape[0]
    thresholds_ = torch.tile(thresholds_[None, :], (num_models, 1))
    pos_confs_ = torch.tile(pos_confs_[:, None], (1, num_thresholds))
    neg_confs_ = torch.tile(neg_confs_[:, None], (1, num_thresholds))
    tpr = torch.mean((pos_confs_ >= thresholds_).float(), 0)
    fpr = torch.mean((neg_confs_ >= thresholds_).float(), 0)
    tnr = torch.mean((neg_confs_ < thresholds_).float(), 0)
    fnr = torch.mean((pos_confs_ < thresholds_).float(), 0)
    return tpr, fnr, fpr, tnr


def _get_epsilons(pos_confs, neg_confs, delta, device=None):
    """Per-example epsilon estimates. pos/neg_confs: [num models, num examples]."""
    device = device or DEVICE
    epsilons = []
    num_examples = pos_confs.shape[1]
    for i in range(num_examples):
        pos_confs_ = pos_confs[:, i].reshape(-1)
        neg_confs_ = neg_confs[:, i].reshape(-1)

        pos_diff = np.max(pos_confs_) - np.min(pos_confs_)
        neg_diff = np.max(neg_confs_) - np.min(neg_confs_)
        largest = pos_diff if pos_diff > neg_diff else neg_diff
        smallest = neg_diff if pos_diff > neg_diff else pos_diff
        if largest == 0 or smallest / largest < 0.01:
            # One distribution is (almost) a point mass: treat as maximally
            # distinguishable, exactly as the official code does.
            epsilons.append(EPSILON_CAP)
            continue

        p = torch.from_numpy(pos_confs_).to(device)
        n = torch.from_numpy(neg_confs_).to(device)
        tpr_d, fnr_d, fpr_d, tnr_d = _get_double_threshold_rates(p, n)
        tpr_s, fnr_s, fpr_s, tnr_s = _get_single_threshold_rates(p, n)
        tpr = torch.cat([tpr_d, tpr_s]).double()
        fnr = torch.cat([fnr_d, fnr_s]).double()
        fpr = torch.cat([fpr_d, fpr_s]).double()
        tnr = torch.cat([tnr_d, tnr_s]).double()

        # Perfect separation (fpr = fnr = 0 and tpr = tnr = 1) -> eps = inf.
        perfect = (fpr + fnr == 0) & (tpr + tnr == 2)
        # Exactly one of fpr / fnr is zero -> discard threshold (insufficient samples).
        discard = (~perfect) & (fpr * fnr == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr_n, fnr_n = fpr.cpu().numpy(), fnr.cpu().numpy()
            eps = np.clip(
                np.nanmax([np.log(1 - delta - fpr_n) - np.log(fnr_n),
                           np.log(1 - delta - fnr_n) - np.log(fpr_n)], axis=0),
                0, None)
        eps[perfect.cpu().numpy()] = np.inf
        eps[discard.cpu().numpy()] = np.nan
        kept = eps[~np.isnan(eps)]
        assert kept.size, "Something went wrong, all thresholds gave nan epsilon..."
        epsilons.append(np.nanmax(kept))

    return np.clip(np.array(epsilons, dtype=np.float64), 0, EPSILON_CAP)


def compute_forget_score_from_confs(unlearned_confs, retrained_confs, return_eps=False,
                                    device=None):
    """Forgetting quality F from confidences.

    Args:
        unlearned_confs: [num models, num examples] logit-scaled confidences of
            each forget example under each unlearned model.
        retrained_confs: [num models, num examples] same under retrained models.
            Rows need not be paired, but columns (examples) must align.
        return_eps: also return the per-example epsilon array.
    """
    unlearned_confs = np.asarray(unlearned_confs, dtype=np.float64)
    retrained_confs = np.asarray(retrained_confs, dtype=np.float64)
    if unlearned_confs.shape[1] != retrained_confs.shape[1]:
        raise ValueError("unlearned and retrained confs must cover the same examples")
    _, forget_set_size = retrained_confs.shape

    retrain_medians = np.median(retrained_confs, axis=0)
    unlearned_medians = np.median(unlearned_confs, axis=0)
    # Positive class per example = whichever distribution has the larger median.
    retrain_is_positive = retrain_medians > unlearned_medians
    pos_confs = np.where(retrain_is_positive, retrained_confs, unlearned_confs)
    neg_confs = np.where(retrain_is_positive, unlearned_confs, retrained_confs)
    eps = _get_epsilons(pos_confs, neg_confs, delta=DELTA, device=device)

    # Max epsilon resolvable with this many models (official back-of-envelope).
    num_models = min(unlearned_confs.shape[0], retrained_confs.shape[0])
    max_epsilon = np.ceil(np.log(num_models - 1))

    start_idx, bucket_size, points = 0.0, 0.5, 1.0
    bucket_points = {}
    while start_idx + bucket_size <= max_epsilon:
        bucket_points[start_idx] = points
        points /= 2
        start_idx += bucket_size

    forget_score = 0.0
    for e in eps:
        for s, pts in bucket_points.items():
            if e < s + bucket_size:
                forget_score += pts
                break
    forget_score /= forget_set_size
    return (forget_score, eps) if return_eps else forget_score


def competition_score(forget_score, unlearned_retain_accs, unlearned_test_accs,
                      retrain_retain_accs, retrain_test_accs):
    """Final competition score = F * (RA_u / RA_r) * (TA_u / TA_r) (means over models)."""
    ra = np.mean(unlearned_retain_accs) / np.mean(retrain_retain_accs)
    ta = np.mean(unlearned_test_accs) / np.mean(retrain_test_accs)
    return forget_score * ra * ta
