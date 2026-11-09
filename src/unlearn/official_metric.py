"""CPU implementation of the NeurIPS unlearning challenge forgetting metric."""

import numpy as np

DELTA = 1e-5
EPSILON_CAP = 50.0
NUM_THRESHOLDS_PER_UNIT = 100
BUCKET_WIDTH = 0.5


def logit_scaled_confidence(logits, targets):
    """Return log(p(correct) / p(wrong)) for each example."""
    return compute_logit_scaled_confidence(logits, targets)[1]


def compute_logit_scaled_confidence(logits, targets):
    """Official metric API: return probabilities of correctness and confidences."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.size:
        raise ValueError("logits must be [examples, classes] and targets [examples].")
    if logits.shape[1] < 2 or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("Targets must index one of at least two output classes.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    rows = np.arange(targets.size)
    p_correct = probs[rows, targets].copy()
    probs[rows, targets] = 0.0
    p_wrong = np.sum(probs, axis=1)
    confidence = np.log(p_correct + 1e-45) - np.log(p_wrong + 1e-45)
    return p_correct, confidence


def _window_rate(confidences, left, right, chunk_size=8192):
    rates = np.empty(left.size, dtype=np.float64)
    for start in range(0, left.size, chunk_size):
        stop = min(start + chunk_size, left.size)
        values = confidences[:, None]
        rates[start:stop] = np.mean(
            (values >= left[None, start:stop])
            & (values <= right[None, start:stop]),
            axis=0,
            dtype=np.float64,
        )
    return rates


def _double_threshold_rates(pos, neg):
    pos_diff, neg_diff = np.ptp(pos), np.ptp(neg)
    width = min(pos_diff, neg_diff)
    if pos_diff >= neg_diff:
        pos, neg = neg, pos

    buffer = 2.0
    lo = pos.min() + width - buffer
    hi = pos.max() + buffer
    num_right = max(
        int(np.ceil((hi - lo) * NUM_THRESHOLDS_PER_UNIT)),
        2,
    )
    right = np.linspace(lo, hi, num_right, dtype=np.float64)
    num_left = int(np.ceil(2 * buffer * NUM_THRESHOLDS_PER_UNIT))
    left = np.concatenate(
        [
            np.linspace(r - width - buffer, r - width + buffer, num_left)
            for r in right
        ]
    )
    right = np.repeat(right, num_left)
    tpr = _window_rate(pos, left, right)
    fpr = _window_rate(neg, left, right)
    return tpr, 1.0 - tpr, fpr, 1.0 - fpr


def _single_threshold_rates(pos, neg):
    values = np.concatenate((pos, neg))
    lo, hi = values.min(), values.max()
    count = max(
        int(np.ceil((hi - lo) * NUM_THRESHOLDS_PER_UNIT)),
        2,
    )
    thresholds = np.linspace(lo, hi, count, dtype=np.float64)
    tpr = np.empty(count, dtype=np.float64)
    fpr = np.empty(count, dtype=np.float64)
    for start in range(0, count, 8192):
        stop = min(start + 8192, count)
        current = thresholds[None, start:stop]
        tpr[start:stop] = np.mean(pos[:, None] >= current, axis=0, dtype=np.float64)
        fpr[start:stop] = np.mean(neg[:, None] >= current, axis=0, dtype=np.float64)
    return tpr, 1.0 - tpr, fpr, 1.0 - fpr


def _eps_from_rates(fnr, fpr, tpr, tnr, delta):
    perfect = ((fpr + fnr) == 0) & ((tpr + tnr) == 2)
    one_zero = ((fpr == 0) | (fnr == 0)) & ~perfect
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.log(1.0 - delta - fpr) - np.log(fnr)
        e2 = np.log(1.0 - delta - fnr) - np.log(fpr)
        eps = np.clip(np.fmax(e1, e2), 0.0, None)
    eps[one_zero] = np.nan
    eps[perfect] = np.inf
    return eps


def example_epsilon(pos, neg, delta=DELTA):
    """Estimate the strongest distinguishing attack for one forget example."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.ndim != 1 or neg.ndim != 1 or pos.size != neg.size or pos.size < 2:
        raise ValueError("pos and neg must be equal-length vectors with at least two models.")

    pos_diff, neg_diff = np.ptp(pos), np.ptp(neg)
    largest, smallest = max(pos_diff, neg_diff), min(pos_diff, neg_diff)
    if largest == 0.0 or smallest / largest < 0.01:
        return EPSILON_CAP

    epsilons = []
    for tpr, fnr, fpr, tnr in (
        _double_threshold_rates(pos, neg),
        _single_threshold_rates(pos, neg),
    ):
        eps = _eps_from_rates(fnr, fpr, tpr, tnr, delta)
        epsilons.append(eps)

    eps = np.concatenate(epsilons)
    eps = eps[~np.isnan(eps)]
    if eps.size == 0:
        return EPSILON_CAP
    return float(np.clip(np.max(eps), 0.0, EPSILON_CAP))


def forget_score(unlearned_confs, retrained_confs):
    """Compute official forgetting quality F from [models, forget examples] arrays."""
    unlearned = np.asarray(unlearned_confs, dtype=np.float64)
    retrained = np.asarray(retrained_confs, dtype=np.float64)
    if unlearned.ndim != 2 or retrained.ndim != 2:
        raise ValueError("Confidence arrays must have shape [models, forget examples].")
    if unlearned.shape != retrained.shape or unlearned.shape[0] < 2:
        raise ValueError("Confidence arrays must have equal shape and at least two models.")
    if unlearned.shape[1] == 0:
        raise ValueError("At least one forget example is required.")

    retrain_medians = np.median(retrained, axis=0)
    unlearned_medians = np.median(unlearned, axis=0)
    retrain_is_positive = retrain_medians > unlearned_medians
    positive = np.where(retrain_is_positive, retrained, unlearned)
    negative = np.where(retrain_is_positive, unlearned, retrained)
    epsilons = np.array(
        [
            example_epsilon(positive[:, i], negative[:, i])
            for i in range(unlearned.shape[1])
        ],
        dtype=np.float64,
    )

    max_epsilon = np.ceil(np.log(unlearned.shape[0] - 1))
    buckets = []
    start, points = 0.0, 1.0
    while start + BUCKET_WIDTH <= max_epsilon:
        buckets.append((start, points))
        points /= 2.0
        start += BUCKET_WIDTH

    score = 0.0
    for epsilon in epsilons:
        for start, points in buckets:
            if epsilon < start + BUCKET_WIDTH:
                score += points
                break
    return float(score / unlearned.shape[1])


compute_forget_score_from_confs = forget_score
