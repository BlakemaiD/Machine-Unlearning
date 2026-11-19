"""Single-model metrics: accuracy, losses, membership-inference attacks, divergences.

Everything except :func:`accuracy`/:func:`compute_losses`/:func:`mia_accuracy`
(the starter-kit API, kept for the notebooks) works on *saved logits*, so
metrics can be recomputed offline from the runner's ``.npz`` artifacts without
reloading models. The multi-model competition score lives in
:mod:`unlearn.forget_quality`; aggregation across runs in :mod:`unlearn.evaluation`.

Metric families (see document.md, "Metrics", for interpretation):

* Utility: retain / forget / test / val accuracy (RA, FA, TA, VA).
* Privacy (attack-based):
    - ``simple_mia``/``mia_accuracy``: starter-kit loss-threshold attack,
      forget vs. test, logistic regression with 10-fold CV. Ideal ~ the value
      obtained on retrained models (about 0.5).
    - ``mia_efficacy``: attack trained to separate retain (member) from test
      (non-member) using entropy/confidence features, then applied to the
      forget set; reports the fraction of forget examples predicted
      *non-member* (Jia et al. 2023, Fan et al. 2024 "MIA-Efficacy").
* Distribution match to the retrained oracle (needs retrained logits):
    - ``js_divergence`` / ``kl_divergence`` of predictive distributions,
    - ``prediction_agreement`` (top-1 agreement),
    - ``activation_distance`` (mean L2 between softmax outputs; Chundawat et al.).
"""

import numpy as np
import torch
from scipy.special import log_softmax, softmax
from sklearn import linear_model, model_selection
from torch import nn

from .device import DEVICE


# ---------------------------------------------------------------------------
# Model forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(net, loader):
    """Return ``(logits [N, C] float32 numpy, targets [N] int64 numpy)`` in loader order."""
    net.to(DEVICE)
    was_training = net.training
    net.eval()
    all_logits, all_targets = [], []
    for inputs, targets in loader:
        all_logits.append(net(inputs.to(DEVICE)).float().cpu())
        all_targets.append(targets)
    if was_training:
        net.train()
    return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()


@torch.no_grad()
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    net.to(DEVICE)
    net.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


@torch.no_grad()
def compute_losses(net, loader):
    """Per-sample cross-entropy losses."""
    net.to(DEVICE)
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        losses = criterion(net(inputs), targets).numpy(force=True)
        all_losses.extend(losses)
    return np.array(all_losses)


# ---------------------------------------------------------------------------
# Logit-based helpers
# ---------------------------------------------------------------------------

def acc_from_logits(logits, targets):
    return float((np.asarray(logits).argmax(1) == np.asarray(targets)).mean())


def losses_from_logits(logits, targets):
    logp = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return -logp[np.arange(len(targets)), np.asarray(targets)]


def entropy_from_logits(logits):
    logp = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return -(np.exp(logp) * logp).sum(1)


# ---------------------------------------------------------------------------
# Membership inference attacks
# ---------------------------------------------------------------------------

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def mia_from_losses(forget_losses, test_losses, seed=0):
    """Starter-kit forget-vs-test MIA accuracy from per-example losses (class-balanced)."""
    f = np.asarray(forget_losses)
    t = np.asarray(test_losses)
    rng = np.random.default_rng(seed)
    if len(f) > len(t):
        f = rng.permutation(f)[: len(t)]
    elif len(t) > len(f):
        t = rng.permutation(t)[: len(f)]
    X = np.concatenate((t, f)).reshape(-1, 1)
    y = [0] * len(t) + [1] * len(f)
    return float(simple_mia(X, y).mean())


def mia_accuracy(net, forget_loader, test_loader, seed=0):
    """Forget-vs-test MIA accuracy, class-balanced. Replaces the repeated notebook cells."""
    return mia_from_losses(compute_losses(net, forget_loader),
                           compute_losses(net, test_loader), seed=seed)


def _mia_features(logits, targets):
    """[N, 3] features: loss, prediction entropy, max confidence."""
    p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return np.stack([losses_from_logits(logits, targets),
                     entropy_from_logits(logits),
                     p.max(1)], axis=1)


def mia_efficacy(retain_logits, retain_targets, test_logits, test_targets,
                 forget_logits, forget_targets, seed=0):
    """MIA-Efficacy: fraction of forget examples an attack labels *non-member*.

    The attack (logistic regression on loss/entropy/confidence) is fit on a
    class-balanced sample of retain (members) vs test (non-members) under the
    same model, then applied to the forget set. Higher = more forgetting; the
    reference value is that of the retrained model (forget data truly unseen).
    """
    rng = np.random.default_rng(seed)
    n = min(len(retain_targets), len(test_targets))
    r_idx = rng.choice(len(retain_targets), n, replace=False)
    t_idx = rng.choice(len(test_targets), n, replace=False)
    X = np.concatenate([_mia_features(retain_logits[r_idx], retain_targets[r_idx]),
                        _mia_features(test_logits[t_idx], test_targets[t_idx])])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit((X - mu) / sd, y)
    pred = clf.predict((_mia_features(forget_logits, forget_targets) - mu) / sd)
    return float(1.0 - pred.mean())


# ---------------------------------------------------------------------------
# Distribution match to the retrained oracle
# ---------------------------------------------------------------------------

def kl_divergence(p_logits, q_logits):
    """Mean KL(p || q) between predictive distributions (rows = examples)."""
    lp = log_softmax(np.asarray(p_logits, dtype=np.float64), axis=1)
    lq = log_softmax(np.asarray(q_logits, dtype=np.float64), axis=1)
    return float((np.exp(lp) * (lp - lq)).sum(1).mean())


def js_divergence(p_logits, q_logits):
    """Mean Jensen-Shannon divergence (natural log, in [0, ln 2])."""
    p = softmax(np.asarray(p_logits, dtype=np.float64), axis=1)
    q = softmax(np.asarray(q_logits, dtype=np.float64), axis=1)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(1)

    return float((0.5 * _kl(p, m) + 0.5 * _kl(q, m)).mean())


def prediction_agreement(p_logits, q_logits):
    """Fraction of examples where both models predict the same class."""
    return float((np.asarray(p_logits).argmax(1) == np.asarray(q_logits).argmax(1)).mean())


def activation_distance(p_logits, q_logits):
    """Mean L2 distance between softmax outputs."""
    p = softmax(np.asarray(p_logits, dtype=np.float64), axis=1)
    q = softmax(np.asarray(q_logits, dtype=np.float64), axis=1)
    return float(np.linalg.norm(p - q, axis=1).mean())
