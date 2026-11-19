"""Baseline: SalUn - saliency unlearning (Fan et al., ICLR 2024).

1. Saliency map: accumulate |grad| of the cross-entropy loss over the forget set.
2. Mask = the top ``threshold`` fraction of weights (global, across all
   parameters) by gradient magnitude.
3. Random-label training (see :mod:`random_label`) where only masked weights
   are updated.

Defaults follow the SalUn paper's random-forgetting setup for CIFAR-10 (50%
sparsity mask) with untuned epochs / lr.
"""

import torch
import torch.nn.functional as F

from ..device import DEVICE
from ..utils import merge_hparams
from .random_label import relabel_train

INFO = {"rank": None, "team": "baseline (Fan et al. 2024)", "family": "saliency mask + relabel"}

DEFAULTS = {"epochs": 5, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "batch_size": 256,
            "threshold": 0.5}


def saliency_mask(net, forget, threshold):
    net.eval()
    net.zero_grad()
    for x, y in forget:
        x, y = x.to(DEVICE), y.to(DEVICE)
        F.cross_entropy(net(x), y).backward()
    params = [p for p in net.parameters() if p.grad is not None]
    scores = torch.cat([p.grad.abs().flatten() for p in params])
    k = max(1, int(len(scores) * threshold))
    cutoff = torch.topk(scores, k, sorted=True).values[-1]
    mask = {p: (p.grad.abs() >= cutoff).float() for p in params}
    net.zero_grad()
    return mask


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "salun")
    net.to(DEVICE)
    mask = saliency_mask(net, forget, hp["threshold"])
    rl_hp = {k: hp[k] for k in ("epochs", "lr", "momentum", "weight_decay", "batch_size")}
    return relabel_train(net, retain, forget, rl_hp, grad_mask=mask)
