"""Baseline: Selective Synaptic Dampening, SSD (Foster et al., AAAI 2024).

Retraining-free. Diagonal Fisher importances (mean squared gradients) are
computed on the full training data D (= retain + forget) and on the forget set
D_f. Parameters that are disproportionately important for the forget set,
``I_f > alpha * I_D``, are dampened: ``theta *= min(lambda * I_D / I_f, 1)``.

Defaults are the paper's ResNet-18 / CIFAR settings (alpha=10, lambda=1).
"""

import torch
import torch.nn.functional as F

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "baseline (Foster et al. 2024)", "family": "parameter dampening (no training)"}

DEFAULTS = {"alpha": 10.0, "lam": 1.0, "loss_free": False}


def _importance(net, loaders, loss_free=False):
    """Per-parameter importance, averaged over batches.

    SSD: squared gradient of the cross-entropy (diagonal Fisher).
    LFSSD (``loss_free``): |gradient| of the squared L2 norm of the output
    (Aljundi et al.'s MAS sensitivity), which needs no labels.
    """
    imp = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    batches = 0
    for loader in loaders:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            net.zero_grad()
            out = net(x)
            if loss_free:
                out.pow(2).sum(1).mean().backward()
            else:
                F.cross_entropy(out, y).backward()
            for n, p in net.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    imp[n] += g.abs() if loss_free else g ** 2
            batches += 1
    return {n: v / max(batches, 1) for n, v in imp.items()}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "ssd")
    net.to(DEVICE)
    net.eval()  # importances are computed in eval mode (BN statistics frozen)
    imp_full = _importance(net, [retain, forget], hp["loss_free"])
    imp_forget = _importance(net, [forget], hp["loss_free"])
    with torch.no_grad():
        for n, p in net.named_parameters():
            sel = imp_forget[n] > hp["alpha"] * imp_full[n]
            beta = torch.clamp(hp["lam"] * imp_full[n] / (imp_forget[n] + 1e-12), max=1.0)
            p[sel] *= beta[sel]
    net.zero_grad()
    net.eval()
    return net
