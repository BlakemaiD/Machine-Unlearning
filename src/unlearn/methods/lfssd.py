"""SOTA: Loss-Free Selective Synaptic Dampening, LFSSD (Foster et al., ICLR 2024 Tiny Paper).

SSD with the Fisher importance replaced by a label-free sensitivity: the
mean |d ||f(x)||_2^2 / d theta| over batches (Aljundi et al., MAS). The same
selection (I_f > alpha * I_D) and dampening (theta *= min(lambda I_D / I_f, 1))
rule as SSD; no training, no labels for the forget set.
Defaults: alpha = 10, lambda = 1 (the paper's ResNet-18 setting).
Source: references/papers/2024_Foster_LFSSD_LossFreeUnlearning_2402.19308.pdf (Eq. 1-3).
"""

from ..utils import merge_hparams
from . import ssd

INFO = {"rank": None, "team": "SOTA (Foster et al. 2024)", "family": "label-free parameter dampening (no training)"}

DEFAULTS = {"alpha": 10.0, "lam": 1.0}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "lfssd")
    return ssd.unlearning(net, retain, forget, validation, loss_free=True, **hp)
