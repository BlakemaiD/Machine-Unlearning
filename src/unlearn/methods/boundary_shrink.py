"""SOTA: Boundary Shrink (Chen et al., CVPR 2023, "Boundary Unlearning").

For each forget batch, a single FGSM step (L_inf bound 0.1 in pixel space)
against a frozen copy of the original model finds each sample's *nearest
incorrect* class (the prediction on the adversarial image). The model is then
fine-tuned on the clean forget images with those labels, so the forget
samples' decision region is absorbed by neighbouring classes instead of
being scattered randomly (as with random labels).

Defaults from the reference implementation
(references/code/airoca_Boundary-Unlearning/Boundary_Unlearning.ipynb):
FGSM bound 0.1, SGD lr 1e-5, momentum 0.9, 10 passes over the forget set.
The method was designed for whole-class forgetting; here it is applied to a
random forget set, as in later benchmarks. Inputs are de-normalised for the
attack and re-normalised afterwards (CIFAR-10 statistics, or none for SURF).
"""

import copy

import torch
import torch.nn.functional as F

from ..data import CIFAR10_MEAN, CIFAR10_STD
from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Chen et al. 2023)", "family": "nearest-wrong-class relabel (adversarial)"}

DEFAULTS = {"bound": 0.1, "lr": 1e-5, "momentum": 0.9, "epochs": 10, "normalization": "cifar10"}


def norm_stats(mode):
    if mode == "cifar10":
        return (torch.tensor(CIFAR10_MEAN, device=DEVICE).view(1, 3, 1, 1),
                torch.tensor(CIFAR10_STD, device=DEVICE).view(1, 3, 1, 1))
    return None, None  # raw-pixel datasets (SURF): no normalisation


def fgsm_nearest_label(model, x, y, bound, mean, std):
    """Prediction of ``model`` on a one-step FGSM image of ``x`` (pixel-space L_inf ``bound``)."""
    x = x.detach().clone().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    grad_sign = x.grad.sign()
    with torch.no_grad():
        if mean is None:  # raw 0-255 pixels
            adv = (x + grad_sign * bound * 255).clamp(0, 255)
        else:
            pix = x * std + mean
            pix = torch.clamp(pix + grad_sign * bound, 0, 1)
            pix = torch.round(pix * 255) / 255  # discretise, as in the reference attack
            adv = (pix - mean) / std
        return model(adv).argmax(1)


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "boundary_shrink")
    net.to(DEVICE)
    frozen = copy.deepcopy(net).eval()
    mean, std = norm_stats(hp["normalization"])
    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"])
    for _ in range(hp["epochs"]):
        for x, y in forget:
            x, y = x.to(DEVICE), y.to(DEVICE)
            frozen.zero_grad()
            nearest = fgsm_nearest_label(frozen, x, y, hp["bound"], mean, std)
            net.train()
            optimizer.zero_grad()
            F.cross_entropy(net(x), nearest).backward()
            optimizer.step()
    net.eval()
    return net
