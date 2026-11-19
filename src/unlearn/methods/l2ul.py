"""SOTA: L2UL - Learning to Unlearn, instance-wise (Cha et al., AAAI 2024).

Designed for forgetting individual training instances without retain data:

1. Adversarial examples: for each forget image, ``num_adv_images`` targeted
   L2-PGD examples toward random *other* labels (pixel-space ``eps``) from the
   original model. Training on them with their adversarial labels keeps the
   class boundaries around the forget samples intact.
2. Weight importance (MAS on the forget set: |grad of mean L2 norm of the
   output|), min-max normalised and inverted, so weights unimportant for the
   forget data are anchored to their original values.
3. Loop: ``-CE(forget) + CE(adversarial)`` (size-weighted) ``+ reg_lamb *
   sum(importance * (theta - theta_0)**2) / 2``; SGD momentum 0.9, wd 1e-4;
   stop when forget accuracy reaches ``stop_forget_acc`` or after
   ``max_epochs`` passes.

Reference: references/code/csm9493_L2UL (``main_unlearn_cifar10_mixed_label_resnet18.py``,
``run_unlearn_cifar_imagenet.sh``): lr 1e-3, reg_lamb 1.0, PGD eps 4.0,
step 0.1, and in the paper the loop runs until forget accuracy is 0.

ADAPTATIONS for a 5,000-image forget set (the paper uses at most 128):
the reference's 20 adversarial images x 100 PGD steps per forget image would
take hours per run, so defaults are 2 images x 10 steps (step 0.5) and at most
2 passes. The paper's stopping rule (forget accuracy 0) over-forgets relative
to the retrain oracle (~88% forget accuracy), so it is optional:
``stop_forget_acc=0.0`` reproduces it.
"""

import copy
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams
from .boundary_shrink import norm_stats

INFO = {"rank": None, "team": "SOTA (Cha et al. 2024)", "family": "instance-wise ascent + adversarial anchors + MAS",
        "note": "attack budget reduced for 5k forget images"}

DEFAULTS = {"lr": 1e-3, "reg_lamb": 1.0, "num_adv_images": 2, "pgd_eps": 4.0, "pgd_step": 0.5,
            "pgd_iter": 10, "max_epochs": 2, "stop_forget_acc": None, "batch_size": 128,
            "normalization": "cifar10"}


def _targeted_l2_pgd(model, x, target, eps, step, iters, mean, std):
    """Targeted L2 PGD in pixel space; returns normalised adversarial images."""
    pix = (x * std + mean) if mean is not None else x / 255.0
    delta = torch.zeros_like(pix, requires_grad=True)
    for _ in range(iters):
        adv = (pix + delta).clamp(0, 1)
        inp = (adv - mean) / std if mean is not None else adv * 255.0
        loss = F.cross_entropy(model(inp), target)
        g, = torch.autograd.grad(loss, delta)
        g = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        with torch.no_grad():
            delta -= step * g  # minimise CE toward the target label
            n = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            delta *= torch.clamp(eps / (n + 1e-12), max=1.0)
            delta.copy_((pix + delta).clamp(0, 1) - pix)
    adv = (pix + delta.detach()).clamp(0, 1)
    return ((adv - mean) / std if mean is not None else adv * 255.0).detach()


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "l2ul")
    net.to(DEVICE)
    k = get_classifier(net).out_features
    mean, std = norm_stats(hp["normalization"])
    forget_ld = DataLoader(forget.dataset, batch_size=hp["batch_size"], shuffle=True)

    # 1. adversarial anchors from the original model
    frozen = copy.deepcopy(net).eval()
    adv_x, adv_y = [], []
    for x, y in forget_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        for _ in range(hp["num_adv_images"]):
            t = torch.randint(0, k - 1, y.shape, device=DEVICE)
            t = t + (t >= y).long()  # random label != true label
            adv_x.append(_targeted_l2_pgd(frozen, x, t, hp["pgd_eps"], hp["pgd_step"],
                                          hp["pgd_iter"], mean, std).cpu())
            adv_y.append(t.cpu())
    adv_ld = DataLoader(TensorDataset(torch.cat(adv_x), torch.cat(adv_y)),
                        batch_size=hp["batch_size"], shuffle=True)
    del adv_x, adv_y, frozen

    # 2. MAS importance on the forget set, normalised and inverted
    theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    imp = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    net.train()
    n_seen = 0
    for x, _ in forget_ld:
        net.zero_grad()
        net(x.to(DEVICE)).norm(p=2, dim=1).mean().backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.abs() * len(x)
        n_seen += len(x)
    for n in imp:
        v = imp[n] / n_seen
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        imp[n] = 1 - v
    net.zero_grad()

    # 3. unlearning loop
    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=0.9, weight_decay=1e-4)
    forget_eval = DataLoader(forget.dataset, batch_size=512, shuffle=False)
    for _ in range(hp["max_epochs"]):
        for (xa, ya), (xf, yf) in zip(adv_ld, cycle(forget_ld)):
            net.train()
            xa, ya, xf, yf = xa.to(DEVICE), ya.to(DEVICE), xf.to(DEVICE), yf.to(DEVICE)
            na, nf = len(xa), len(xf)
            loss = (-F.cross_entropy(net(xf), yf) * nf / (na + nf)
                    + F.cross_entropy(net(xa), ya) * na / (na + nf))
            reg = sum((imp[n] * (p - theta0[n]).pow(2)).sum() for n, p in net.named_parameters()) / 2
            optimizer.zero_grad()
            (loss + hp["reg_lamb"] * reg).backward()
            optimizer.step()
        if hp["stop_forget_acc"] is not None:
            net.eval()
            with torch.no_grad():
                hits = sum((net(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
                           for x, y in forget_eval)
            if hits / len(forget.dataset) <= hp["stop_forget_acc"]:
                break
    net.eval()
    return net
