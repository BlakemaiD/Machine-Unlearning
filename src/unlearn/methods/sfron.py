"""SOTA: SFR-on (Huang et al., NeurIPS 2024 spotlight, "Unified Gradient-Based Machine
Unlearning with Remain Geometry Enhancement").

* Saliency mask: diagonal Fisher on forget vs retain; a weight is updated by
  the forget step only if ``F_forget / F_retain >= threshold`` (gamma = 1).
* Every ``forget_freq``-th iteration: *adaptive* gradient ascent on a forget
  batch - per-sample CE weighted by 1 / CE**lambda (normalised), scaled by
  ``forget_alpha`` with cosine decay over training, masked, grad-norm clipped.
* Every iteration: a cross-entropy step on a retain batch.
* SGD with cosine LR decay over ``n_iters`` iterations.

Defaults are the reference class defaults, used by the authors' CIFAR-10,
10% random, ResNet-18 script
(references/code/K1nght_Unified-Unlearning-w-Remain-Geometry/Classification/unlearn/sfron.py,
scripts/unlearn.sh): lr 0.01, momentum 0.9, wd 5e-4, 1500 iterations, batch
128, forget every 5 steps, alpha 25, max grad norm 7, lambda 0.5, gamma 1.
The reference's EMA step is a no-op with its shipped ema_beta = 1.0, so it is
omitted.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Huang et al. 2024)", "family": "masked adaptive ascent + retain descent"}

DEFAULTS = {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "n_iters": 1500, "batch_size": 128,
            "forget_freq": 5, "forget_alpha": 25.0, "max_norm": 7.0, "lambd": 0.5, "threshold": 1.0}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _fisher(net, loader):
    out = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    net.eval()
    for x, y in loader:
        net.zero_grad()
        F.cross_entropy(net(x.to(DEVICE)), y.to(DEVICE)).backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                out[n] += p.grad.detach() ** 2 / len(loader)
    net.zero_grad()
    return out


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "sfron")
    net.to(DEVICE)
    bs = hp["batch_size"]
    forget_ld = DataLoader(forget.dataset, batch_size=bs, shuffle=True)
    retain_ld = DataLoader(retain.dataset, batch_size=bs, shuffle=True)

    f_fisher, r_fisher = _fisher(net, forget_ld), _fisher(net, retain_ld)
    mask = {n: ((f_fisher[n] + 1e-15) / (r_fisher[n] + 1e-15) >= hp["threshold"]).float()
            for n in f_fisher}
    del f_fisher, r_fisher

    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                                weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["n_iters"])
    f_it, r_it = _cycle(forget_ld), _cycle(retain_ld)
    n = hp["n_iters"]
    for step in range(n):
        net.train()
        if step % hp["forget_freq"] == 0:
            alpha = hp["forget_alpha"] * (1 + math.cos(math.pi * step / n)) / 2
            x, y = next(f_it)
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = net(x)
            ce = F.cross_entropy(out, y, reduction="none")
            coef = 1.0 / (ce.detach().pow(hp["lambd"]) + 1e-15)
            adaptive = ((coef / coef.sum()) * ce * out.shape[0]).mean()
            optimizer.zero_grad()
            (-alpha * adaptive).backward()
            for name, p in net.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(mask[name])
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=hp["max_norm"])
            optimizer.step()
        x, y = next(r_it)
        optimizer.zero_grad()
        F.cross_entropy(net(x.to(DEVICE)), y.to(DEVICE)).backward()
        optimizer.step()
        scheduler.step()
    net.eval()
    return net
