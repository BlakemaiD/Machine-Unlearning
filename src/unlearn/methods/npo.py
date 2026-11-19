"""SOTA: NPO - Negative Preference Optimization (Zhang et al., COLM 2024), "NPO + GD" variant.

NPO replaces gradient ascent's unbounded ``-CE(forget)`` with a bounded loss
that stops pushing once the model is far below the reference model:

    L_NPO = -(2 / beta) * mean log sigmoid( beta * (CE_theta(x, y) - CE_ref(x, y)) )

on forget data, with the original model as reference ("fine_tuned" policy).
The retain term is plain cross-entropy (the paper's NPO+GD / "grad_diff"):
``loss = npo_coeff * L_NPO + retain_coeff * CE(retain)``.

Loss exactly as in references/code/licong-lin_negative-preference-optimization
(TOFU/dataloader.py, ``npo_grad_diff``). ADAPTATION: NPO was proposed for
LLMs; on image classification we pair each retain batch with a (cycled) forget
batch like NegGrad+, with SGD; beta = 0.1 is the paper's default, the optimiser
settings are untuned starting points.
"""

import copy
from itertools import cycle

import torch
import torch.nn.functional as F

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Zhang et al. 2024)", "family": "bounded preference-style forgetting",
        "note": "LLM method adapted to classification"}

DEFAULTS = {"beta": 0.1, "npo_coeff": 1.0, "retain_coeff": 1.0, "epochs": 5, "lr": 0.01,
            "momentum": 0.9, "weight_decay": 5e-4}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "npo")
    net.to(DEVICE)
    ref = copy.deepcopy(net).eval()
    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                                weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    beta = hp["beta"]
    for _ in range(hp["epochs"]):
        net.train()
        for (x_r, y_r), (x_f, y_f) in zip(retain, cycle(forget)):
            x_r, y_r, x_f, y_f = x_r.to(DEVICE), y_r.to(DEVICE), x_f.to(DEVICE), y_f.to(DEVICE)
            with torch.no_grad():
                ce_ref = F.cross_entropy(ref(x_f), y_f, reduction="none")
            ce_cur = F.cross_entropy(net(x_f), y_f, reduction="none")
            l_npo = -F.logsigmoid(beta * (ce_cur - ce_ref)).mean() * 2 / beta
            loss = hp["npo_coeff"] * l_npo + hp["retain_coeff"] * F.cross_entropy(net(x_r), y_r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    net.eval()
    return net
