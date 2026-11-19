"""8th place, NeurIPS 2023 Machine Unlearning (team "Forget", Kyungpook National Univ.): forget-remember cycles.

A forget-set-free method. Repeat for each cycle:

1. **Forgetting phase:** stochastically pick ``layers_per_cycle`` (=6) layers,
   *with replacement*, from a pool of Conv2d layers that excludes the
   classifier ``fc`` and the projection-shortcut (``downsample``) convs, and
   re-initialise them. Across cycles the total re-initialised fraction grows.
2. **Remembering phase:** knowledge-preserving loss
   ``MSE(student logits, original-model logits)`` on retain images with
   additive Gaussian noise (sigma=0.01 in the final submission) as the only
   augmentation, for ``cycle_epochs[i]`` epochs.

Defaults = the team's "Algorithm 1": 3 cycles, epochs [1, 2, 2], cosine LR
from 1e-3 with T_max=2.

Source: team write-up (https://junhyunb.github.io/projects/neurips2023-unlearning,
Kaggle write-up "forget-9th-place-solution-forget-set-free-approach") and the
organisers' slides. No code was published, so these details are ASSUMPTIONS
(all overridable): SGD with momentum 0.9 and weight decay 5e-4; the cosine
schedule restarts every cycle and steps per epoch; re-initialisation uses each
layer's ``reset_parameters()`` (PyTorch default init).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": 8, "team": "Forget (KNU)", "family": "stochastic re-init + distillation cycles",
        "note": "re-implemented from write-up; optimizer details assumed"}

DEFAULTS = {
    "cycle_epochs": (1, 2, 2),
    "layers_per_cycle": 6,
    "lr": 0.001,
    "t_max": 2,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "noise_std": 0.01,
}


def _reinit_pool(net):
    # Excludes ResNet projection shortcuts ("downsample"); the fc is not a conv.
    # MobileNet/EfficientNet have no projection shortcuts, so all convs are eligible.
    return [(name, m) for name, m in net.named_modules()
            if isinstance(m, nn.Conv2d) and "downsample" not in name]


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Forget_8")
    net.to(DEVICE)
    teacher = copy.deepcopy(net).eval()
    pool = _reinit_pool(net)

    for n_epochs in hp["cycle_epochs"]:
        # Forgetting phase
        picks = np.random.choice(len(pool), hp["layers_per_cycle"], replace=True)
        with torch.no_grad():
            for i in picks:
                pool[i][1].reset_parameters()

        # Remembering phase
        optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                              weight_decay=hp["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["t_max"])
        for _ in range(n_epochs):
            net.train()
            for x, _ in retain:
                x = x.to(DEVICE)
                x = x + hp["noise_std"] * torch.randn_like(x)
                with torch.no_grad():
                    target = teacher(x)
                optimizer.zero_grad()
                F.mse_loss(net(x), target).backward()
                optimizer.step()
            scheduler.step()

    net.eval()
    return net
