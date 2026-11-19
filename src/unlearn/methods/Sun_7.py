"""7th place, NeurIPS 2023 Machine Unlearning (Jiaxi Sun): reset head + repeated noise injection (retain-only).

Schedule over ``epochs`` (=9) passes of the retain set:

* epoch 0: reset the classifier ``fc``; train only a random selection of
  ``layer_select`` (=9) non-BN parameter tensors plus all BN parameters
  (everything else frozen); no noise yet.
* epochs 1 .. epochs-fine_epochs-1 ("noised" epochs): select 9 random non-BN
  parameter tensors (+ all BN); add Gaussian noise (std 0.04) to
  ``int(noise_ratio * 9)`` of them that were *not selected before*; train only
  the selected tensors for one epoch.
* last ``fine_epochs`` (=4) epochs: train all parameters, with a per-epoch LR
  from ``fine_lr``.

A fresh SGD optimizer is created every epoch (as in the original). "BN"
parameters are those whose name contains "bn"; note that ResNet-18's
downsample BatchNorms (``layerX.0.downsample.1``) do not match and are treated
like other layers, exactly as in the original code.

Sources: organisers' slides, paper A.4, DeepUnlearn port ``kgltop7.py`` (copied
from the submission notebook).
"""

import numpy as np
import torch
from torch import nn, optim

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": 7, "team": "Jiaxi Sun", "family": "head reset + noise injection + finetune"}

DEFAULTS = {
    "epochs": 9,
    "fine_epochs": 4,
    "reset_epochs": 1,
    "layer_select": 9,
    "noise_ratio": 0.7,
    "noise_std": 0.04,
    "lr": 0.0002, "momentum": 0.9, "weight_decay": 5e-4,
    "fine_lr": (0.0002, 0.0002, 0.0002, 0.001, 0.001, 0.0001),
    "fine_momentum": 0.9, "fine_weight_decay": 5e-4,
}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Sun_7")
    net.to(DEVICE)
    epochs, fine_epochs, layer_select = hp["epochs"], hp["fine_epochs"], hp["layer_select"]
    criterion = nn.CrossEntropyLoss()

    get_classifier(net).reset_parameters()
    params = dict(net.named_parameters())
    keys = list(params)
    bn_keys = [k for k in keys if "bn" in k]
    selectable = [k for k in keys if "bn" not in k]
    already_selected = set()

    for ep in range(epochs):
        fine_phase = ep >= epochs - fine_epochs
        if fine_phase:
            optim_keys = keys
        else:
            chosen = np.random.choice(selectable, min(layer_select, len(selectable)),
                                      replace=False).tolist()
            optim_keys = sorted(set(chosen + bn_keys))
            # noise injection (skipped during the reset epochs)
            if ep >= hp["reset_epochs"]:
                candidates = [k for k in optim_keys if "bn" not in k]
                noisy = np.random.choice(candidates, int(hp["noise_ratio"] * layer_select),
                                         replace=False)
                with torch.no_grad():
                    for k in sorted(noisy):
                        if k in already_selected:
                            continue
                        params[k].add_(torch.randn_like(params[k]) * hp["noise_std"])

        for k in keys:
            params[k].requires_grad_(k in optim_keys)

        if fine_phase:
            fine_lr = hp["fine_lr"]
            lr = fine_lr[ep - epochs + fine_epochs] if hasattr(fine_lr, "__getitem__") else fine_lr
            optimizer = optim.SGD(params.values(), lr=lr, momentum=hp["fine_momentum"],
                                  weight_decay=hp["fine_weight_decay"])
        else:
            optimizer = optim.SGD(params.values(), lr=hp["lr"], momentum=hp["momentum"],
                                  weight_decay=hp["weight_decay"])
        already_selected.update(optim_keys)

        net.train()
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()

    for p in net.parameters():
        p.requires_grad_(True)
    net.eval()
    return net
