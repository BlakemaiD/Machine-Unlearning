"""Baseline: NegGrad+ (Kurmanji et al., NeurIPS 2023, "Towards Unbounded Machine Unlearning").

Joint descent on retain and ascent on forget:
``loss = alpha * CE(retain) - (1 - alpha) * CE(forget)``, one forget batch
(cycled) per retain batch. DeepUnlearn (Cadet et al. 2024) recommends a tuned
NegGrad+ as the reference baseline; the organisers' paper tunes it too.
Defaults are untuned starting points for CIFAR-10 - tune alpha / lr on val.
"""

from itertools import cycle

from torch import nn, optim

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "baseline (Kurmanji et al. 2023)", "family": "gradient ascent + descent"}

DEFAULTS = {"epochs": 5, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "alpha": 0.99}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "neggrad_plus")
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    a = hp["alpha"]
    for _ in range(hp["epochs"]):
        net.train()
        for (x_r, y_r), (x_f, y_f) in zip(retain, cycle(forget)):
            x_r, y_r, x_f, y_f = x_r.to(DEVICE), y_r.to(DEVICE), x_f.to(DEVICE), y_f.to(DEVICE)
            optimizer.zero_grad()
            loss = a * criterion(net(x_r), y_r) - (1 - a) * criterion(net(x_f), y_f)
            loss.backward()
            optimizer.step()
        scheduler.step()
    net.eval()
    return net
