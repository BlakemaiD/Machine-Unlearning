"""Baseline: L1-sparse unlearning (Jia et al., NeurIPS 2023, "Model Sparsity Can Simplify Machine Unlearning").

Fine-tune on the retain set with an L1 penalty on all weights whose strength
decays linearly from ``alpha`` to 0 over training ("linear decay" schedule of
the reference code, github.com/OPTML-Group/Unlearn-Sparse, ``FT_l1``).
"""

from torch import nn, optim

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "baseline (Jia et al. 2023)", "family": "sparsity-regularised finetune"}

DEFAULTS = {"epochs": 10, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "alpha": 5e-4}


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "l1_sparse")
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    total = hp["epochs"] * len(retain)
    step = 0
    for _ in range(hp["epochs"]):
        net.train()
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            alpha_t = hp["alpha"] * (1 - step / total)
            optimizer.zero_grad()
            l1 = sum(p.abs().sum() for p in net.parameters())
            (criterion(net(x), y) + alpha_t * l1).backward()
            optimizer.step()
            step += 1
        scheduler.step()
    net.eval()
    return net
