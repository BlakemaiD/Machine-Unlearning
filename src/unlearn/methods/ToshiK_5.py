"""5th place, NeurIPS 2023 Machine Unlearning (toshi_k & marvelworld): retrain from transposed weights.

The submission was an *ensemble over the 512 runs* of two approaches
(246 / 266 checkpoints each):

1. **Retraining from transposed weights** (implemented here): every Conv2d
   kernel is spatially transposed (``w[..., i, j] -> w[..., j, i]``), which
   breaks the memorised mapping while keeping useful filter statistics, then
   the network is fine-tuned on the retain set with cross-entropy
   (3 epochs, SGD lr 1e-3, cosine).
2. **Fine-tuning with pseudo-labels** that reproduce the *errors* a scratch
   model makes on forget data. Its implementation details were not published
   in a readable form (only in the Kaggle write-up), and the organisers'
   paper could not reproduce this team's entry. It is NOT implemented here.

Sources: organisers' slides, DeepUnlearn port ``kgltop5.py`` ("CT").
"""

import torch
from torch import nn, optim

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": 5, "team": "toshi_k & marvelworld", "family": "weight transpose + finetune",
        "note": "only approach (1) of the 2-approach ensemble"}

DEFAULTS = {"epochs": 3, "lr": 0.001, "momentum": 0.9, "weight_decay": 5e-4}


@torch.no_grad()
def _transpose_conv_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.copy_(m.weight.transpose(2, 3).clone())


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "ToshiK_5")
    net.to(DEVICE)
    _transpose_conv_weights(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    net.train()
    for _ in range(hp["epochs"]):
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net
