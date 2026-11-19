"""6th place, NeurIPS 2023 Machine Unlearning (team "Algorithmic Amnesiacs"): reset first/last layer + distillation.

1. **Reset** the first conv layer (``conv1``) and the classifier (``fc``) of
   the original model. (On CIFAR-10 these two layers showed the most negative
   cosine similarity between full-data and retain-only models.)
2. **Warm-up (3 epochs)** on the *validation* set: minimise
   KL(teacher || student) at temperature T, teacher = frozen original model.
   No training labels or forget data are touched.
3. **Fine-tune (3 epochs)** on the retain set with
   ``CE(y) + soft-CE(teacher soft labels) + KL(teacher || student)``
   (all distillation terms at temperature T), SGD + cosine.

Sources: organisers' slides, paper A.4, DeepUnlearn port ``kgltop6.py`` (adapted
from the team's Kaggle notebook "unlearning-by-resetting-layers").
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn, optim

from ..device import DEVICE
from ..models import get_classifier, get_first_conv
from ..utils import merge_hparams

INFO = {"rank": 6, "team": "Algorithmic Amnesiacs", "family": "layer reset + distillation"}

DEFAULTS = {
    "temperature": 5.0,
    "warmup_epochs": 3, "warmup_lr": 9e-4, "warmup_momentum": 0.9, "warmup_weight_decay": 5e-4,
    "retain_epochs": 3, "retain_lr": 1e-3, "retain_momentum": 0.9, "retain_weight_decay": 5e-4,
}


def _kl(student_logits, teacher_logits, t):
    return F.kl_div(F.log_softmax(student_logits / t, dim=1),
                    F.softmax(teacher_logits / t, dim=1), reduction="batchmean")


def _soft_ce(preds, soft_targets):
    return torch.mean(torch.sum(-soft_targets * torch.log_softmax(preds, dim=1), dim=1))


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Amnesiacs_6")
    net.to(DEVICE)
    teacher = copy.deepcopy(net).eval()
    get_first_conv(net).reset_parameters()
    get_classifier(net).reset_parameters()
    t = hp["temperature"]

    # Warm-up: distil the teacher on held-out validation data.
    optimizer = optim.SGD(net.parameters(), lr=hp["warmup_lr"], momentum=hp["warmup_momentum"],
                          weight_decay=hp["warmup_weight_decay"])
    for _ in range(hp["warmup_epochs"]):
        net.train()
        for x, _ in validation:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                t_out = teacher(x)
            _kl(net(x), t_out, t).backward()
            optimizer.step()

    # Fine-tune on retain with CE + soft-CE + KL.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hp["retain_lr"], momentum=hp["retain_momentum"],
                          weight_decay=hp["retain_weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["retain_epochs"])
    for _ in range(hp["retain_epochs"]):
        net.train()
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = net(x)
            with torch.no_grad():
                t_out = teacher(x)
            loss = (_soft_ce(torch.log_softmax(out / t, dim=1), torch.softmax(t_out / t, dim=1))
                    + criterion(out, y) + _kl(out, t_out, t))
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net
