"""Baseline: Random Label (Golatkar et al. 2020; as implemented in SalUn, Fan et al. ICLR 2024).

Fine-tune on the union of the retain set (true labels) and the forget set with
labels resampled uniformly from the *other* classes (fresh labels each epoch).
The SalUn repository's ``RL`` script trains on this union with plain SGD.
Defaults are untuned starting points for CIFAR-10.
"""

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": None, "team": "baseline (Golatkar 2020 / Fan 2024)", "family": "relabel forget set"}

DEFAULTS = {"epochs": 5, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "batch_size": 256}


class _RandomLabels(Dataset):
    """Wrap a dataset; each item gets a random label != its true label (resampled per access)."""

    def __init__(self, ds, num_classes):
        self.ds, self.k = ds, num_classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        y = int(y)
        r = int(torch.randint(0, self.k - 1, (1,)))
        return x, torch.tensor(r if r < y else r + 1)


class _AsTensorLabel(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, torch.as_tensor(y)


def relabel_train(net, retain, forget, hp, grad_mask=None):
    """Shared RL / SalUn training loop. ``grad_mask``: {param: 0/1 tensor} or None."""
    k = get_classifier(net).out_features
    data = ConcatDataset([_AsTensorLabel(retain.dataset), _RandomLabels(forget.dataset, k)])
    loader = DataLoader(data, batch_size=hp["batch_size"], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    for _ in range(hp["epochs"]):
        net.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            if grad_mask is not None:
                for p, m in grad_mask.items():
                    if p.grad is not None:
                        p.grad.mul_(m)
            optimizer.step()
        scheduler.step()
    net.eval()
    return net


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "random_label")
    net.to(DEVICE)
    return relabel_train(net, retain, forget, hp)
