#data will be loadedfrom the google storage along with the split retain adn forget datasets.

import os
import numpy as np
import requests
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

FORGET_IDX_URL = "https://storage.googleapis.com/unlearning-challenge/forget_idx.npy"

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def _get_forget_idx(path):
    if not os.path.exists(path):
        r = requests.get(FORGET_IDX_URL)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return np.load(path)

def get_cifar10_loaders(root="./data", forget_idx_path="forget_idx.npy",
                        batch_size=128, num_workers=2, seed=42):
    rng = torch.Generator().manual_seed(seed)

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=NORMALIZE)
    held_out = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=NORMALIZE)

    # same order as the starter kit: random_split first, then retain_loader uses rng
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=rng)

    forget_idx = _get_forget_idx(forget_idx_path)
    mask = np.zeros(len(train_set.targets), dtype=bool)
    mask[forget_idx] = True
    retain_idx = np.arange(mask.size)[~mask]

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)

    def dl(ds, shuffle, **kw):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)

    return {
        "train":  dl(train_set, True),
        "test":   dl(test_set, False),
        "val":    dl(val_set, False),
        "forget": dl(forget_set, True),
        "retain": dl(retain_set, True, generator=rng),
    }