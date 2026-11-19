"""CIFAR-10 data for the unlearning experiments.

Splits follow the NeurIPS 2023 starting kit exactly:

* ``forget``: the challenge's 5,000 training indices (``forget_idx.npy``).
* ``retain``: the remaining 45,000 training images.
* ``test`` / ``val``: a seeded 50/50 ``random_split`` of the CIFAR-10 test set.

Two dataset backends are available:

* ``in_memory=False`` (starter-kit behaviour): torchvision ``Subset`` objects,
  transformed on the fly.
* ``in_memory=True`` (default for the pipeline scripts): the same images,
  normalised once and stored as tensors. Images and splits are identical; data
  loading is just much faster (no PIL decode), which matters when every method
  is run tens of times.

Both return ``(image, label)`` tuples, so all methods work with either.
"""

import os

import numpy as np
import requests
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

FORGET_IDX_URL = "https://storage.googleapis.com/unlearning-challenge/forget_idx.npy"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = 10

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _get_forget_idx(path):
    if not os.path.exists(path):
        r = requests.get(FORGET_IDX_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return np.load(path)


def _to_normalized_tensor(images_uint8):
    """[N, 32, 32, 3] uint8 numpy -> [N, 3, 32, 32] float32 normalised tensor."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def get_split_indices(root="./data", forget_idx_path="forget_idx.npy", seed=42):
    """Return forget/retain train indices and test/val held-out indices."""
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    held_out = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    rng = torch.Generator().manual_seed(seed)
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=rng)

    forget_idx = np.asarray(_get_forget_idx(forget_idx_path))
    mask = np.zeros(len(train_set.targets), dtype=bool)
    mask[forget_idx] = True
    retain_idx = np.arange(mask.size)[~mask]
    return {
        "forget": forget_idx,
        "retain": retain_idx,
        "test": np.asarray(test_set.indices),
        "val": np.asarray(val_set.indices),
    }


def get_cifar10_loaders(root="./data", forget_idx_path="forget_idx.npy",
                        batch_size=128, num_workers=2, seed=42, in_memory=False,
                        shuffle_seed=None):
    """Return a dict of CIFAR-10 data loaders.

    Keys: ``train``, ``test``, ``val``, ``forget``, ``retain`` (as in the
    starting kit) plus ``forget_eval`` and ``retain_eval``, which iterate in a
    fixed order. Per-example metrics (e.g. the competition's forgetting score)
    need the same example order across every model, so always evaluate with the
    ``*_eval`` loaders.

    ``seed`` fixes the test/val split (and, as in the starting kit, the retain
    shuffle order). Pass ``shuffle_seed`` to give the retain loader its own
    shuffle stream: every unlearning run must see a different data order, or
    N runs of a method lose the run-to-run randomness the competition metric
    measures.
    """
    rng = torch.Generator().manual_seed(seed)

    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=None if in_memory else NORMALIZE)
    held_out = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=None if in_memory else NORMALIZE)

    # same order as the starter kit: random_split first, then retain_loader uses rng
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=rng)

    forget_idx = _get_forget_idx(forget_idx_path)
    mask = np.zeros(len(train_set.targets), dtype=bool)
    mask[forget_idx] = True
    retain_idx = np.arange(mask.size)[~mask]

    if in_memory:
        x_train = _to_normalized_tensor(train_set.data)
        y_train = torch.tensor(train_set.targets, dtype=torch.long)
        x_held = _to_normalized_tensor(held_out.data)
        y_held = torch.tensor(held_out.targets, dtype=torch.long)

        def tds(x, y, idx=None):
            if idx is None:
                return TensorDataset(x, y)
            idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            return TensorDataset(x[idx], y[idx])

        full_train = tds(x_train, y_train)
        test_ds = tds(x_held, y_held, test_set.indices)
        val_ds = tds(x_held, y_held, val_set.indices)
        forget_ds = tds(x_train, y_train, forget_idx)
        retain_ds = tds(x_train, y_train, retain_idx)
        num_workers = 0  # tensors are already in memory; workers only add overhead
    else:
        full_train = train_set
        test_ds, val_ds = test_set, val_set
        forget_ds = Subset(train_set, forget_idx)
        retain_ds = Subset(train_set, retain_idx)

    def dl(ds, shuffle, **kw):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, **kw)

    return {
        "train": dl(full_train, True),
        "test": dl(test_ds, False),
        "val": dl(val_ds, False),
        "forget": dl(forget_ds, True),
        "retain": dl(retain_ds, True, generator=(
            rng if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed))),
        "forget_eval": dl(forget_ds, False),
        "retain_eval": dl(retain_ds, False),
    }


def get_train_tensors(root="./data"):
    """Raw CIFAR-10 training images/labels as uint8 / int64 tensors.

    Used by :mod:`unlearn.train` for fast GPU-resident training from scratch.
    """
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    x = torch.from_numpy(train_set.data).permute(0, 3, 1, 2).contiguous()  # uint8
    y = torch.tensor(train_set.targets, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# Dataset-agnostic entry points used by the pipeline scripts
# ---------------------------------------------------------------------------

DATASETS = ("cifar10", "surf")


def get_loaders(dataset, root, forget_idx_path="forget_idx.npy", batch_size=64, seed=42,
                shuffle_seed=None):
    """Loaders dict (same keys as :func:`get_cifar10_loaders`) for ``dataset``."""
    if dataset == "cifar10":
        return get_cifar10_loaders(root=root, forget_idx_path=forget_idx_path,
                                   batch_size=batch_size, seed=seed, in_memory=True,
                                   shuffle_seed=shuffle_seed)
    if dataset == "surf":
        from .data_surf import load_surf
        d = load_surf(root)
        x, y = d["x"].float(), d["y"]  # raw 0-255 floats, as in the official pipeline

        def tds(idx):
            idx = torch.as_tensor(idx, dtype=torch.long)
            return TensorDataset(x[idx], y[idx])

        rng = torch.Generator().manual_seed(seed if shuffle_seed is None else shuffle_seed)

        def dl(ds, shuffle, **kw):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)

        return {
            "train": dl(tds(d["train"]), True),
            "test": dl(tds(d["test"]), False),
            "val": dl(tds(d["val"]), False),
            "forget": dl(tds(d["forget"]), True),
            "retain": dl(tds(d["retain"]), True, generator=rng),
            "forget_eval": dl(tds(d["forget"]), False),
            "retain_eval": dl(tds(d["retain"]), False),
        }
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


def get_train_arrays(dataset, root, forget_idx_path="forget_idx.npy"):
    """uint8 images, labels and retain/forget indices for training from scratch."""
    if dataset == "cifar10":
        x, y = get_train_tensors(root)
        forget_idx = np.asarray(_get_forget_idx(forget_idx_path))
        mask = np.zeros(len(y), dtype=bool)
        mask[forget_idx] = True
        return {"x": x, "y": y, "forget": forget_idx, "retain": np.arange(len(y))[~mask],
                "train": np.arange(len(y)), "normalize": "cifar10", "num_classes": NUM_CLASSES}
    if dataset == "surf":
        from .data_surf import NUM_CLASSES as SURF_CLASSES, load_surf
        d = load_surf(root)
        return {"x": d["x"], "y": d["y"], "forget": d["forget"], "retain": d["retain"],
                "train": d["train"], "normalize": "none", "num_classes": SURF_CLASSES}
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
