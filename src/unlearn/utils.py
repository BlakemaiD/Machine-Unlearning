"""Small shared helpers: seeding, hyperparameter merging, timing/memory tracking."""

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from .device import DEVICE


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def merge_hparams(defaults, overrides, method_name):
    """Return defaults updated with overrides, rejecting unknown keys.

    Every method module keeps its published hyperparameters in a ``DEFAULTS``
    dict; this keeps overrides explicit and catches typos such as ``lr`` vs
    ``learning_rate``.
    """
    unexpected = set(overrides) - set(defaults)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"{method_name}: unexpected hyperparameter(s): {names}. "
                        f"Known: {', '.join(sorted(defaults))}")
    merged = dict(defaults)
    merged.update(overrides)
    return merged


@contextmanager
def track_cost():
    """Measure wall-clock time and peak GPU memory of the enclosed block.

    Usage::

        with track_cost() as cost:
            run()
        cost["seconds"], cost["peak_mem_mb"]
    """
    cost = {}
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield cost
    finally:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            cost["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        else:
            cost["peak_mem_mb"] = float("nan")
        cost["seconds"] = time.perf_counter() - start
