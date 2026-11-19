"""3rd place, NeurIPS 2023 Machine Unlearning (Seif Eddine Achour): "vision confusion - reconstitution".

1. **Vision confusion:** replace conv weights by a Gaussian sample centred on the
   current weights, ``w <- N(w, sigma=0.6)``.
2. **Reconstitution:** 4 epochs of (class-weighted) cross-entropy fine-tuning on
   the retain set, SGD + cosine schedule.
3. **Stabilisation:** before the final epoch, a small extra confusion
   (``sigma=0.005``) to increase robustness.

Faithfulness notes (from the DeepUnlearn port ``kgltop3.py`` of the original notebook):

* The original ``vision_confuser`` iterates over ``model.named_children()`` and
  perturbs children whose *name contains "conv"*. For torchvision's ResNet-18
  this is only the stem ``conv1`` (the residual blocks are children named
  ``layer1``..``layer4``). We keep that behaviour by default
  (``confuse_all_convs=False``); set it to True to perturb every Conv2d, which
  matches the paper's wording "noise on the convolutional weights".
* The original loss computes ``mean(CE) * mean(batch class weights)``, i.e. the
  class weights act as a per-batch learning-rate multiplier (paper A.4). The
  submitted weights ``[1, w, w, ...]`` with ``w=0.05`` were specific to the
  imbalanced age-group face dataset; on balanced CIFAR-10 we default to
  sklearn "balanced" weights (~1 for every class, as DeepUnlearn does).
  ``class_weights="original"`` reproduces the submitted weights literally.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": 3, "team": "Seif Eddine Achour", "family": "weight noise + finetune"}

DEFAULTS = {
    "epochs": 4,
    "lr": 0.0007,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "start_std": 0.6,
    "final_std": 0.005,
    "class_weights": "balanced",  # or "original"
    "original_w": 0.05,
    "confuse_all_convs": False,
}


@torch.no_grad()
def _vision_confuser(net, std, all_convs):
    if all_convs:
        modules = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    else:  # original behaviour: top-level children whose name contains "conv"
        modules = [m for name, m in net.named_children()
                   if hasattr(m, "weight") and "conv" in name]
    for m in modules:
        m.weight.copy_(torch.normal(mean=m.weight.detach().clone(), std=std))


def _class_weights(retain, mode, w):
    targets = np.concatenate([y.numpy() for _, y in retain])
    classes = np.unique(targets)
    if mode == "original":
        weights = np.full(classes.max() + 1, w, dtype=np.float64)
        weights[0] = 1.0
        return torch.tensor(weights, dtype=torch.float32)
    if mode == "balanced":
        counts = np.bincount(targets, minlength=classes.max() + 1)
        weights = len(targets) / (len(classes) * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)
    raise ValueError("class_weights must be 'balanced' or 'original'")


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Seif_3")
    net.to(DEVICE)
    _vision_confuser(net, hp["start_std"], hp["confuse_all_convs"])

    class_w = _class_weights(retain, hp["class_weights"], hp["original_w"]).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    epochs = hp["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        net.train()
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            # original CustomCrossEntropyLoss: mean CE scaled by mean batch weight
            loss = F.cross_entropy(net(x), y) * class_w[y].mean()
            loss.backward()
            optimizer.step()
        if ep == epochs - 2:
            _vision_confuser(net, hp["final_std"], hp["confuse_all_convs"])
        scheduler.step()

    net.eval()
    return net
