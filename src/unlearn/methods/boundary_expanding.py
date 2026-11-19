"""SOTA: Boundary Expanding (Chen et al., CVPR 2023, "Boundary Unlearning").

1. Widen the classifier with one extra "shadow" output (weights for the
   existing classes copied, the shadow row freshly initialised).
2. Fine-tune on the forget set with every sample labelled as the shadow class,
   which pulls the forget samples' activations away from their true classes.
3. Prune the shadow output, restoring the original number of classes.

The paper's implementation was distributed as a zip (not on GitHub); the
optimiser settings here mirror Boundary Shrink's reference code (SGD lr 1e-5,
momentum 0.9, 10 passes over the forget set) and are ASSUMPTIONS, overridable.
Designed for whole-class forgetting; applied here to a random forget set.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Chen et al. 2023)", "family": "shadow-class expand + prune",
        "note": "optimiser settings assumed (paper code not public)"}

DEFAULTS = {"lr": 1e-5, "momentum": 0.9, "epochs": 10}


def _replace_classifier(net, new_layer):
    old = get_classifier(net)
    for name, module in net.named_modules():
        for child_name, child in module.named_children():
            if child is old:
                setattr(module, child_name, new_layer)
                return
    raise RuntimeError("classifier not found")


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "boundary_expanding")
    net.to(DEVICE)
    fc = get_classifier(net)
    k = fc.out_features
    wide = nn.Linear(fc.in_features, k + 1, bias=fc.bias is not None).to(DEVICE)
    with torch.no_grad():
        wide.weight[:k].copy_(fc.weight)
        if fc.bias is not None:
            wide.bias[:k].copy_(fc.bias)
    _replace_classifier(net, wide)

    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"])
    for _ in range(hp["epochs"]):
        net.train()
        for x, _ in forget:
            x = x.to(DEVICE)
            shadow = torch.full((x.shape[0],), k, dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(net(x), shadow).backward()
            optimizer.step()

    pruned = nn.Linear(fc.in_features, k, bias=fc.bias is not None).to(DEVICE)
    with torch.no_grad():
        pruned.weight.copy_(wide.weight[:k])
        if fc.bias is not None:
            pruned.bias.copy_(wide.bias[:k])
    _replace_classifier(net, pruned)
    net.eval()
    return net
