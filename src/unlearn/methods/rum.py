"""SOTA: RUM - Refined-Unlearning Meta-algorithm (Zhao et al., NeurIPS 2024,
"What makes unlearning hard and what to do about it").

Unlearning gets harder the more memorised the forget examples are. RUM
(1) splits the forget set into homogeneous groups by a memorisation proxy and
(2) unlearns the groups sequentially, low -> mid -> high memorisation, each
with its own hyperparameters, using an inner algorithm (NegGrad+ here, as in
the authors' proxy pipeline).

Group settings from the authors' ``main_rum_proxy.py``
(references/code/kairanzhao_RUM): NegGrad+ with
  low : alpha 0.99, 5 epochs, lr 0.01
  mid : alpha 0.97, 5 epochs, lr 0.01
  high: alpha 0.97, 10 epochs, lr 0.001

ADAPTATIONS (documented, overridable):
* Proxy: the authors' "confidence" proxy averages the true-class confidence
  over training checkpoints; only the final original model is available
  here, so we use its true-class confidence (memorisation ~ 1 - confidence).
  The given forget set is split into tertiles (the paper builds its own forget
  sets of 3,000 = 3 x 1,000 per group instead).
* Retain set: the reference's sequential mode puts already-unlearned
  (lower-memorisation) forget examples back into the retain set for later
  steps, i.e. trains on forget data; by default we keep retain = retain set
  only (``retain_includes_unlearned=False``). Set True for the reference behaviour.
"""

from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Zhao et al. 2024)", "family": "memorisation-aware meta-algorithm (NegGrad+ inside)",
        "note": "single-checkpoint confidence proxy"}

DEFAULTS = {
    "groups": (("low", 0.99, 5, 0.01), ("mid", 0.97, 5, 0.01), ("high", 0.97, 10, 0.001)),
    "momentum": 0.9, "weight_decay": 5e-4, "retain_includes_unlearned": False,
}


@torch.no_grad()
def _confidence(net, dataset):
    net.eval()
    conf = []
    for x, y in DataLoader(dataset, batch_size=512, shuffle=False):
        p = F.softmax(net(x.to(DEVICE)), 1)
        conf.append(p[torch.arange(len(y)), y.to(DEVICE)].cpu())
    return torch.cat(conf).numpy()


def _neggrad_plus(net, retain_ld, forget_ld, alpha, epochs, lr, momentum, wd):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        for (x_r, y_r), (x_f, y_f) in zip(retain_ld, cycle(forget_ld)):
            x_r, y_r, x_f, y_f = x_r.to(DEVICE), y_r.to(DEVICE), x_f.to(DEVICE), y_f.to(DEVICE)
            loss = alpha * F.cross_entropy(net(x_r), y_r) - (1 - alpha) * F.cross_entropy(net(x_f), y_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "rum")
    net.to(DEVICE)
    bs = retain.batch_size
    conf = _confidence(net, forget.dataset)
    order = np.argsort(-conf)                 # most confident (least memorised) first
    thirds = np.array_split(order, 3)          # low, mid, high memorisation
    done = []
    for (name, alpha, epochs, lr), idx in zip(hp["groups"], thirds):
        group = Subset(forget.dataset, idx.tolist())
        r_ds = retain.dataset
        if hp["retain_includes_unlearned"] and done:
            r_ds = ConcatDataset([r_ds] + done)
        _neggrad_plus(net, DataLoader(r_ds, batch_size=bs, shuffle=True),
                      DataLoader(group, batch_size=bs, shuffle=True),
                      alpha, epochs, lr, hp["momentum"], hp["weight_decay"])
        done.append(group)
    net.eval()
    return net
