"""4th place, NeurIPS 2023 Machine Unlearning (Sebastian Oleszko): prune/re-init + entropy-matched finetune.

1. **Prune & re-initialise 99%** of conv + linear parameters by global
   unstructured L1 magnitude. Pruned weights are re-drawn from PyTorch's default
   uniform(-sqrt(k), sqrt(k)) (k = 1 / fan_in); pruned biases stay zero.
2. **Fine-tune on retain** for a fractional number of epochs (3.2 in the final
   submission) with
   ``CE(class weights = count**-0.1) + MSE(H(student), H(original))``
   where H is prediction entropy; the entropy term keeps the confidence profile
   of the original model.
3. Round-trip the weights through fp16 (the "quantize" step of the submission).

Sources: organisers' slides (references/papers/2023_NeurIPS_Unlearning_Top_Solution_Highlights_slides.pdf),
paper A.4, DeepUnlearn port ``kgltop4.py`` (copied from the submission notebook).
"""

import copy
from math import sqrt

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import nn, optim

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": 4, "team": "Sebastian Oleszko", "family": "prune/re-init + distill entropy"}

DEFAULTS = {
    "prune_amount": 0.99,
    "epochs": 3.2,  # fractional: training stops after int(len(retain) * epochs) steps
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "class_weight_power": -0.1,
    "half_precision_roundtrip": True,
}


@torch.no_grad()
def _prune_and_reinit(net, amount):
    params = []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append((m, "weight"))
            if m.bias is not None:
                params.append((m, "bias"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.remove(m, "weight")
            if m.bias is not None:
                prune.remove(m, "bias")
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask = m.weight == 0
            fan_in = m.weight[0].numel()
            bound = sqrt(1.0 / fan_in)
            rand = (torch.rand_like(m.weight) - 0.5) * 2 * bound
            m.weight[mask] = rand[mask]


def _class_weights(retain, power, num_classes):
    counts = torch.zeros(num_classes)
    for _, y in retain:
        counts += torch.bincount(y, minlength=num_classes).float()
    return counts.clamp_min(1) ** power


def _loss(outputs, labels, target_logp, class_weights):
    ce = F.cross_entropy(outputs, labels, weight=class_weights)
    h_target = torch.sum(-torch.exp(target_logp) * target_logp, dim=1)
    h_out = torch.sum(-torch.softmax(outputs, 1) * torch.log_softmax(outputs, 1), dim=1)
    return ce + F.mse_loss(h_out, h_target)


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Sebastian_4")
    net.to(DEVICE)
    original = copy.deepcopy(net).eval()
    num_classes = get_classifier(net).out_features
    class_w = _class_weights(retain, hp["class_weight_power"], num_classes).to(DEVICE)

    _prune_and_reinit(net, hp["prune_amount"])
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    max_iters = int(len(retain) * hp["epochs"])

    it = 0
    net.train()
    while it <= max_iters:
        for x, y in retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                target_logp = torch.log_softmax(original(x), dim=1)
            optimizer.zero_grad()
            _loss(net(x), y, target_logp, class_w).backward()
            optimizer.step()
            it += 1
            if it > max_iters:
                break

    net.eval()
    if hp["half_precision_roundtrip"]:
        net = net.half().float()
    return net
