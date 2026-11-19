"""SOTA: Scissorhands (Wu & Harandi, ECCV 2024).

1. **Trim:** connection sensitivity |g * w| (SNIP) from ``prune_num`` forget
   batches; the top (1 - ``sparsity``) most forget-salient weights in the
   later layers are re-initialised (kaiming-uniform weights, zero biases).
2. **Repair (``epochs``):** iterate retain batches paired with forget batches
   (until the forget set is exhausted, then retain only), minimising
   ``CE(retain) - lam * CE(forget)``. With ``project``, the gradient over the
   later layers is projected (GEM) so it does not conflict with the
   forget-ascent direction g_o of the trimmed model (memory of 1 batch).

Defaults are the authors' CIFAR-10 / 10% random-forgetting command
(references/code/JingWu321_Scissorhands/Classification/main.sh): lr 5e-3,
sparsity 0.97, lam 0.05, 10 epochs, memory 1, prune batches 1, batch 256,
SGD momentum 0.9, weight decay 5e-4, constant lr.

"Later layers" are ``layer3``/``layer4``/``fc`` (non-BN) for ResNet, as in
the reference; for other architectures, the last half of the non-norm
parameter tensors. The reference solves the GEM QP with ``quadprog``; with a
single memory it has the closed form used here:
``g <- g + max(-(g.g_o)/|g_o|^2, margin) * g_o`` when ``g.g_o < 0``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Wu & Harandi 2024)", "family": "saliency re-init + projected repair"}

DEFAULTS = {"lr": 5e-3, "sparsity": 0.97, "lam": 0.05, "epochs": 10, "prune_num": 1,
            "project": True, "margin": 0.5, "batch_size": 256, "momentum": 0.9,
            "weight_decay": 5e-4}


def _later_layer_names(net):
    names = [n for n, _ in net.named_parameters()]
    if any(n.startswith("layer4") for n in names):  # ResNet: exactly the reference rule
        return {n for n in names if "bn" not in n and ("layer3" in n or "layer4" in n or "fc" in n)}
    norm_params = {f"{mn}.{pn}" for mn, m in net.named_modules()
                   if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
                   for pn, _ in m.named_parameters(recurse=False)}
    cand = [n for n in names if n not in norm_params]
    return set(cand[len(cand) // 2:])


def _snip_reinit(net, forget_ld, sparsity, prune_num, later):
    grads = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    it = iter(forget_ld)
    for _ in range(prune_num):
        x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        net.zero_grad()
        F.cross_entropy(net(x), y).backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                grads[n] += p.grad.abs()
    with torch.no_grad():
        sal = {n: (grads[n] * p).abs() for n, p in net.named_parameters()}
        flat = torch.cat([v.flatten() for v in sal.values()])
        k = max(1, int(sparsity * flat.numel()))
        thr = flat.kthvalue(k).values.item()
        if thr >= flat.max().item() - 1e-12 or thr <= flat.min().item() + 1e-12:
            thr = (flat.max().item() - flat.min().item()) / 2.0
        for n, p in net.named_parameters():
            if n not in later:
                continue
            hit = sal[n] > thr
            if p.dim() >= 2:
                fresh = torch.empty_like(p)
                nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
            elif n.endswith("bias"):
                fresh = torch.zeros_like(p)
            else:  # 1-D weights (e.g. downsample BN in ResNet), as the reference does
                fresh = torch.empty_like(p).unsqueeze(1)
                nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
                fresh = fresh.squeeze(1)
            p[hit] = fresh[hit]
    net.zero_grad()


def _flat_grad(net, names):
    return torch.cat([p.grad.flatten() for n, p in net.named_parameters()
                      if n in names and p.grad is not None])


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "scissorhands")
    net.to(DEVICE)
    bs = hp["batch_size"]
    forget_ld = DataLoader(forget.dataset, batch_size=bs, shuffle=True)
    retain_ld = DataLoader(retain.dataset, batch_size=bs, shuffle=True)
    later = _later_layer_names(net)

    _snip_reinit(net, forget_ld, hp["sparsity"], hp["prune_num"], later)
    optimizer = torch.optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                                weight_decay=hp["weight_decay"])

    for _ in range(hp["epochs"]):
        g_o = None
        if hp["project"]:  # memory: forget-ascent gradient of the current model
            net.eval()
            x, y = next(iter(forget_ld))
            net.zero_grad()
            (-F.cross_entropy(net(x.to(DEVICE)), y.to(DEVICE))).backward()
            g_o = _flat_grad(net, later).detach().clone()
            net.zero_grad()
        net.train()
        forget_it = iter(forget_ld)
        for x_r, y_r in retain_ld:
            x_r, y_r = x_r.to(DEVICE), y_r.to(DEVICE)
            batch_f = next(forget_it, None)
            optimizer.zero_grad()
            if batch_f is None:
                F.cross_entropy(net(x_r), y_r).backward()
            else:
                x_f, y_f = batch_f[0].to(DEVICE), batch_f[1].to(DEVICE)
                loss = F.cross_entropy(net(x_r), y_r) - hp["lam"] * F.cross_entropy(net(x_f), y_f)
                loss.backward()
                if g_o is not None:
                    g = _flat_grad(net, later)
                    dot = torch.dot(g, g_o)
                    if dot < 0:
                        v = max(-(dot / (g_o.dot(g_o) + 1e-3)).item(), hp["margin"])
                        g_new = g + v * g_o
                        ptr = 0
                        for n, p in net.named_parameters():
                            if n in later and p.grad is not None:
                                p.grad.copy_(g_new[ptr:ptr + p.numel()].view_as(p))
                                ptr += p.numel()
            optimizer.step()
    net.eval()
    return net
