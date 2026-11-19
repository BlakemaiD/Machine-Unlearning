"""SOTA: SCAR (Bonato, Cotogni & Sabetta, ECCV 2024).

1. Class statistics of the *original* model's penultimate features on the
   retain set: Tukey-transformed (x**delta) class means and shrunk, normalised
   covariance matrices (inverse via pseudo-inverse).
2. Each forget sample is assigned its closest *other* class by Mahalanobis
   distance (L2-normalised features), fixed at the first epoch.
3. Training (Adam): ``lambda_1 * mean Mahalanobis distance`` pulls forget
   embeddings to that class, and ``lambda_2 * KL(original || model)`` on
   distillation images (temperature-scaled teacher) keeps everything else.
   Each forget batch is paired with ``num_retain_samp + 1`` distillation batches.
   In random ("HR") mode, classes whose forget accuracy is already below
   ``class_threshold`` stop being pushed; training stops early if forget
   accuracy drops below ``target_accuracy``.

Defaults are the authors' CIFAR-10 ResNet-18 settings
(references/code/jbonato1_SCAR/src/opts.py): Adam lr 5e-4, wd 0, batch 1024,
up to 30 epochs, lambda_1 = 1, lambda_2 = 5, T = 1, delta = 0.5,
gamma_1 = gamma_2 = 3, num_retain_samp = 5, class threshold 0.8, target 0.01.

ADAPTATION: the paper distils on ~10k surrogate out-of-distribution images
(an ImageNet subset), which we do not have; we use retain images in that role
(the code's ``retain_sur`` input accepts either). Per-step forget-accuracy
checks use the full forget set, as in the reference.
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Bonato et al. 2024)", "family": "Mahalanobis feature push + distillation",
        "note": "retain images replace the paper's surrogate OOD images"}

DEFAULTS = {"lr": 5e-4, "weight_decay": 0.0, "batch_size": 1024, "epochs": 30,
            "lambda_1": 1.0, "lambda_2": 5.0, "temperature": 1.0, "delta": 0.5,
            "gamma1": 3.0, "gamma2": 3.0, "num_retain_samp": 5,
            "class_threshold": 0.8, "target_accuracy": 0.01}


def _split(net):
    """(backbone -> flat features, classifier). ResNet: all children but fc."""
    fc = get_classifier(net)
    if hasattr(net, "fc") and net.fc is fc:
        bbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
    else:  # torchvision MobileNet / EfficientNet: features + avgpool, classifier
        bbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
    return bbone, fc


def _shrink(cov, g1, g2):
    eye = torch.eye(cov.shape[0], device=cov.device)
    v1 = torch.mean(torch.diagonal(cov))
    off = cov.clone()
    off.fill_diagonal_(0.0)
    mask = off != 0.0
    v2 = (off * mask).sum() / mask.sum().clamp_min(1)
    return cov + g1 * eye * v1 + g2 * (1 - eye) * v2


def _normalize_cov(cov):
    s = torch.sqrt(torch.diagonal(cov))
    return cov / (s[:, None] * s[None, :])


@torch.no_grad()
def _accuracy(net, loader, num_classes):
    net.eval()
    correct = torch.zeros(num_classes, device=DEVICE)
    total = torch.zeros(num_classes, device=DEVICE)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        hit = (net(x).argmax(1) == y).float()
        correct.index_add_(0, y, hit)
        total.index_add_(0, y, torch.ones_like(hit))
    net.train()
    return (correct.sum() / total.sum()).item(), correct / total.clamp_min(1)


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "scar")
    net.to(DEVICE)
    original = copy.deepcopy(net).eval()
    bbone, fc = _split(net)
    k = fc.out_features
    tukey = lambda v: torch.pow(v.clamp_min(0), hp["delta"])  # features are post-ReLU (>= 0)

    retain_eval = DataLoader(retain.dataset, batch_size=hp["batch_size"], shuffle=False)
    bbone.eval()
    with torch.no_grad():
        embs, labs = [], []
        for x, y in retain_eval:
            embs.append(bbone(x.to(DEVICE)))
            labs.append(y.to(DEVICE))
        embs, labs = torch.cat(embs), torch.cat(labs)
        means, inv_covs = [], []
        for c in range(k):
            s = tukey(embs[labs == c])
            means.append(s.mean(0))
            cov = _normalize_cov(_shrink(_shrink(torch.cov(s.T), hp["gamma1"], hp["gamma2"]),
                                         hp["gamma1"], hp["gamma2"]))
            inv_covs.append(torch.linalg.pinv(cov))
        means = F.normalize(torch.stack(means), p=2, dim=-1)
        inv_covs = torch.stack(inv_covs)
    del embs, labs

    def mahalanobis(feats):  # [B, D] -> [B, K]
        diff = F.normalize(tukey(feats), p=2, dim=-1)[:, None, :] - means  # [B,K,D]
        right = torch.einsum("bkd,kde->bke", diff, inv_covs)
        return (right * diff).sum(-1)

    forget_ld = DataLoader(forget.dataset, batch_size=hp["batch_size"], shuffle=True)
    distill_ld = DataLoader(retain.dataset, batch_size=hp["batch_size"], shuffle=True)
    forget_eval = DataLoader(forget.dataset, batch_size=hp["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    closest = []
    vec_forg = None
    done = False
    net.train()
    for epoch in range(hp["epochs"]):
        for b, (x_f, y_f) in enumerate(forget_ld):
            x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)
            for r, (x_r, _) in enumerate(distill_ld):
                if r > hp["num_retain_samp"]:
                    break
                x_r = x_r.to(DEVICE)
                optimizer.zero_grad()
                d = mahalanobis(bbone(x_f))
                if epoch == 0 and r == 0:
                    order = torch.argsort(d, dim=1)
                    c = torch.where(order[:, 0] == y_f, order[:, 1], order[:, 0])
                    closest.append(c)
                c = closest[b]
                d = d[torch.arange(d.shape[0], device=DEVICE), c[:d.shape[0]]]
                if vec_forg is not None:  # random ("HR") mode class gating
                    gate = torch.where(vec_forg < hp["class_threshold"], torch.zeros_like(vec_forg), vec_forg)
                    d = d * gate[y_f]
                loss_f = d.mean() * hp["lambda_1"]
                with torch.no_grad():
                    t_out = original(x_r) / hp["temperature"]
                out_r = fc(bbone(x_r))
                loss_r = F.kl_div(F.log_softmax(out_r + 1e-4, 1), F.log_softmax(t_out + 1e-4, 1),
                                  reduction="batchmean", log_target=True) * hp["lambda_2"]
                (loss_f + loss_r).backward()
                optimizer.step()
                acc, vec_forg = _accuracy(net, forget_eval, k)
                if acc < hp["target_accuracy"] and epoch > 1:
                    done = True
                    break
            if done:
                break
        if done:
            break
    net.eval()
    return net
