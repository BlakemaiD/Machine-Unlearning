"""Train a classifier from scratch (original model or retrain-from-scratch oracle).

The competition's forgetting score compares the distribution of N unlearned
models with N models *retrained from scratch on the retain set*. Those oracle
models (and our own "original" models trained on retain+forget) are produced
here. Original and retrained models must share one recipe; only the training
indices and the seed differ.

For speed the training subset lives on the GPU and augmentation (random crop
with 4px padding + horizontal flip) runs on-GPU per batch.

Recipes:

* ``RECIPES["cifar10"]`` - 100 epochs, SGD (momentum 0.9, Nesterov), batch 128,
  5-epoch linear warm-up to lr 0.1, then cosine decay to ``final_lr`` 1e-4;
  weight decay 5e-4; random crop + flip. Check after the first models that
  the retrained ones are near the challenge's retrained checkpoint
  (retain ~99.5%, test ~88%); see document.md.
* ``RECIPES["surf"]`` - the official recipe of google-deepmind/unlearning_evaluation
  (``train_lib.py``): Adam lr 1e-4, wd 5e-3, cosine, 30 epochs, batch 64,
  inverse-frequency class-weighted CE, no augmentation, raw 0-255 inputs.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn, optim

from .data import CIFAR10_MEAN, CIFAR10_STD
from .device import DEVICE
from .models import build_model
from .utils import seed_everything

RECIPES = {
    "cifar10": {
        "epochs": 100, "batch_size": 128, "optimizer": "sgd", "lr": 0.1, "final_lr": 1e-4,
        "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True, "warmup_epochs": 5,
        "label_smoothing": 0.0,
        "augment": True, "class_weighted": False, "normalize": "cifar10", "amp": True,
    },
    "surf": {
        "epochs": 30, "batch_size": 64, "optimizer": "adam", "lr": 1e-4, "final_lr": 0.0,
        "momentum": 0.0,
        "weight_decay": 5e-3, "nesterov": False, "warmup_epochs": 0, "label_smoothing": 0.0,
        "augment": False, "class_weighted": True, "normalize": "none", "amp": False,
    },
}
DEFAULT_RECIPE = RECIPES["cifar10"]


def _gpu_augment(x_uint8, pad=4):
    """Random crop with zero padding + random horizontal flip, on GPU, uint8 in/out."""
    b, c, h, w = x_uint8.shape
    dev = x_uint8.device
    padded = F.pad(x_uint8, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, :, None]  # [b,h,1]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, :]  # [b,1,w]
    bidx = torch.arange(b, device=dev)[:, None, None]
    out = padded.permute(0, 2, 3, 1)[bidx, rows, cols].permute(0, 3, 1, 2)  # [b,c,h,w]
    flip = torch.rand(b, device=dev) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()


def normalize_batch(x_uint8, mode):
    """uint8 -> float input exactly as the evaluation loaders produce it."""
    if mode == "cifar10":
        mean = torch.tensor(CIFAR10_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, device=x_uint8.device).view(1, 3, 1, 1)
        return (x_uint8.float().div_(255.0) - mean) / std
    if mode == "none":
        return x_uint8.float()
    raise ValueError(f"unknown normalize mode {mode!r}")


def train_from_scratch(x_uint8, y, indices, seed, recipe="cifar10", overrides=None,
                       arch="resnet18", num_classes=10, log_every=0):
    """Train a fresh model on ``x_uint8[indices]``.

    Args:
        x_uint8: [N, 3, H, W] uint8 tensor of the training pool.
        y: [N] int64 labels.
        indices: training indices (retain-only for the oracle, all for the original).
        seed: controls init, batch order and augmentation.
        recipe: key of :data:`RECIPES` (or a full recipe dict).
        overrides: dict of recipe overrides.
        arch: architecture name from :data:`unlearn.models.ARCHITECTURES`.
    Returns:
        (model in eval mode, recipe dict actually used)
    """
    cfg = dict(RECIPES[recipe] if isinstance(recipe, str) else recipe)
    cfg.update(overrides or {})
    seed_everything(seed)

    idx = torch.as_tensor(indices, dtype=torch.long)
    xs = x_uint8[idx].to(DEVICE)
    ys = y[idx].to(DEVICE)
    n = xs.shape[0]

    net = build_model(arch, num_classes).to(DEVICE).to(memory_format=torch.channels_last)
    if cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
    elif cfg["optimizer"] == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")

    steps_per_epoch = math.ceil(n / cfg["batch_size"])
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch

    floor = cfg.get("final_lr", 0.0) / cfg["lr"]  # cosine decays from lr to final_lr

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        cos = 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        return floor + (1 - floor) * cos

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(cfg["amp"]) and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weight = None
    if cfg["class_weighted"]:
        counts = torch.bincount(ys, minlength=num_classes).float()
        weight = torch.where(counts > 0, 1.0 / counts, torch.ones_like(counts))
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    net.train()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n, device=DEVICE, generator=gen)
        running, seen = 0.0, 0
        for s in range(steps_per_epoch):
            b = perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]
            xb = xs[b]
            if cfg["augment"]:
                xb = _gpu_augment(xb)
            xb = normalize_batch(xb, cfg["normalize"]).contiguous(memory_format=torch.channels_last)
            yb = ys[b]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(net(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(b)
            seen += len(b)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1}/{cfg['epochs']} loss {running / seen:.4f}", flush=True)

    net = net.to(memory_format=torch.contiguous_format).eval()
    del xs, ys
    return net, cfg
