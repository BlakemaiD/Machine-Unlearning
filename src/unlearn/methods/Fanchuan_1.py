"""1st place, NeurIPS 2023 Machine Unlearning (team "fanchuan").

Two stages:

1. **Forget (1 pass over the forget set):** minimise KL(uniform || softmax(f(x)))
   on forget batches, pushing forget predictions towards a uniform distribution.
2. **Adversarial fine-tuning (8 epochs), each epoch =**
   a. *forget round*: for each forget batch paired with a random retain batch,
      minimise ``mean(-log_softmax(z_f @ z_r.T / tau))`` (a temperature-scaled
      dot-product contrastive loss on logits; retain logits detached). This
      pushes forget logits away from retain logits.
   b. *retain round*: one epoch of cross-entropy on the retain set (batch 256).

Sources: competition winner write-up (Kaggle discussion 458721), reproduction in
references/code/hishamcse_NeurIPS_Machine_Unlearning/Codes/neurips-machine-unlearning-1st-place.ipynb,
DeepUnlearn port ``kgltop1.py``. Hyperparameters below are the submitted ones.
The forget loader is used as given (competition batch size: 64).
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": 1, "team": "fanchuan", "family": "gradient-ascent-like + contrastive"}

DEFAULTS = {
    # stage 1
    "kl_lr": 0.005, "kl_momentum": 0.9, "kl_weight_decay": 0.0,
    # stage 2
    "epochs": 8,
    "retain_batch_size": 256,
    "temperature": 1.15,
    "forget_lr": 3e-4, "forget_momentum": 0.9, "forget_weight_decay": 0.0,
    "eta_min": 1e-6,
    "retain_lr": 0.001,  # scaled by retain_batch_size / 64, as in the original
    "retain_momentum": 0.9, "retain_weight_decay": 0.01,
}


def _kl_to_uniform(logits):
    uniform = torch.full_like(logits, 1.0 / logits.shape[1])
    return nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=-1)(logits), uniform)


def _contrastive(out_forget, out_retain, tau):
    return (-1.0 * nn.LogSoftmax(dim=-1)(out_forget @ out_retain.T / tau)).mean()


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "Fanchuan_1")
    net.to(DEVICE)

    # ---- Stage 1: KL to uniform on the forget set -------------------------
    optimizer = optim.SGD(net.parameters(), lr=hp["kl_lr"], momentum=hp["kl_momentum"],
                          weight_decay=hp["kl_weight_decay"])
    net.train()
    for inputs, _ in forget:
        inputs = inputs.to(DEVICE)
        optimizer.zero_grad()
        _kl_to_uniform(net(inputs)).backward()
        optimizer.step()

    # ---- Stage 2: alternating forget / retain rounds ----------------------
    epochs, bs = hp["epochs"], hp["retain_batch_size"]
    criterion_retain = nn.CrossEntropyLoss()
    optimizer_forget = optim.SGD(net.parameters(), lr=hp["forget_lr"],
                                 momentum=hp["forget_momentum"],
                                 weight_decay=hp["forget_weight_decay"])
    optimizer_retain = optim.SGD(net.parameters(), lr=hp["retain_lr"] * bs / 64,
                                 momentum=hp["retain_momentum"],
                                 weight_decay=hp["retain_weight_decay"])
    scheduler_forget = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_forget, T_max=epochs * len(forget), eta_min=hp["eta_min"])

    # Two independently shuffled retain loaders (one per round), as in the original.
    retain_for_forget = DataLoader(retain.dataset, batch_size=bs, shuffle=True)
    retain_for_retain = DataLoader(retain.dataset, batch_size=bs, shuffle=True)

    for _ in range(epochs):
        net.train()
        for (x_f, _), (x_r, _) in zip(forget, retain_for_forget):
            x_f, x_r = x_f.to(DEVICE), x_r.to(DEVICE)
            optimizer_forget.zero_grad()
            out_f, out_r = net(x_f), net(x_r).detach()
            _contrastive(out_f, out_r, hp["temperature"]).backward()
            optimizer_forget.step()
            scheduler_forget.step()

        for x, y in retain_for_retain:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_retain.zero_grad()
            criterion_retain(net(x), y).backward()
            optimizer_retain.step()

    net.eval()
    return net
