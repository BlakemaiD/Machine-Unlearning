"""Baseline: SCRUB (Kurmanji et al., NeurIPS 2023).

Teacher = frozen original model, student = model being unlearned. Each epoch:

* *max-step* (first ``msteps`` epochs only): on forget batches, maximise
  KL(teacher || student) (move away from the teacher on forget data);
* *min-step*: on retain batches, minimise
  ``gamma * CE(student, y) + alpha * KL(teacher || student)``.

KL uses temperature ``kd_t`` (scaled by T^2, as in the reference
implementation, RepDistiller's DistillKL). LR decays by ``lr_decay_rate`` at
``lr_decay_epochs``. Defaults follow the SCRUB repository's CIFAR settings
(github.com/meghdadk/SCRUB, ``small_scale_unlearning.ipynb``); SCRUB's optional
SWA and the "rewind" checkpoint selection are not used.
"""

import copy

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "baseline (Kurmanji et al. 2023)", "family": "teacher-student max/min"}

DEFAULTS = {
    "epochs": 3,
    "msteps": 2,
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_decay_epochs": (3, 5, 9),
    "lr_decay_rate": 0.1,
    "kd_t": 4.0,
    "gamma": 0.99,
    "alpha": 0.001,
    "retain_batch_size": 128,
    "forget_batch_size": 32,
}


def _distill_kl(student_logits, teacher_logits, t):
    p_s = F.log_softmax(student_logits / t, dim=1)
    p_t = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / student_logits.shape[0]


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "scrub")
    net.to(DEVICE)
    teacher = copy.deepcopy(net).eval()
    retain_ld = DataLoader(retain.dataset, batch_size=hp["retain_batch_size"], shuffle=True)
    forget_ld = DataLoader(forget.dataset, batch_size=hp["forget_batch_size"], shuffle=True)
    optimizer = optim.SGD(net.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])

    for epoch in range(1, hp["epochs"] + 1):
        steps = sum(epoch > e for e in hp["lr_decay_epochs"])
        for g in optimizer.param_groups:
            g["lr"] = hp["lr"] * (hp["lr_decay_rate"] ** steps)

        net.train()
        if epoch <= hp["msteps"]:
            for x, _ in forget_ld:
                x = x.to(DEVICE)
                with torch.no_grad():
                    t_out = teacher(x)
                optimizer.zero_grad()
                (-_distill_kl(net(x), t_out, hp["kd_t"])).backward()
                optimizer.step()

        for x, y in retain_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_out = teacher(x)
            optimizer.zero_grad()
            out = net(x)
            loss = hp["gamma"] * F.cross_entropy(out, y) + hp["alpha"] * _distill_kl(out, t_out, hp["kd_t"])
            loss.backward()
            optimizer.step()

    net.eval()
    return net

