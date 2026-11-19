"""SOTA: Bad Teacher / "blindspot" unlearning (Chundawat et al., AAAI 2023).

Student = the original model. Two frozen teachers: the *competent* teacher
(the original model) and an *incompetent* teacher (a randomly initialised
network of the same architecture). On a mix of forget samples and a subset of
retain samples, the student minimises KL to

    p_target = u * softmax(bad_teacher / T) + (1 - u) * softmax(good_teacher / T)

where u = 1 for forget samples and 0 for retain samples.

Defaults are the authors' random-sample setting as run in the SSD authors'
benchmark code (references/code/if-loops_selective-synaptic-dampening,
``forget_random_strategies.blindspot``): 1 epoch, Adam lr 1e-4, batch 256,
T = 1, all forget samples + a random 30% of the retain set.
Note: the reference code calls ``F.kl_div`` with its default reduction
("mean" over all elements); we keep that so the effective step size matches.
"""

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "SOTA (Chundawat et al. 2023)", "family": "competent/incompetent teacher distillation"}

DEFAULTS = {"epochs": 1, "lr": 1e-4, "batch_size": 256, "temperature": 1.0, "retain_fraction": 0.3}


class _Tagged(Dataset):
    """Return (x, y, is_forget) for every item of ``ds``."""

    def __init__(self, ds, flag):
        self.ds, self.flag = ds, flag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, torch.as_tensor(y), torch.tensor(self.flag)


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "bad_teacher")
    net.to(DEVICE)
    good = copy.deepcopy(net).eval()
    bad = copy.deepcopy(net)
    for m in bad.modules():  # random re-initialisation of the same architecture
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    bad.eval()

    n_ret = int(hp["retain_fraction"] * len(retain.dataset))
    sub = torch.randperm(len(retain.dataset))[:n_ret].tolist()
    data = ConcatDataset([_Tagged(forget.dataset, 1), _Tagged(Subset(retain.dataset, sub), 0)])
    loader = DataLoader(data, batch_size=hp["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    t = hp["temperature"]

    for _ in range(hp["epochs"]):
        net.train()
        for x, _, u in loader:
            x, u = x.to(DEVICE), u.to(DEVICE).float().unsqueeze(1)
            with torch.no_grad():
                target = u * F.softmax(bad(x) / t, 1) + (1 - u) * F.softmax(good(x) / t, 1)
            optimizer.zero_grad()
            F.kl_div(F.log_softmax(net(x) / t, 1), target).backward()
            optimizer.step()
    net.eval()
    return net
