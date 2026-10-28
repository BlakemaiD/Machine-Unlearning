"""Port of the final configuration in Notebooks/KookminNotebook.ipynb.

SNIP-Little-Grad combines retain and negated forget gradients to score convolution weights.
It reinitializes the lowest-gradient fraction of each convolution, then fine-tunes on retain.
During fine-tuning, convolution-weight gradients are scaled to match the notebook's mask.

Hyperparameters: init_rate=0.3, lr=0.001, epoch=5, momentum=0.9,
weight_decay=5e-4, sch="linear" for unlearning().
"""

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from ..device import DEVICE 


def _ce_loss(net, x, y):
    return F.cross_entropy(net(x), y)


def unlearning(net, retain, forget, validation, *, loss_fn=None, **hparams):
    """Apply the notebook's SNIP-Little-Grad unlearning procedure."""
    defaults = {
        "init_rate": 0.3,
        "lr": 0.001,
        "epoch": 5,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "sch": "linear",
    }
    unexpected = set(hparams) - set(defaults)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"unlearning() got unexpected hyperparameter(s): {names}")
    defaults.update(hparams)

    if defaults["sch"] != "linear":
        raise ValueError("Kookmin_2 only ports the notebook's linear schedule.")
    if not 0 <= defaults["init_rate"] <= 1:
        raise ValueError("init_rate must be between 0 and 1.")

    loss_fn = loss_fn or _ce_loss
    net.to(DEVICE)

    indices = torch.randperm(
        len(retain.dataset), dtype=torch.int32, device="cpu"
    )[: len(forget.dataset)]
    gradient_retain = DataLoader(
        Subset(retain.dataset, indices), batch_size=64, shuffle=True
    )

    net.zero_grad()
    # Keep the incoming mode during gradient scoring, as in the notebook.
    for inputs, targets in gradient_retain:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        loss_fn(net, inputs, targets).backward()

    for inputs, targets in forget:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        (-loss_fn(net, inputs, targets)).backward()

    gradient_masks = []
    with torch.no_grad():
        for module in net.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.weight.grad is None:
                raise RuntimeError("A convolution weight has no SNIP gradient.")

            mask = torch.zeros_like(module.weight, dtype=torch.bool, device=DEVICE)
            count = round(defaults["init_rate"] * mask.numel())
            smallest_gradient = torch.topk(
                -module.weight.grad.abs().reshape(-1), k=count
            )
            mask.reshape(-1)[smallest_gradient.indices] = True

            # Keep the existing module installed while matching the notebook's initialization.
            reinitialized_conv = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
                device=DEVICE,
            )
            nn.init.kaiming_normal_(
                reinitialized_conv.weight, mode="fan_out", nonlinearity="relu"
            )
            reinitialized_conv.weight[~mask] = module.weight[~mask]
            module.weight.copy_(reinitialized_conv.weight)
            if module.bias is not None:
                module.bias.copy_(reinitialized_conv.bias)
            module.weight.grad = None
            if module.bias is not None:
                module.bias.grad = None

            gradient_mask = torch.full_like(module.weight, 0.1)
            gradient_mask[mask] = 1.0
            gradient_masks.append((module.weight, gradient_mask))

    epochs = defaults["epoch"]
    optimizer = optim.SGD(
        net.parameters(),
        lr=defaults["lr"],
        momentum=defaults["momentum"],
        weight_decay=defaults["weight_decay"],
    )
    annealing_steps = (epochs + 1) // 2
    total_steps = epochs + 1

    def linear_scale(last_epoch):
        step = last_epoch + 1
        if step <= annealing_steps:
            return step / annealing_steps
        return (total_steps - step) / (total_steps - annealing_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_scale)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(net, inputs, targets).backward()
            for weight, gradient_mask in gradient_masks:
                if weight.grad is not None:
                    weight.grad.mul_(gradient_mask)
            optimizer.step()
        scheduler.step()

    net.eval()
    return net
