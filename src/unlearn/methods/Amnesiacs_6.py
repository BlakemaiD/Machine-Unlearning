"""Amnesiacs challenge method: reset layers, distill, then repair on retain data.

The competition notebook reinitializes ResNet-18's first convolution (conv1)
and final classifier (fc), warms up on validation age groups 0 and 1 using the
original model as teacher, and then trains on retain data with classification
and teacher-distillation losses.
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn, optim

from ..device import DEVICE


def _unpack_batch(sample):
    """Support CIFAR tuple batches and the competition's dictionary batches."""
    if isinstance(sample, dict):
        images = sample.get("image", sample.get("images"))
        labels = None
        for key in ("label", "labels", "target", "targets", "age_group"):
            if key in sample:
                labels = sample[key]
                break
        if images is None or labels is None:
            raise KeyError(
                f"Could not find image and label fields in batch keys {list(sample)}."
            )
        return images, labels
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]
    raise TypeError("A batch must be a mapping or an (images, labels) pair.")


def kl_loss(model_logits, teacher_logits, temperature=1.0):
    """KL divergence from teacher probabilities to student probabilities."""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(model_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def soft_cross_entropy(student_logits, soft_targets):
    """Cross-entropy between student logits and a teacher probability target."""
    return torch.sum(
        -soft_targets * F.log_softmax(student_logits, dim=1),
        dim=1,
    ).mean()


def _keep_validation_classes(images, labels, validation_classes):
    if validation_classes is None:
        return images, labels
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in validation_classes:
        keep |= labels == class_id
    if not torch.any(keep):
        return None, None
    return images[keep], labels[keep]


def unlearning(
    net,
    retain,
    forget,
    validation,
    *,
    teacher=None,
    temperature=5.0,
    warmup_epochs=3,
    retain_epochs=3,
    warmup_lr=9e-4,
    retain_lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
    class_weight=None,
    validation_classes=(0, 1),
):
    """Unlearn by resetting conv1/fc, distilling, then fine-tuning on retain.

    Args follow the repository's common method interface. ``teacher`` may be
    supplied by a caller; otherwise a frozen copy of ``net`` is made before
    its layers are reset. The competition filtered validation data to age
    groups 0 and 1; ``validation_classes`` applies that filter to ordinary
    tuple and dictionary batches. Pass ``None`` to use all validation classes.
    The forget loader is accepted for interface compatibility; the notebook's
    method does not directly train on it.
    """
    del forget
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if warmup_epochs < 0 or retain_epochs <= 0:
        raise ValueError("warmup_epochs must be nonnegative and retain_epochs positive.")
    if validation is None:
        raise ValueError("A validation loader is required for the distillation warm-up.")
    if not hasattr(net, "conv1") or not hasattr(net, "fc"):
        raise TypeError("Amnesiacs requires a model with ResNet-style conv1 and fc layers.")

    net.to(DEVICE)
    if teacher is None:
        teacher = copy.deepcopy(net)
    teacher.to(DEVICE)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    net.conv1.reset_parameters()
    net.fc.reset_parameters()

    criterion = nn.CrossEntropyLoss(
        weight=(
            class_weight.to(DEVICE)
            if isinstance(class_weight, torch.Tensor)
            else class_weight
        )
    )
    warmup_optimizer = optim.SGD(
        net.parameters(),
        lr=warmup_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for _ in range(warmup_epochs):
        net.train()
        for sample in validation:
            images, labels = _unpack_batch(sample)
            images = images.to(DEVICE)
            labels = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
            images, _ = _keep_validation_classes(
                images, labels, validation_classes
            )
            if images is None:
                continue

            warmup_optimizer.zero_grad()
            student_logits = net(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = kl_loss(student_logits, teacher_logits, temperature)
            loss.backward()
            warmup_optimizer.step()

    retain_optimizer = optim.SGD(
        net.parameters(),
        lr=retain_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        retain_optimizer,
        T_max=retain_epochs,
    )

    for _ in range(retain_epochs):
        net.train()
        for sample in retain:
            images, labels = _unpack_batch(sample)
            images = images.to(DEVICE)
            labels = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)

            retain_optimizer.zero_grad()
            student_logits = net(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

            loss = soft_cross_entropy(
                student_logits / temperature,
                teacher_probs,
            )
            loss = loss + criterion(student_logits, labels)
            loss = loss + kl_loss(
                student_logits,
                teacher_logits,
                temperature,
            )
            loss.backward()
            retain_optimizer.step()
        scheduler.step()

    net.eval()
    return net
