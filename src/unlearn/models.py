"""Model construction / loading.

Architectures (all torchvision, random init, ``num_classes`` outputs):

* ``resnet18``  - the challenge architecture (ImageNet stem). The challenge
  provides a CIFAR-10 checkpoint for it (``weights_resnet18_cifar10.pth``).
* ``mobilenet_v2`` and ``efficientnet_b0`` - next options; no public
  CIFAR-10 checkpoint, so originals are trained with :mod:`unlearn.train`.

Methods that need a specific layer (first conv, classifier head) use
:func:`get_first_conv` / :func:`get_classifier` so they work on every
architecture above.
"""

from pathlib import Path

import requests
import torch
from torch import nn
from torchvision.models import efficientnet_b0, mobilenet_v2, resnet18

from .device import DEVICE

WEIGHTS_URL = "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
RETRAIN_WEIGHTS_URL = (
    "https://storage.googleapis.com/unlearning-challenge/retrain_weights_resnet18_cifar10.pth"
)

ARCHITECTURES = {
    "resnet18": lambda n: resnet18(weights=None, num_classes=n),
    "mobilenet_v2": lambda n: mobilenet_v2(weights=None, num_classes=n),
    "efficientnet_b0": lambda n: efficientnet_b0(weights=None, num_classes=n),
}


def build_model(arch="resnet18", num_classes=10):
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](num_classes)


def build_resnet18(num_classes=10):
    """The challenge architecture: torchvision ResNet-18 (ImageNet stem), random init."""
    return build_model("resnet18", num_classes)


def get_classifier(net):
    """Final nn.Linear layer (ResNet ``fc``; MobileNet/EfficientNet ``classifier[-1]``)."""
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        return net.fc
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("model has no nn.Linear classifier")
    return linears[-1]


def get_first_conv(net):
    """First nn.Conv2d of the network (ResNet ``conv1``)."""
    return next(m for m in net.modules() if isinstance(m, nn.Conv2d))


def _download(url, path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def load_checkpoint(path, arch="resnet18", num_classes=10):
    """Load a state dict (plain or ``{"state_dict": ...}``) onto DEVICE, eval mode."""
    model = build_model(arch, num_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_cifar10_model(weights_path="weights_resnet18_cifar10.pth"):
    """Load the challenge's CIFAR-10-trained ResNet-18 onto the selected device."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        _download(WEIGHTS_URL, weights_path)
    return load_checkpoint(weights_path)


def load_cifar10_retrained_model(weights_path="retrain_weights_resnet18_cifar10.pth"):
    """Load the challenge's single retrain-from-scratch reference (trained on retain only)."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        _download(RETRAIN_WEIGHTS_URL, weights_path)
    return load_checkpoint(weights_path)
