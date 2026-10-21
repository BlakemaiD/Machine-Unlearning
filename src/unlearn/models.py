from pathlib import Path

import requests
import torch
from torchvision.models import resnet18

from .device import DEVICE


WEIGHTS_URL = "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"


def load_cifar10_model(weights_path="weights_resnet18_cifar10.pth"):
    """Load the challenge's CIFAR-10-trained ResNet-18 onto the selected device."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        response = requests.get(WEIGHTS_URL, timeout=60)
        response.raise_for_status()
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(response.content)

    model = resnet18(weights=None, num_classes=10)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()