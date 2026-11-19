"""Shared paths / argument helpers for the pipeline scripts.

Artifact layout (everything under ``--out``, default ``artifacts/``)::

    <out>/<dataset>/<arch>/<version>/       # version: v1, v2, ... (never overwritten)
        reference/
            original/seed_0000.pt  .npz     # our own original model (all train data)
            retrain/seed_XXXX.pt   .npz     # N retrain-from-scratch oracle models
        <orig>/                             # which original the methods start from:
            runs/<method>/<tag>/run_XXXX.npz    #   "own" or "challenge"
            models/<method>/<tag>/run_XXXX.pt  .json   # checkpoint + model card
            logs/
            results/                        # written by evaluate.py

Every ``.npz`` holds the model's logits on the fixed-order forget / retain /
test / val sets plus a JSON ``meta`` record (seed, hyperparameters, timing,
peak GPU memory, software versions).
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CHALLENGE_WEIGHTS = {("cifar10", "resnet18"): REPO / "weights_resnet18_cifar10.pth"}


def exp_dir(out, dataset, arch, version):
    return Path(out) / dataset / arch / version


def reference_path(out, dataset, arch, version, role, seed, ext):
    return exp_dir(out, dataset, arch, version) / "reference" / role / f"seed_{seed:04d}.{ext}"


def run_path(out, dataset, arch, version, orig, method, tag, seed, ext="npz"):
    sub = "runs" if ext == "npz" else "models"
    return (exp_dir(out, dataset, arch, version) / orig / sub / method / tag
            / f"run_{seed:04d}.{ext}")


def code_hash():
    """sha256 over src/unlearn/**/*.py: identifies the exact code that produced a model."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted((REPO / "src" / "unlearn").rglob("*.py")):
        h.update(f.relative_to(REPO).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def save_model(net, path, card):
    """Save a checkpoint plus a JSON model card (how it was made, how it scored)."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    path.with_suffix(".json").write_text(dump_json(card) + "\n")


def parse_kv(pairs):
    """['lr=0.01', 'epochs=3', 'fine_lr=[0.1,0.2]'] -> dict (values parsed as YAML)."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def add_common_args(p):
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "surf"])
    p.add_argument("--arch", default="resnet18",
                   choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    p.add_argument("--data-root", default=str(REPO / "data"),
                   help="CIFAR-10 download dir, or the CASIA-SURF root for --dataset surf")
    p.add_argument("--forget-idx", default=str(REPO / "Notebooks" / "forget_idx.npy"))
    p.add_argument("--out", default=str(REPO / "artifacts"))
    p.add_argument("--version", default="v1",
                   help="experiment version folder; bump it when recipe/code changes")
    return p


def environment_info():
    import numpy
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "host": platform.node(), "code_sha": code_hash()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def dump_json(obj):
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
