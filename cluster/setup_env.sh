#!/usr/bin/env bash
# One-time setup on the cluster (run on head0 or dijkstra; /home is shared).
#   bash ~/Projects/MachineUnlearning/cluster/setup_env.sh
# Creates conda env "unlearn" with a PyTorch build that supports the
# RTX PRO 6000 Blackwell GPUs (sm_120 needs CUDA >= 12.8 wheels), then
# downloads CIFAR-10, the forget indices and the challenge checkpoints, so
# that the GPU jobs never download anything concurrently.
set -euo pipefail
PROJ=${PROJ:-$HOME/Projects/MachineUnlearning}
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^unlearn '; then
  conda create -y -n unlearn python=3.11
fi
conda activate unlearn
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-learn pandas matplotlib pyyaml requests tqdm tabulate jupyter ipykernel

cd "$PROJ"
python - <<'PY'
import sys; sys.path.insert(0, "src")
import torch, torchvision
from unlearn.models import load_cifar10_model, load_cifar10_retrained_model
from unlearn.data import _get_forget_idx
torchvision.datasets.CIFAR10("data", train=True, download=True)
torchvision.datasets.CIFAR10("data", train=False, download=True)
_get_forget_idx("Notebooks/forget_idx.npy")
load_cifar10_model("weights_resnet18_cifar10.pth")
load_cifar10_retrained_model("retrain_weights_resnet18_cifar10.pth")
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("setup done")
PY
