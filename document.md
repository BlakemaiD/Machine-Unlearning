# Local CIFAR-10 Unlearning Notes

This document describes how the local experiment notebook uses the Python
modules under `src/unlearn/` to reproduce the CIFAR-10 starter-kit workflow.

## Project layout

```text
Notebooks/
  01_starter_kit.ipynb   # Original starter-kit notebook
  02_local_test.ipynb    # Local experiment driver
src/unlearn/
  data.py                # CIFAR-10 datasets and data loaders
  device.py              # Shared compute-device selection
  models.py              # Challenge model and pretrained weights
  metrics.py             # Accuracy and membership-inference metrics
  methods/
    finetune.py          # Retain-set fine-tuning unlearning method
    Fauchan_1.py         # 1st place code
    Kookmin_2.py         # 2nd place code
    Seif_3.py            # 3rd place code
    Amnesiacs_6.py       # 6th place code
```

The starter-kit notebook is the reference implementation. The local notebook is
kept lean: it finds `src/`, imports the package functions, creates loaders and a
model, runs a method, and reports metrics.

## Compute device

`src/unlearn/device.py` defines a shared PyTorch `DEVICE`:

1. CUDA when `torch.cuda.is_available()` is true.
2. Apple MPS when the MPS backend exists and is available.
3. CPU otherwise.

The training method, metrics, and model loader all use this value. The model and
each input/target batch must be on the same device before a forward pass.
`02_local_test.ipynb` prints the selected device when its first cell runs.

## Model loading

`load_cifar10_model()` in `src/unlearn/models.py` builds a ResNet-18 with
10 output classes, downloads the challenge's CIFAR-10 pretrained weights if the
requested weight file does not exist, loads the state dictionary, moves the
model to `DEVICE`, and returns it in evaluation mode.

The function defaults to `weights_resnet18_cifar10.pth` in the current working
directory. The notebook passes `repo_root / "weights_resnet18_cifar10.pth"` so
the weight file is stored at the repository root. Weight files should remain
local and are not intended to be committed.

## Dataset loaders

`get_cifar10_loaders()` in `src/unlearn/data.py` downloads/loads CIFAR-10 and
returns a dictionary with:

- `train`: the complete CIFAR-10 training set.
- `test` and `val`: a reproducible 50/50 split of the held-out set.
- `forget` and `retain`: training subsets from the challenge's forget indices.

The forget-index file is downloaded when missing. Loader options such as batch
size, worker count, dataset root, and random seed can be passed as arguments to
`get_cifar10_loaders()`.

## Fine-tuning method and metrics

`finetune.unlearning(net, retain, forget, validation, ...)` moves the network to
`DEVICE`, trains it on the retain loader, and returns it in evaluation mode.
The current implementation accepts the forget and validation loaders to match
the experiment interface, but does not use them during this fine-tuning method.

`accuracy()` reports classification accuracy. `mia_accuracy()` estimates
forget-vs-test membership-inference accuracy using per-example losses and
cross-validation.

## Running the local notebook

1. Open `Notebooks/02_local_test.ipynb` and select the Python environment with
   PyTorch, torchvision, NumPy, requests, and scikit-learn installed.
2. Restart the notebook kernel after changing files under `src/unlearn/`.
   Python caches imported modules in a running kernel, so editing `device.py`
   does not automatically replace an already-imported `unlearn.device`.
3. Run all cells from the top, in order.
4. Check the first cell's device output. On a compatible Apple Silicon PyTorch
   environment it should report MPS; otherwise the code falls back to CPU.
5. The model cell loads the challenge weights, then the last cell runs
   fine-tuning and reports retain accuracy, test accuracy, and MIA.

If an import error persists after a kernel restart, inspect the module path and
exported names:

```python
import unlearn.device

print(unlearn.device.__file__)
print([name for name in dir(unlearn.device) if name.isupper()])
```

The path should point to this repository's `src/unlearn/device.py`, and the
exported names should include `DEVICE`. If not, verify the repository root and
`src/` path added to `sys.path` in the notebook's first cell.
