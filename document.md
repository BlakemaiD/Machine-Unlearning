# Local CIFAR-10 Unlearning Notes

This document tracks the local CIFAR-10 unlearning experiment: what is
implemented, how the notebook is organized, and how to add competition
algorithms while keeping comparisons reproducible.

## Project layout

```text
Notebooks/
  01_starter_kit.ipynb   # Original starter-kit notebook
  02_local_test.ipynb    # Local experiment driver and comparison notebook
src/unlearn/
  data.py                # CIFAR-10 datasets and data loaders
  device.py              # Shared compute-device selection
  models.py              # Challenge model and pretrained weights
  metrics.py             # Accuracy and membership-inference metrics
  methods/
    finetune.py          # Retain-set fine-tuning unlearning method
    Fauchan_1.py         # Competition method integration in progress
    Kookmin_2.py         # Competition method integration in progress
    Seif_3.py            # Competition method integration in progress
    Amnesiacs_6.py       # Competition method integration in progress
```

The starter-kit notebook is the reference implementation. The local notebook is
kept as an experiment driver: it finds `src/`, imports package functions,
creates loaders and a starting model, runs an unlearning method, reports
metrics, and plots a comparison. The current `02_local_test.ipynb` run checks
that this local setup can execute a method through the challenge-style
interface. The first algorithm is a retain-set fine-tuning baseline; the
competition algorithms are intended to be added one at a time under
`src/unlearn/methods/`.

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
local and are not intended to be committed. This is to ensure that the weights are not cosntantly adjusted/change.

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
cross-validation. The MIA score is a diagnostic rather than proof that data has
been erased; interpret it together with utility and other evaluations.

## Current experiment status

The local notebook runs in this order:

1. **Environment and imports:** adds the repository's `src/` directory to
   `sys.path`, imports the shared helpers, and prints the selected device.
2. **Data:** loads CIFAR-10 and creates `train`, `test`, `val`, `forget`, and
   `retain` data loaders.
3. **Starting model:** loads the challenge-pretrained model as `model`.
4. **Baseline run:** deep-copies `model`, calls
   `finetune.unlearning(...)`, and prints retain accuracy, test accuracy, and
   forget-vs-test MIA.
5. **Visualization:** plots loss distributions for the original and baseline
   models. Assigning the returned figure to `fig` avoids the duplicate inline
   display that can occur when a Matplotlib figure is both returned as the
   cell's final expression and auto-displayed by the notebook.

The fine-tuning baseline is primarily a pipeline check and comparison
reference. It trains on retain data only; it does not use the forget or
validation loaders. The competition-method files are not yet integrated into
the local API. Treat them as work in progress until each contains a working
method function and has been run through the local evaluation workflow.

## Adding a competition method

The cleanest division of responsibility is:

- Keep algorithm-specific training logic in its own module under
  `src/unlearn/methods/`.
- Keep `02_local_test.ipynb` responsible for inputs, invoking methods,
  evaluation, and plots—not for implementing the algorithm itself.
- Keep shared evaluation functions in `metrics.py` and shared dataset/device
  setup in their existing modules.

For each algorithm, adapt its code into a callable function with a consistent
interface, for example:

```python
def unlearning(net, retain, forget, validation, **kwargs):
    # Run this algorithm and return the modified net.
    ...
```

Use the exact arguments the algorithm actually needs; do not add unused
arguments merely to make the signature look uniform. A common interface is
useful when it fits, but the implementation should make any method-specific
requirements explicit. Prefer imports at the top of the module and avoid
executing a training run, referring to notebook variables such as `model`, or
calling an unavailable `competition` module when the file is imported. Importing
a method module should define functions and constants; the notebook should
decide when to run training.

Recommended integration sequence:

1. **Read the original method and its requirements.** Record its optimizer,
   hyperparameters, number of epochs, expected data format, and any extra
   dependencies or helper code. Preserve its algorithm and settings when the
   goal is to reproduce it; separate deliberate local adaptations from the
   faithful reproduction.
2. **Add one method module.** Make its entry point accept the model and the
   required loaders (or clearly document different inputs), move batches and
   model to the appropriate device, and return the trained/unlearned model.
   Keep method-specific utilities in that module or in a clearly named shared
   helper if another method will also use them.
3. **Export the method.** Add it to `src/unlearn/methods/__init__.py` so the
   notebook can import it from the package, following the existing `finetune`
   export pattern.
4. **Run it from the notebook on a fresh copy.** For example:

   ```python
   import copy
   from unlearn.methods import finetune, Fauchan_1

   method_models = {
       "Fine-tune": finetune.unlearning(
           copy.deepcopy(model), retain_loader, forget_loader, val_loader
       ),
       "Fauchan": Fauchan_1.unlearning(
           copy.deepcopy(model), retain_loader, forget_loader, val_loader
       ),
   }
   ```

   Each method must start from `copy.deepcopy(model)`, not from another
   method's output. Otherwise the order of notebook execution changes the
   experiment and methods are no longer compared from the same starting point.
   Python module names are case-sensitive and must match the filename; the
   example uses the current `Fauchan_1.py` filename.
5. **Evaluate consistently.** For every returned model, report the same
   retain accuracy, test accuracy, and MIA, and add it to the same plotting
   comparison. Use validation data only for choices such as checkpoint
   selection or hyperparameter tuning. Keep the test set for final evaluation,
   rather than repeatedly tuning against test results.
6. **Check the baseline and reference behavior.** Confirm the new method
   imports, runs on a small smoke-test configuration if possible, and returns a
   model that can be evaluated. Compare preprocessing, data splits, training
   settings, and metrics with the original implementation. A successful run
   alone does not establish that a port reproduces the competition algorithm.
7. **Record the run settings and outcomes.** Keep method name, key
   hyperparameters, seed, device, and metrics together in notebook output or a
   structured results table. This makes later comparisons interpretable.

If a competition implementation has a substantially different interface or
requires custom preprocessing, keep that adaptation explicit in its module
rather than hiding it in notebook cells. If multiple methods share meaningful
logic, extract that logic only after the first integration demonstrates what
is actually common.

## Running the local notebook

1. Open `Notebooks/02_local_test.ipynb` and select the Python environment with
   PyTorch, torchvision, NumPy, requests, and scikit-learn installed.
2. Restart the notebook kernel after changing files under `src/unlearn/`.
   Python caches imported modules in a running kernel, so editing `device.py`
   does not automatically replace an already-imported `unlearn.device`.
3. Run all cells from the top, in order.
4. Check the first cell's device output. On a compatible Apple Silicon PyTorch
   environment it should report MPS; otherwise the code falls back to CPU.
5. The model cell loads the challenge weights. The baseline cell runs
   fine-tuning and reports retain accuracy, test accuracy, and MIA. The plot
   cell compares the original model with the baseline; newly integrated
   methods should be added to this same evaluation flow, each from a fresh
   copy of the starting model.

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
