# Local CIFAR-10 Unlearning Notes

This document tracks the local CIFAR-10 unlearning experiment: what is
implemented, how the notebook is organized, and how to add competition
algorithms while keeping comparisons reproducible.

## Project layout

```text
Notebooks/
  01_starter_kit.ipynb   # Original starter-kit notebook
  02_local_test.ipynb    # Bounded smoke test for baseline and registered methods
  03_data_comparison.ipynb # CIFAR split sizes and class distributions
  04_official_score.ipynb  # Multi-seed official-style score comparison
src/unlearn/
  data.py                # CIFAR-10 datasets and data loaders
  device.py              # Shared compute-device selection
  models.py              # Challenge model and pretrained weights
  metrics.py             # Accuracy, loss-MIA, and official confidence helpers
  official_metric.py     # CPU implementation of competition forgetting quality F
  evaluate.py            # Retraining cache, method runner, and final score
  methods/
    finetune.py          # Retain-set fine-tuning baseline
    Fauchan_1.py         # Uniform-forget and retain/forget contrast method
    Kookmin_2.py         # Gradient-ranked convolution reinitialization
    Seif_3.py            # Convolution perturbation and weighted retain tuning
    Amnesiacs_6.py       # conv1/fc reset, distillation, and retain repair
```

The starter-kit notebook is the CIFAR-10 and pretrained-checkpoint reference.
The three local notebooks have separate purposes: `02_local_test.ipynb` checks
that each registered method runs on fixed small data subsets, `03_data_comparison.ipynb`
explains the split sizes and class balance, and `04_official_score.ipynb`
performs the multi-seed comparison against retain-only retraining references.

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

## Methods and metrics

Each method is registered in `src/unlearn/methods/__init__.py` under `METHODS`.
The shared call shape is `unlearning(net, retain, forget, validation, ...)`;
each notebook gives every method a fresh copy of the same original model.

`finetune.unlearning(...)` is the retain-only baseline: it fine-tunes the
starting model and does not use the forget or validation loaders.
`Kookmin_2.unlearning(...)` scores convolution gradients using retain and
negated forget losses, reinitializes the selected low-gradient weights, then
fine-tunes on retain data. `Amnesiacs_6.unlearning(...)` copies an original
teacher, resets the student ResNet-18 `conv1` and `fc`, distills using
validation classes 0/1, and repairs on retain data. The original notebook's
soft-cross-entropy call applies `log_softmax` twice; the port preserves that
reference behavior. Its class-0/1 filtering mirrors CASIA age groups only
approximately on CIFAR-10. The Amnesiacs method accepts but does not directly
use the forget loader, matching its source notebook.

`Fauchan_1` first encourages uniform outputs on forget examples, then alternates
a forget/retain contrastive-style update and retain-label training.
`Seif_3` perturbs convolution weights and fine-tunes on retain data with
class-weighted loss. Treat these as local ports of the code in their method
files; passing a smoke test does not prove equivalence to a competition
submission.

`accuracy()` is the legacy accuracy helper and does not set evaluation mode.
Use `accuracy_eval()` for model scoring; it temporarily sets `eval()` and
restores the previous mode. `compute_losses()` returns per-example
cross-entropy losses. `mia_accuracy()` runs a logistic-regression attack that
distinguishes forget from test losses; it is a secondary diagnostic, not the
competition forgetting metric.

`get_forget_confs()` calculates correct-class logit-scaled confidences, in the
stable example order provided by a non-shuffled loader. `official_metric.py`
compares these confidence distributions over unlearned and retain-only
retrained models to calculate forgetting quality **F**. `evaluate.py` combines
F with retain/test accuracy ratios to produce the final score and compares
unlearning time with retraining time. Accuracy and confidence forward passes
use `DEVICE`; the score math itself uses CPU NumPy float64 arrays.

## What each notebook outputs

### `02_local_test.ipynb`: smoke-test diagnostics

The notebook loads the pretrained checkpoint once and runs the baseline plus
all methods in `METHODS`. Each method starts from a fresh model copy and uses
the same sampled data: 2,048 retain examples, 256 forget examples, and 512
validation examples. It prints a one-run table with:

- retain accuracy on the sampled retain subset;
- test accuracy on the full held-out test split;
- forget accuracy on the sampled forget subset;
- elapsed method time (not including evaluation);
- loss-based MIA accuracy using 256 forget and 256 test examples.

These settings are for fast integration checks. They do not measure official F
or serve as final benchmark results. The fixed sample seeds make the smoke
inputs repeatable; the method's own optimization may still be device-dependent.

### `03_data_comparison.ipynb`: split and class balance

The notebook prints sizes and per-class counts for train, retain, forget,
validation, and test, and plots class percentages for retain/forget/validation/
test. It checks that retain plus forget equals train. This explains the local
CIFAR-10 data; it does not train a model or score forgetting.

### `04_official_score.ipynb`: official-style model comparison

The notebook trains (or reuses a matching cache of) N scratch-initialized
retain-only reference models, then runs the registered unlearning methods from
fresh copies of the challenge checkpoint with the same seeds. It prints F,
retain/test accuracy, final score, runtime ratio, loss-MIA, and N. The default
N=8 and 256-example forget subsample are development settings; compare methods
only with identical seeds, N, and forget examples. Use the full forget set for
final results.

The original checkpoint's full training duration/schedule is undocumented in
the starter kit. `TrainingRecipe` therefore records a chosen 30-epoch cosine
scratch-training recipe using the documented CIFAR-10 SGD, learning rate,
momentum, weight decay, and augmentations. Results using it are local
comparisons, not an exact reproduction of the checkpoint-training recipe.

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

## Running the local notebooks

1. Open a notebook and select the Python environment with
   PyTorch, torchvision, NumPy, requests, and scikit-learn installed.
2. Restart the notebook kernel after changing files under `src/unlearn/`.
   Python caches imported modules in a running kernel, so editing `device.py`
   does not automatically replace an already-imported `unlearn.device`.
3. Run all cells from the top, in order.
4. Start with `02_local_test.ipynb` to verify methods, then use
   `03_data_comparison.ipynb` to inspect data, and `04_official_score.ipynb` for
   the multi-seed score.
5. Check the first cell's device output. On a compatible Apple Silicon PyTorch
   environment it should report MPS; otherwise the code falls back to CPU.
6. The smoke-test uses bounded subsets and does not replace full evaluation.
   The official notebook can require substantial compute because it trains
   multiple scratch models.

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
