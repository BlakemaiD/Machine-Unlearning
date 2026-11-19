# Machine Unlearning: Experiment Notes

This document tracks what is implemented, how to run it, and how to read the
results. We reproduce the top entries of the NeurIPS 2023 Machine Unlearning
Challenge, plus the main literature baselines. Each method is scored with the
competition's own metric and a set of complementary metrics. We run on
CIFAR-10 first, then on the competition's real data (CASIA-SURF) and other
architectures (MobileNetV2, EfficientNet-B0).

Status (2026-11-09): pipeline built and verified end to end on the cluster
with a smoke test (every method, 3 seeds, tiny reference models); the
100-epoch training recipe is calibrated (below). Runs happen on the dijkstra
cluster, on one node (g4, 8 GPUs).

## Project layout

```text
configs/experiments.yaml   # experiment matrix: datasets, architectures, N, methods
cluster/                   # setup_env.sh, run_pool.sbatch, sync.sh (dijkstra)
scripts/
  train_reference.py       # train one original / retrain-from-scratch model
  run_unlearning.py        # run one method once (one seed), save logits + model
  evaluate.py              # all metrics from saved runs -> results/ tables + plots
  make_tasks.py            # config -> task lists;  pool.py: run them on 8 GPUs
  check_metric_port.py     # our metric == official metric.py (run on the cluster)
src/unlearn/
  data.py, data_surf.py    # CIFAR-10 / CASIA-SURF splits and loaders
  models.py                # ResNet-18, MobileNetV2, EfficientNet-B0; checkpoints
  train.py                 # fast GPU training from scratch (oracle models)
  metrics.py               # single-model metrics (accuracy, MIAs, divergences)
  forget_quality.py        # official competition metric (epsilon, F, final score)
  evaluation.py            # run artifacts + aggregation over N runs
  methods/                 # one module per method (table below)
results/<version>_<date>_<time>/  # one folder per run: findings, tables, plots
Notebooks/
  01_starter_kit.ipynb     # original starter kit
  02_local_test.ipynb      # interactive single-method check
  03_results_comparison.ipynb  # browse evaluate.py outputs
references/                # papers, cloned code, notebooks, write-ups (SOURCES.md)
```

## Methods

All methods share the competition interface
`unlearning(net, retain, forget, validation, **hparams) -> net`.
Each module documents its algorithm and source, and keeps its published
hyperparameters in `DEFAULTS`. Overrides are checked by name, so a typo raises
an error.

| Name (`--method`) | Place | Idea | Notes |
|---|---|---|---|
| `fanchuan_1` | 1 | KL-to-uniform on forget, then 8 × (contrastive push of forget logits away from retain logits + retain CE) | Faithful to the submitted code |
| `kookmin_2` | 2 | Re-init the 30% of conv weights with the smallest NegGrad+ gradients; fine-tune with 0.1× grads on the other weights | Port of the team's notebook |
| `seif_3` | 3 | Gaussian noise on conv weights (σ=0.6), 4 epochs weighted-CE fine-tuning, small noise before the last epoch | Two deviations, both documented in the module: the original noise reaches only the stem `conv1` (kept by default), and its class weights were face-data specific (balanced weights by default) |
| `sebastian_4` | 4 | Prune/re-init 99% of weights by L1, fine-tune 3.2 epochs with CE + entropy-matching to the original | |
| `toshik_5` | 5 | Transpose every conv kernel, fine-tune 3 epochs | Only approach 1 of the team's 2-approach ensemble; approach 2 is unpublished (the organisers couldn't reproduce it either) |
| `amnesiacs_6` | 6 | Reset first conv + classifier, distil the original on validation data, fine-tune with CE + KD | |
| `sun_7` | 7 | Reset classifier, noise into random layers while training only those, then full fine-tune | |
| `forget_8` | 8 | Cycles of random layer re-init + MSE distillation from the original on noisy retain images | Re-implemented from the write-up (no code was published); optimizer details are assumptions |
| `finetune` | baseline | Retain-set fine-tuning (starter kit) | |
| `neggrad_plus` | baseline | α·CE(retain) − (1−α)·CE(forget) | DeepUnlearn's recommended baseline; tune α |
| `random_label` | baseline | Train on retain + forget with random wrong labels | |
| `salun` | baseline | Random-label training restricted to the 50% most forget-salient weights | |
| `scrub` | baseline | Teacher–student: move away from the teacher on forget, stay close on retain | |
| `l1_sparse` | baseline | Retain fine-tuning with decaying L1 penalty | |
| `ssd` | baseline | Dampen weights whose Fisher importance is much higher on forget than on all data (no training) | |
| `original` | reference | No-op: the original model | Sanity check: should get the worst F |

**Further state-of-the-art methods** (group `SOTA`, added after the first
run; config `methods: sota`):

| Name (`--method`) | Idea | Notes |
|---|---|---|
| `bad_teacher` | Distil from the original model on retain and from a random network on forget | Authors' random-forget setting (1 epoch, 30% of retain) |
| `boundary_shrink` | Relabel forget images with their nearest wrong class (one FGSM step), fine-tune | Designed for class forgetting |
| `boundary_expanding` | Extra "shadow" output trained on forget images, then removed | Class-forgetting method; optimiser settings assumed |
| `lfssd` | SSD with a label-free importance (output-norm gradient) | No training |
| `scissorhands` | Re-initialise the most forget-salient later-layer weights, repair with gradient projection | Authors' CIFAR-10 10% settings |
| `scar` | Pull forget features to the nearest other class (Mahalanobis) + distillation | Retain images replace the paper's ImageNet surrogate images |
| `npo` | Bounded "negative preference" loss on forget + retain CE | LLM method adapted to classification |
| `rum` | Split forget set by memorisation (proxy), unlearn low -> mid -> high with NegGrad+ | Single-checkpoint confidence proxy |
| `sfron` | Fisher-masked adaptive gradient ascent interleaved with retain descent | Authors' CIFAR-10 10% settings |
| `l2ul` | Instance-wise ascent with adversarial anchors and MAS weight anchoring | Attack budget reduced for 5,000 forget images |

The 9th- and 10th-place write-ups exist only as Kaggle posts that need a
logged-in browser to read. If someone on the team can open them
(competition discussion tab), add them as `methods/<Team>_9.py` and `_10.py`.
The organisers' paper analysed only places 1–8 (minus 5th).

SCRUB's published defaults were tuned for small forget sets; with 5,000
forget images its maximisation steps diverged in the smoke test (accuracy
collapsed to ~16%). The config therefore also runs SCRUB with the organisers'
tuned setting (1 epoch, 1 max-step) under the tag `tuned_1ep`; both appear
in the results.

Baseline defaults are reasonable starting points, not tuned values. The
organisers tuned NegGrad+, SCRUB, Random Label, SalUn and L1-sparse before
comparing, so tune these before drawing conclusions about baselines vs
competition methods. Use the validation set for tuning, not test.

Architecture notes: methods that touch a specific layer use
`get_first_conv` / `get_classifier`, so all of them run on every architecture.
Two behave differently outside ResNet:

- `seif_3`: by default perturbs nothing on MobileNet/EfficientNet. Their
  top-level children aren't named `conv`. Pass `--hp confuse_all_convs=true`.
- `sun_7`: treats all normalisation layers as "layers" there, because none
  have `bn` in their names.

## Evaluation protocol

The question being asked: is an unlearned model distributed like a model that
never saw the forget set? So each method is run **N times** (seeds 0..N−1) from
**the same original model**. We also train **N retrain-from-scratch models**
on the retain set only (the oracle). Two distributions are then compared:

- the N unlearned models,
- the N retrained models.

This is the competition's "Reuse-N-1" protocol. The competition used N = 512;
we default to N = 64 (`configs/experiments.yaml`). With N models, ε can be
resolved up to ⌈ln(N−1)⌉, so N = 64 resolves ε up to 5. Raise N if time
allows.

Two choices of original model (`--orig`):

- `own`: our model trained on all 50k training images, using exactly the same
  recipe as the oracle models. This is the clean comparison and the main
  setting.
- `challenge`: the organisers' `weights_resnet18_cifar10.pth`, as in the
  starter kit. Its training recipe isn't published, so differences between
  it and our oracle models can leak into the metric. Treat as secondary.

Measured on 2026-11-09 for the challenge checkpoints, before any
recipe calibration:

| Checkpoint | Retain | Forget | Test | MIA |
|---|---|---|---|---|
| challenge original | 99.5% | 99.3% | 88.3% | 0.579 |
| challenge retrained | 99.5% | 88.2% | 88.0% | 0.499 |

Our recipe for training originals and oracle models from scratch
(`train.py`, `RECIPES["cifar10"]`): 100 epochs, SGD with Nesterov momentum
0.9, batch 128, 5-epoch linear warm-up to lr 0.1, cosine decay to 1e-4,
weight decay 5e-4, random crop + flip. The calibration models (2026-11-09,
g4, about 5.5 min each while sharing a GPU) match the organisers'
checkpoints closely:

| Our model | Retain | Forget | Test |
|---|---|---|---|
| original (seed 0) | 98.7% | 98.6% | 88.6% |
| retrained (seeds 0, 1) | 99.0% | 88.4% / 89.2% | 87.7% / 87.6% |

Retain accuracy is about 0.5 points below the challenge models because the
augmentation keeps the model from fully memorising the training set; test
and forget accuracy agree within about half a point. If a change to the
recipe is needed, change `RECIPES` (or `recipe_override` in the config) and
bump `version` so the old models stay.

## Metrics

Every run saves the model's logits on the fixed-order forget, retain, test
and validation sets, plus time and peak GPU memory. A run whose weights blow
up to NaN/inf is kept, flagged `diverged`, scored as a uniform predictor,
and counted in the `n_diverged` column (it does not crash the evaluation). All metrics are computed
afterwards from these files by `scripts/evaluate.py`, so new metrics can be
added without rerunning anything.

**Competition metric (primary)**

- `forget_quality_F`: per forget example, the logit-scaled confidence under
  the N unlearned vs N retrained models. The best of many threshold attacks
  gives an (ε, δ=1e-5) estimate. The example scores 1 point if ε<0.5, 0.5 if
  ε<1, and so on, halving per 0.5 of ε. F is the mean over the forget set
  (higher is better; 1 = indistinguishable from retraining). Exact port of the
  organisers' `metric.py`, verified by `check_metric_port.py`.
- `competition_score` = F × (RA_unlearned / RA_retrain) × (TA_unlearned /
  TA_retrain). This is the leaderboard number.
- `eps_median`, `eps_frac_below_0.5`, and per-example ε in `results/eps/`.

**Utility** (mean and std over the N runs)

- `retain_acc` (RA), `test_acc` (TA), `val_acc`.
- `forget_acc` (FA): the goal is to match the oracle's FA (about 88%), not 0%.

**Gaps to the oracle** (lower is better)

- `gap_retain_acc`, `gap_forget_acc`, `gap_test_acc`, `gap_mia_*`.
- `avg_gap`: mean of the RA, FA, TA and MIA-efficacy gaps. This is the summary
  number used by SalUn, L1-sparse and similar papers.

**Privacy attacks**

- `mia_forget_vs_test`: starter-kit loss attack. Logistic regression separating
  forget from test losses (10-fold CV accuracy). Ideal = the oracle's value
  (about 0.5).
- `mia_efficacy`: attack trained on retain (member) vs test (non-member) using
  loss, entropy and confidence, then applied to the forget set. Reports the
  fraction judged non-member (SalUn / L1-sparse papers). Compare to the
  oracle's value.

**Match to the oracle's outputs** (unlearned run i vs retrained run i)

- `js_forget` / `js_test`: Jensen–Shannon divergence of the predicted class
  distributions.
- `kl_retrain_to_unlearned_*`: KL divergence, retrained to unlearned.
- `agreement_*`: how often both predict the same class.
- `activation_dist_*`: mean L2 distance between softmax outputs.

**Cost**

- `seconds` per run, `peak_mem_mb`.
- `rte_speedup_vs_retrain`: retrain time ÷ unlearning time. The competition
  capped methods at roughly 20% of retraining time.

The paper also used U-LiRA (Hayes et al. 2024), a per-example likelihood-ratio
attack that needs shadow models per example. It is not implemented. F already
performs a per-example two-distribution test of the same kind.

Note on the official code: the published `metric.py` contains
`assert thr_eps, ...` on a NumPy array, which raises "truth value of an array
is ambiguous" whenever more than one threshold survives, so it cannot run as
published. Our port uses `len(...)`; `check_metric_port.py` patches only that
line in memory and confirms identical scores on all test regimes.

## Running on the cluster (dijkstra, one node)

Code lives at `~/Projects/MachineUnlearning` on the cluster. Artifacts are
written under `artifacts/` there and stay on the cluster; only summaries are
pulled back.

1. **Copy code:** `bash cluster/sync.sh push` from this repo. Needs SSH access
   to `head0`; see `DIJKSTRA-CLUSTER.md`, section 2.
2. **One-time setup** on head0: `bash ~/Projects/MachineUnlearning/cluster/setup_env.sh`.
   This creates conda env `unlearn`, with PyTorch for CUDA 12.8 (needed for
   Blackwell GPUs). It also downloads CIFAR-10 and the checkpoints.
3. **Smoke test** (about 15 min): `cd ~/Projects/MachineUnlearning && sbatch -J mu-smoke -w g3 cluster/run_pool.sbatch smoke`.
   It runs every method 3 times with tiny 2-epoch oracle models, then the
   evaluation, and starts with the metric-port check. Look at
   `cluster/logs/mu-smoke-<job>.out` and `tasks/smoke/logs/status.tsv`.
4. **Main run:** `sbatch -J mu-c10 -w g3 cluster/run_pool.sbatch cifar10_resnet18`.
   That's 1 own original + 64 oracle models, then 16 methods × 64 seeds × 2
   originals = 2,048 unlearning runs.
5. **Next architectures:** `cluster/run_pool.sbatch cifar10_mobilenet_v2 cifar10_efficientnet_b0`.
6. **Results back:** `bash cluster/sync.sh pull-results`, then open
   `Notebooks/03_results_comparison.ipynb`.

**Where results live.** Every job ends by collecting its results into one
folder in the repo, `results/<version>_<job start date>_<time>/` (for example
`results/v1_2026-11-09_1931/`). It contains:

- `finding_summary.md`: the written findings, with links;
- `manifest.json`: job id, node, code hash, N, settings, artifact paths;
- per experiment: `reference_models.csv`, `task_status.tsv`, and per original
  (`own`, `challenge`) `summary.csv/.md/.json`, `per_run.csv`, `eps/`, `plots/`.

These folders are small and are committed. The large artifacts (every model
checkpoint with its model card, and every run's logits) stay on the cluster
under the same version, `artifacts/<dataset>/<arch>/<version>/`, and the
manifest points to them.

Each job takes one whole node (all 8 GPUs) and runs `SLOTS` tasks per GPU
(default 7, so 56 at once). The other node stays free for other users.
Tasks skip themselves when their output already exists, so after the 24-hour
limit you just resubmit the same command.

**Saved models and versions.** Every trained model is kept: the originals,
every oracle model, and every unlearned model. Each `.pt` checkpoint has a
`.json` model card next to it with how it was made (method, hyperparameters,
recipe, seed, original checkpoint, code hash of `src/unlearn`, software and
GPU) and how it scored (accuracies, MIA, time). Everything sits under
`artifacts/<dataset>/<arch>/<version>/`. The version comes from `version:` in
the config (now `v1`); bump it whenever the recipe or code changes in a way
that matters, and earlier models stay untouched next to the new ones.

Disk: each saved ResNet-18 is 45 MB. One CIFAR-10/ResNet-18 experiment with
models saved is about 95 GB: 2,048 unlearned + 65 reference models. Pass
`--no-save-model` (via the config/task) if space gets tight; logits are always
kept, and they are about 1 MB per run.

**CASIA-SURF (the competition's data)** is licence-gated. Request it as
described in `references/code/unlearning_evaluation/README.md`. Then set
`data_root` and `enabled: true` for `surf_resnet18` in the config. The loader
and training recipe mirror the organisers' code.

## Adding a method

1. Add `src/unlearn/methods/<Name>_<rank>.py` (or a lowercase baseline name)
   with `INFO`, `DEFAULTS` and
   `unlearning(net, retain, forget, validation, **hparams)`. Use
   `merge_hparams(DEFAULTS, hparams, name)` and the shared `DEVICE`. Don't run
   anything at import time.
2. Register it in `src/unlearn/methods/__init__.py` (`COMPETITION` or
   `BASELINES`).
3. Record where it came from in `references/SOURCES.md`. Keep the original
   notebook/code under `references/`. Separate faithful reproduction from
   deliberate adaptations in the module docstring.
4. To try other hyperparameters, add them under `hparams:` in the config.
   They run under their own tag.

Each run starts from a fresh load of the same original checkpoint, so methods
never contaminate each other. This is the same rule as `copy.deepcopy(model)`
in the notebooks.

## Running a single method interactively

`Notebooks/02_local_test.ipynb` still works for quick looks. It uses
`get_cifar10_loaders()` and `finetune.unlearning(...)`, and any method can be
used the same way:

```python
from unlearn.methods import METHODS
model_u = METHODS["fanchuan_1"](copy.deepcopy(model), retain_loader, forget_loader, val_loader)
```

Restart the kernel after editing files under `src/unlearn/`.
