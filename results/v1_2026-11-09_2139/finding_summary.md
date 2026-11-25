# Findings: CIFAR-10 / ResNet-18, run v1_2026-11-09_2139 (all 26 methods)

Run: Slurm job 10712 on dijkstra node g4 (8 × RTX PRO 6000), started
2026-11-09 21:39, finished 23:03 (1,280 unlearning runs in 40 min, then
evaluation of all 27 settings). Version
`v1`: same 65 reference models (original + 64 retrained) as the first run, so
the numbers are directly comparable.

This run adds **10 further state-of-the-art methods** (2022–2024) to the 17
settings of the first run ([../v1_2026-11-09_1931/finding_summary.md](../v1_2026-11-09_1931/finding_summary.md)).
The evaluation re-scores all methods together, so the tables below are the
complete 27-row comparison (8 competition entries, 8 baselines incl. the
no-op, 10 SOTA methods, plus the tuned SCRUB setting).

Each method ran 64 times (seeds 0–63) from the same original model and is
compared against the 64 retrained models (competition protocol "Reuse-N-1",
N = 64).

## Where everything is

| What | Where |
|---|---|
| Main tables, all 27 settings (mean ± std over 64 runs) | [own/summary.csv](cifar10_resnet18_sota/own/summary.csv), [challenge/summary.csv](cifar10_resnet18_sota/challenge/summary.csv) (also `.md`, `.json`) |
| Every run's scalar metrics | [own/per_run.csv](cifar10_resnet18_sota/own/per_run.csv), [challenge/per_run.csv](cifar10_resnet18_sota/challenge/per_run.csv) |
| Per-example ε over the 5,000 forget images | [own/eps/](cifar10_resnet18_sota/own/eps/), [challenge/eps/](cifar10_resnet18_sota/challenge/eps/) |
| Plots | [own/plots/](cifar10_resnet18_sota/own/plots/), [challenge/plots/](cifar10_resnet18_sota/challenge/plots/) |
| F ceiling (32 vs 32) for the top methods | [own/f_ceiling_n32.csv](cifar10_resnet18_sota/own/f_ceiling_n32.csv), [challenge/f_ceiling_n32.csv](cifar10_resnet18_sota/challenge/f_ceiling_n32.csv) |
| Reference models; task status; job log | [reference_models.csv](cifar10_resnet18_sota/reference_models.csv), [task_status.tsv](cifar10_resnet18_sota/task_status.tsv), [job_log.txt](cifar10_resnet18_sota/job_log.txt) |
| Run settings | [manifest.json](manifest.json) |
| **All checkpoints + model cards, all run logits** | cluster: `head0:~/Projects/MachineUnlearning/artifacts/cifar10/resnet18/v1/` (`reference/`, `own/models/<method>/`, `challenge/models/<method>/`, `*/runs/`) |

Method details and sources: [document.md](../../document.md) and the
docstring of each module in `src/unlearn/methods/`. Papers and code are in
`references/` (see `references/SOURCES.md`).

## The 10 new methods

| Method | Paper | Idea |
|---|---|---|
| bad_teacher | Chundawat et al., AAAI 2023 | distil from the original on retain, from a random "incompetent" teacher on forget |
| boundary_shrink | Chen et al., CVPR 2023 | fine-tune forget images toward their nearest wrong class (FGSM neighbour) |
| boundary_expanding | Chen et al., CVPR 2023 | add a shadow class, push forget images into it, then drop it |
| lfssd | Foster et al., ICLR 2024 TP | SSD with label-free (MAS) importances; no training |
| scissorhands | Wu et al., ECCV 2024 | prune connections salient to the forget set, then repair with gradient projection |
| scar | Bonato et al., ECCV 2024 | push forget embeddings to the nearest wrong-class distribution (Mahalanobis), distil retain |
| npo | Zhang et al., COLM 2024 | negative preference optimisation (bounded gradient ascent) + retain CE |
| rum | Zhao et al., NeurIPS 2024 | split forget set by memorisation (confidence proxy), unlearn each group with NegGrad+ |
| sfron | Huang et al., ICLR 2025 | saliency-masked forget-retain alternation with weight-norm control |
| l2ul | Cha et al., AAAI 2024 | instance-wise gradient ascent + adversarial anchors + MAS regulariser |

All used their published hyperparameters. Where the paper's setting could not
apply to a 5,000-image random forget set, the adaptation is written in the
module docstring (L2UL: smaller attack budget; SCAR: retain images instead
of external OOD data; RUM: confidence instead of the costly memorisation
score).

## Main results (original = own)

Score = competition score = F × (RA/RA_retrain) × (TA/TA_retrain). F =
forgetting quality (higher = closer to retraining). RA/FA/TA =
retain/forget/test accuracy. AvgGap = mean |difference to the oracle| over
RA, FA, TA, MIA, MIA-Eff (lower is better). **New methods in bold.**

| # | Method | Score | F | RA | FA | TA | MIA-Eff | AvgGap | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | neggrad_plus (baseline) | 0.073 | 0.072 | 1.000 | 0.905 | 0.877 | 0.262 | 0.015 | 77 |
| 2 | sebastian_4 (4th) | 0.059 | 0.063 | 0.966 | 0.879 | 0.839 | 0.236 | 0.019 | 35 |
| 3 | forget_8 (8th) | 0.048 | 0.061 | 0.866 | 0.798 | 0.782 | 0.274 | 0.088 | 42 |
| 4 | **sfron** | 0.041 | 0.041 | 1.000 | 0.745 | 0.870 | 0.370 | 0.075 | 34 |
| 5 | **scissorhands** | 0.033 | 0.033 | 1.000 | 0.934 | 0.881 | 0.172 | 0.030 | 42 |
| 6 | kookmin_2 (2nd) | 0.028 | 0.031 | 0.956 | 0.872 | 0.823 | 0.185 | 0.038 | 47 |
| 7 | amnesiacs_6 (6th) | 0.020 | 0.023 | 0.922 | 0.893 | 0.824 | 0.161 | 0.050 | 38 |
| 8 | l1_sparse (baseline) | 0.018 | 0.019 | 0.984 | 0.841 | 0.834 | 0.273 | 0.035 | 112 |
| 9 | finetune (baseline) | 0.016 | 0.020 | 0.917 | 0.800 | 0.796 | 0.273 | 0.071 | 44 |
| 10 | sun_7 (7th) | 0.013 | 0.013 | 0.982 | 0.947 | 0.867 | 0.123 | 0.047 | 76 |
| 11 | **scar** | 0.010 | 0.009 | 0.985 | 0.966 | 0.891 | 0.238 | 0.025 | 190 |
| 12 | ssd (baseline) | 0.009 | 0.009 | 0.985 | 0.985 | 0.885 | 0.094 | 0.061 | 12 |
| 13 | seif_3 (3rd) | 0.008 | 0.012 | 0.847 | 0.768 | 0.731 | 0.256 | 0.109 | 36 |
| 14 | fanchuan_1 (1st) | 0.004 | 0.004 | 1.000 | 0.951 | 0.886 | 0.295 | 0.036 | 33 |
| 15 | **boundary_shrink** | 0.004 | 0.004 | 0.912 | 0.903 | 0.814 | 0.188 | 0.051 | 23 |
| 16 | **rum** | 0.004 | 0.004 | 0.991 | 0.574 | 0.802 | 0.472 | 0.158 | 392 |
| 17 | **boundary_expanding** | 0.003 | 0.004 | 0.927 | 0.930 | 0.830 | 0.156 | 0.058 | 7 |
| 18 | toshik_5 (5th) | 0.003 | 0.004 | 0.864 | 0.785 | 0.780 | 0.263 | 0.090 | 27 |
| 19 | random_label (baseline) | 0.003 | 0.003 | 1.000 | 0.830 | 0.875 | 0.688 | 0.132 | 17 |
| 20 | salun (baseline) | 0.003 | 0.003 | 1.000 | 0.830 | 0.875 | 0.691 | 0.133 | 19 |
| 21 | **bad_teacher** | 0.002 | 0.002 | 0.974 | 0.958 | 0.868 | 0.246 | 0.028 | 3 |
| 22 | **npo** | 0.001 | 0.006 | 0.468 | 0.128 | 0.419 | 0.763 | 0.568 | 120 |
| 23 | scrub [tuned_1ep] | 0.001 | 0.004 | 0.378 | 0.370 | 0.382 | 0.460 | 0.464 | 9 |
| 24 | **l2ul** | 0.000 | 0.006 | 0.183 | 0.167 | 0.156 | 0.543 | 0.641 | 13 |
| 25 | scrub (default) | 0.000 | 0.005 | 0.154 | 0.154 | 0.157 | 0.514 | 0.643 | 22 |
| 26 | **lfssd** | 0.000 | 0.000 | 0.987 | 0.986 | 0.886 | 0.093 | 0.062 | 16 |
| 27 | original (no unlearning) | 0.000 | 0.000 | 0.987 | 0.986 | 0.886 | 0.093 | 0.062 | 0 |
| – | **retrain (oracle)** | – | – | 0.990 | 0.887 | 0.879 | 0.232 | – | 631 |

With the organisers' checkpoint (`challenge`) the order is nearly the same.
The top 5 are NegGrad+, Sebastian, Kookmin, Forget and Scissorhands, with
SFR-on 6th (0.029). See [challenge/summary.md](cifar10_resnet18_sota/challenge/summary.md).

## F as a share of the ceiling (32 vs 32 models)

Two halves of the retrained models scored against each other give F = 0.186,
the best possible at this N (see the first run's findings).

| | F own | share | F challenge | share |
|---|---|---|---|---|
| retrain vs retrain | 0.186 | 100% | 0.186 | 100% |
| neggrad_plus | 0.089 | 48% | 0.093 | 50% |
| sebastian_4 | 0.079 | 43% | 0.085 | 46% |
| **sfron** | 0.061 | 33% | 0.046 | 25% |
| **scissorhands** | 0.049 | 27% | 0.054 | 29% |
| **scar** | 0.021 | 11% | 0.022 | 12% |
| **rum** | 0.010 | 5% | 0.012 | 7% |
| **npo** | 0.009 | 5% | 0.009 | 5% |

The best new methods get a quarter to a third of the way to
"indistinguishable from retraining", against about half for NegGrad+.

## Findings

1. **No new method beats NegGrad+ or Sebastian.** The best SOTA methods,
   SFR-on (4th overall) and Scissorhands (5th), land between the top
   competition entries and the rest. NegGrad+ remains the best overall with
   both originals.

2. **Scissorhands is the best "keep utility" method.** It keeps retain
   accuracy at 100% and test accuracy at 88.1% (the oracle's 87.9%), and is
   very stable across seeds (FA std 0.3 points). Its AvgGap (0.030) is the
   fifth lowest of all methods. It under-forgets: forget accuracy stays at 93.4% vs 88.7%.

3. **SFR-on forgets more strongly but varies across seeds.** Forget accuracy
   averages 74.5%, below the oracle's 88.7%, with a std of 9.3 points between
   seeds. That is the largest spread of any method that doesn't collapse. Its
   MIA-efficacy (0.37 vs 0.23) shows mild over-forgetting. Retain and test
   accuracy stay intact. A lower forget step or fewer iterations would
   probably move it closer to the oracle.

4. **Output-matching is not the same as passing the attack.** SCAR and Bad
   Teacher have small accuracy gaps (AvgGap 0.025 and 0.028, only NegGrad+ and
   Sebastian are lower). SCAR also has the lowest JS divergence to the
   oracle on the forget set of all methods (0.046). Yet their F is near the
   bottom (0.009, 0.002). Their forget accuracy stays high (96.6%, 95.8%), so
   the per-example confidence on forget images still looks "trained on" to
   the threshold attack. Average-accuracy metrics alone would have ranked them
   among the best.

5. **RUM over-forgets with its published NegGrad+ settings.** It runs
   NegGrad+ separately on three confidence groups. Applying that to a random
   5,000-image set drives forget accuracy to 57% (oracle 88.7%), with an
   attack-visible gap (MIA-Eff 0.47). It is also the most expensive method
   (392 s per run, 62% of retraining).

6. **NPO and L2UL collapse the model, as in the smoke test.** NPO (designed
   for LLMs) drops retain accuracy to 47%. L2UL (designed for forgetting up
   to 128 instances) drops it to 18%. Both use gradient ascent on the forget
   set, and 5,000 forget images is far beyond what their published step sizes
   and regularisers were tuned for. They behave like SCRUB's collapse in the
   first run.

7. **LFSSD does nothing on a random forget set.** Its outputs are identical
   to the original (std 0). The selection rule dampens a weight only if it is
   10× more important for the forget set than for the full data. A uniform
   10% sample has about the same importances as the full data, so no weight
   qualifies. SSD (squared-gradient importance) changes only slightly more.
   Both are designed for class or sub-class forgetting, where importances
   differ strongly.

8. **Boundary Shrink/Expanding lose utility without forgetting well.** Test
   accuracy drops 5–7 points (to 81–83%), while forget accuracy stays at
   90–93% and F is near the bottom. They were designed for full-class
   forgetting, where pushing samples to a neighbouring class is the goal.

9. **Cost and the competition budget.** The competition allowed 20% of
   retraining time (~126 s here). Two new methods exceed it: SCAR (190 s, it
   needs per-class covariances and up to 30 epochs at batch size 1,024) and
   RUM (392 s). Bad Teacher (3 s) and Boundary Expanding (7 s) are the
   cheapest of all trained methods.

10. **Summary across all 26 methods.** On a random forget set, methods that
    combine a moderate push on the forget set with continued training on
    the retain set (NegGrad+, Sebastian, Forget, SFR-on, Scissorhands) do
    best. Methods designed for class forgetting (SSD, LFSSD, Boundary
    Shrink/Expanding) or for few instances or LLMs (L2UL, NPO) transfer
    poorly with their published settings.

## Caveats

- Same as the first run: N = 64 per side, and nothing tuned on CIFAR-10.
  The collapsing methods (NPO, L2UL, SCRUB) and the over-forgetting ones
  (RUM, SFR-on) would very likely improve with validation-set tuning of the
  learning rate and number of steps. These results measure out-of-the-box
  transfer.
- Approximate ports, documented in the module docstrings: L2UL (2
  adversarial images × 10 PGD steps instead of 20 × 100), SCAR (retain images
  as the surrogate data), RUM (confidence proxy for memorisation),
  Boundary Expanding (optimiser settings not stated in the paper).
- Completion: 1,280/1,280 runs finished, 0 failed, 0 diverged (NaN) runs
  among the new methods.

## Next (optional, not started)

- Hyperparameter tuning on the validation set for all methods, starting with
  the collapsing and over-forgetting ones.
- MobileNetV2 / EfficientNet-B0 (configs ready); CASIA-SURF (needs the
  dataset licence); N = 512.
