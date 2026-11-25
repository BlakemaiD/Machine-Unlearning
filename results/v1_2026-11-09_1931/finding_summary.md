# Findings: CIFAR-10 / ResNet-18, run v1_2026-11-09_1931

Run: Slurm job 10709 on dijkstra node g4 (8 × RTX PRO 6000), started
2026-11-09 19:31, finished ~21:10. Version `v1`. Code hash of `src/unlearn`
recorded per model in each model card (`.json`).

This compares 16 unlearning settings with the NeurIPS 2023 Machine Unlearning
Challenge's own metric plus complementary metrics:
- competition places 1–8,
- 6 literature baselines plus a second, organiser-tuned SCRUB setting,
- the no-op reference.

Each method ran 64 times (seeds 0–63) from the same original model and was
compared against 64 models retrained from scratch without the forget set
(the competition's "Reuse-N-1" protocol, with N = 64 instead of 512).

## Where everything is

| What | Where |
|---|---|
| Main tables (all metrics, mean ± std over 64 runs) | [own/summary.csv](cifar10_resnet18/own/summary.csv), [challenge/summary.csv](cifar10_resnet18/challenge/summary.csv) (also `.md`, `.json`) |
| Every run's scalar metrics (incl. the 64 retrained models) | [own/per_run.csv](cifar10_resnet18/own/per_run.csv), [challenge/per_run.csv](cifar10_resnet18/challenge/per_run.csv) |
| Per-example ε over the 5,000 forget images | [own/eps/](cifar10_resnet18/own/eps/), [challenge/eps/](cifar10_resnet18/challenge/eps/) |
| Plots (scores, ε histograms, loss histograms, confidence distributions) | [own/plots/](cifar10_resnet18/own/plots/), [challenge/plots/](cifar10_resnet18/challenge/plots/) |
| F ceiling (retrain vs retrain) | [own/f_ceiling_n32.csv](cifar10_resnet18/own/f_ceiling_n32.csv) |
| The 65 reference models (original + 64 retrained): accuracies, time, checkpoint path | [reference_models.csv](cifar10_resnet18/reference_models.csv) |
| Status of all 2,241 tasks; full job log | [task_status.tsv](cifar10_resnet18/task_status.tsv), [job_log.txt](cifar10_resnet18/job_log.txt) |
| Run settings, artifact paths | [manifest.json](manifest.json) |
| **All model checkpoints + model cards, all run logits (96 GB)** | cluster: `head0:~/Projects/MachineUnlearning/artifacts/cifar10/resnet18/v1/` (`reference/`, `own/models/`, `challenge/models/`, `*/runs/`) |

Metric definitions: [document.md, "Metrics"](../../document.md).

## Setup

- **Data:** CIFAR-10, the challenge's forget set of 5,000 training images,
  retain = the other 45,000, test/val = the challenge's 50/50 split of the
  test set.
- **Original models.** `own` is our ResNet-18, trained with the same recipe
  as the oracle models: 100 epochs, SGD with Nesterov momentum, lr 0.1 → 1e-4
  cosine, crop + flip. `challenge` is the organisers' checkpoint. Results are
  reported for both.
- **Oracle:** 64 ResNet-18s retrained on the retain set only.
  - retain 98.97% ± 0.05
  - forget 88.73% ± 0.28
  - test 87.92% ± 0.32
  - These match the organisers' retrained checkpoint (forget 88.2%, test 88.0%).
- Every method used its published hyperparameters; nothing was tuned on our
  data (except the extra SCRUB setting, taken from the organisers' paper).
  Competition methods were tuned by their authors for the face dataset.
- **Completion:** all 65 reference models and 2,176 unlearning runs finished,
  0 failed tasks. 9 of 64 default-SCRUB runs (own) diverged to NaN; they are
  kept and counted.

## Main results (original = own)

Score = competition score = F × (RA/RA_retrain) × (TA/TA_retrain). F =
forgetting quality (higher = closer to retraining). RA/FA/TA =
retain/forget/test accuracy. Target values are the oracle row.

| Method | Score | F | RA | FA | TA | MIA | MIA-Eff | Time (s) |
|---|---|---|---|---|---|---|---|---|
| neggrad_plus (baseline) | **0.073** | 0.072 | 1.000 | 0.905 | 0.877 | 0.518 | 0.262 | 77 |
| sebastian_4 (4th) | 0.059 | 0.063 | 0.966 | 0.879 | 0.839 | 0.523 | 0.236 | 35 |
| forget_8 (8th) | 0.048 | 0.061 | 0.866 | 0.798 | 0.782 | 0.511 | 0.274 | 42 |
| kookmin_2 (2nd) | 0.028 | 0.031 | 0.956 | 0.872 | 0.823 | 0.528 | 0.185 | 47 |
| amnesiacs_6 (6th) | 0.020 | 0.023 | 0.922 | 0.893 | 0.824 | 0.538 | 0.161 | 38 |
| l1_sparse (baseline) | 0.018 | 0.019 | 0.984 | 0.841 | 0.834 | 0.501 | 0.273 | 112 |
| finetune (baseline) | 0.016 | 0.020 | 0.917 | 0.800 | 0.796 | 0.501 | 0.273 | 44 |
| sun_7 (7th) | 0.013 | 0.013 | 0.982 | 0.947 | 0.867 | 0.548 | 0.123 | 76 |
| ssd (baseline) | 0.009 | 0.009 | 0.985 | 0.985 | 0.885 | 0.571 | 0.094 | 12 |
| seif_3 (3rd) | 0.008 | 0.012 | 0.847 | 0.768 | 0.731 | 0.521 | 0.256 | 36 |
| fanchuan_1 (1st) | 0.004 | 0.004 | 1.000 | 0.951 | 0.886 | 0.520 | 0.295 | 33 |
| toshik_5 (5th) | 0.003 | 0.004 | 0.864 | 0.785 | 0.780 | 0.501 | 0.263 | 27 |
| random_label (baseline) | 0.003 | 0.003 | 1.000 | 0.830 | 0.875 | 0.582 | 0.688 | 17 |
| salun (baseline) | 0.003 | 0.003 | 1.000 | 0.830 | 0.875 | 0.582 | 0.691 | 19 |
| scrub [tuned_1ep] | 0.001 | 0.004 | 0.378 | 0.370 | 0.382 | 0.502 | 0.460 | 9 |
| scrub (default) | 0.000 | 0.005 | 0.154 | 0.154 | 0.157 | 0.499 | 0.514 | 22 |
| original (no unlearning) | 0.000 | 0.000 | 0.987 | 0.986 | 0.886 | 0.572 | 0.093 | 0 |
| **retrain (oracle)** | – | – | 0.990 | 0.887 | 0.879 | 0.505 | 0.232 | 631 |

The full table, with stds, gaps to the oracle, JS divergence, agreement and
activation distance, is in [own/summary.csv](cifar10_resnet18/own/summary.csv).
With the challenge's original checkpoint the picture is the same (the top 4
are NegGrad+, Sebastian, Kookmin, Forget). See
[challenge/summary.md](cifar10_resnet18/challenge/summary.md).

## How to read F: the ceiling

F never gets close to 1 at this number of models. Scoring 32 retrained models
against the other 32 retrained models (they are indistinguishable by
construction) gives only **F = 0.186**: with few models the threshold attacks
always find some chance separation, and no forget example reaches ε < 0.5.
The same 32-vs-32 comparison for the methods
([f_ceiling_n32.csv](cifar10_resnet18/own/f_ceiling_n32.csv)):

| | F (32 vs 32) | share of ceiling |
|---|---|---|
| retrain vs retrain | 0.186 | 100% |
| neggrad_plus | 0.089 | 48% |
| sebastian_4 | 0.079 | 43% |
| forget_8 | 0.078 | 42% |
| kookmin_2 | 0.047 | 25% |
| finetune | 0.034 | 19% |
| original | 0.000 | 0% |

So the best methods get about halfway to "indistinguishable from retraining".
Absolute F values are only comparable at the same N.

## Findings

1. **A plain, well-known baseline wins on CIFAR-10.** NegGrad+ has the best
   competition score with both originals (0.073 own, 0.076 challenge). It
   keeps retain accuracy at 100% and brings forget accuracy to 90.5%, close to
   the oracle's 88.7%. This agrees with DeepUnlearn (Cadet et al. 2024), which
   recommends NegGrad+ as the reference baseline.

2. **Competition rank does not transfer to CIFAR-10 with the submitted
   hyperparameters.** The best competition entries here are Sebastian (4th),
   Forget (8th) and Kookmin (2nd). The winner, Fanchuan, ranks 11th of 17 rows: its
   forget accuracy stays at 95%, so it barely moves toward the oracle. The
   organisers' paper reports the same for FEMNIST ("no good values were
   found within its grid" for Fanchuan) and names Sebastian the strongest out
   of the box. The competition methods were tuned for a different dataset (face
   images, age groups), so these results measure out-of-the-box transfer, not
   the methods' best possible performance.

3. **Re-initialisation methods trade utility for forgetting, and some are
   unstable.** Forget (8th) and Amnesiacs (6th) re-initialise layers. Their
   retain accuracy varies a lot between seeds (std 6.2 and 8.5 points, vs ≤ 0.5
   for most others), so some runs lose much more utility than others.
   Seif (3rd) and toshi_k (5th) lose 10–15 points of test accuracy.

4. **Over-forgetting is visible in the privacy attack.** Random Label and
   SalUn give forget samples random wrong labels, so the attack flags 69% of
   them as "never seen" (MIA-efficacy), against 23% for the retrained models.
   That makes them easy to spot (the "Streisand effect"), and their F is near
   the bottom. SalUn's 50% saliency mask made almost no difference here
   (forget accuracy 82.96% vs 82.99% for Random Label).

5. **Some methods change too little.** SSD (12 s, no training) leaves forget
   accuracy at 98.5%, almost the original's 98.6%, with its published α = 10,
   λ = 1; it would need a lower α for a random forget set (untested).
   Sun (7th) keeps 94.7% forget accuracy.

6. **SCRUB collapses with a 5,000-image forget set.** With published defaults,
   accuracy falls to ~15% and 9/64 runs diverge. With the organisers' tuned
   setting (1 epoch, 1 max-step) it falls to ~38%. Its maximisation step needs
   tuning (lower learning rate / fewer steps) before it is a fair baseline.

7. **The two privacy attacks disagree with F.** The starter-kit loss attack
   (MIA) is near 0.50 for most methods, including weak ones like fine-tune.
   So it can't separate methods well, which agrees with the organisers'
   Appendix A.11. F and the gap metrics give a clearer ranking.

8. **Cost.** All methods run well under the competition's 20%-of-retraining
   budget: 12–112 s per run vs 631 s to retrain (with 56 runs sharing 8 GPUs),
   a 5–50× speed-up.

## Caveats

- N = 64 models per side (the competition used 512). Ranking is clear at the
  top, but small F differences (< ~0.01) between lower-ranked methods are
  within noise; use the per-run CSVs for confidence intervals.
- Nothing was tuned on CIFAR-10. Baselines like NegGrad+ have a tunable α that
  the published defaults may not suit; a fair comparison would tune each method
  on the validation set.
- Three competition ports are approximate: toshi_k (only half of its
  ensemble), Forget (re-implemented from the write-up), Seif (balanced class
  weights instead of the face-data weights). See the method module docstrings.
- Our `own` original reaches 98.7% retain accuracy, vs 99.5% for the
  challenge checkpoint (crop/flip augmentation), with matching test accuracy.

## Next

- 10 further state-of-the-art methods (Bad Teacher, Boundary Shrink /
  Expanding, LFSSD, Scissorhands, SCAR, NPO, RUM, SFR-on, L2UL) are
  implemented and will run on the same reference models (`v1`), so their
  results join these tables.
- Optional: hyperparameter tuning on the validation set; MobileNetV2 /
  EfficientNet-B0; CASIA-SURF (needs the dataset licence); N = 512.
