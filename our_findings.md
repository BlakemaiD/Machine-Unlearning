# Our method (LASH): what we tried and what we found

Status as of 2026-11-20. Local notes, not committed. Setting throughout:
CIFAR-10, ResNet-18, our own original model (`--orig own`), forget set
5,000 images, competition metric. All LASH runs ran on aisec102.

Main references:

- Method code: `src/unlearn/methods/lash.py`
- Construction and proofs: `docs/lash.md`
- Sweep inputs, logs and score CSVs: `tasks/lash_sweep*/`

## 1. Goal and constraints

These are the user's requirements for the method.

- **No retraining.** Edit the weights directly while the inputs stay fixed.
  This is an adversarial attack (PGD, C&W) aimed at the weights instead of
  the inputs, used to forget.
- **Use clean data.** A small clean set (200 images) gives per-layer,
  per-class angle and distance statistics. 2,000 retain images help protect
  accuracy.
- **Geometric target.** Push each forget sample toward where unseen samples
  of its own class sit, and never toward other classes.
- **Stay conservative.** Change the weights as little as possible, measured
  against a centre point (an L2 ball around the original weights).
- **Borrow attack ideas.** From backdoor attacks, use rarely used versus
  commonly used neurons. Push hard first, then reconstruct C&W-style so that
  forgetting stays high and accuracy recovers.
- **Be mathematically and geometrically provable, and simple.**
- **Standalone, no retain optimisation (decided 2026-11-19).** Even a short
  optimisation on retain data counts as retraining. Removal requests may come
  weekly for large multi-GPU models, so the edit must be cheap. A short
  repair-training step may be offered only as an optional add-on in an
  appendix, and only if it gives clearly stronger results.
- **Existing papers' methods are references only.** They must not be
  modified and presented as ours. An NGSP variant (NegGrad+ with
  shrink-perturb) was built and dropped for this reason.

## 2. Reference numbers

Competition score = F · (RA/RA_r) · (TA/TA_r).

- F is forgetting quality, between 0 and 1.
- RA and TA are retain and test accuracy; RA_r and TA_r are those of the
  retrained models.
- F compares, for every forget image, the confidence distribution over N
  unlearned models with that over N retrained models.
- F depends on N. Tuning runs use N = 16. Each score is F averaged over 4
  disjoint groups of 16 retrained models (`scripts/quick_score.py`).
- At N = 64, the full benchmark gave NegGrad+ a score of 0.073, so numbers
  below are comparable only with each other.

| Model                           | FA    | RA     | TA    | score (N=16)    |
| ------------------------------- | ----- | ------ | ----- | --------------- |
| Retrained oracle                | 88.7% | 99.0%  | 87.9% | –              |
| Original (before unlearning)    | 98.6% | 98.7%  | 88.6% | –              |
| NegGrad+ (best existing method) | 90.4% | 99.99% | 87.6% | **0.115** |
| Sebastian (2nd)                 | 88.0% | 96.6%  | 83.9% | 0.098           |

FA = forget accuracy (target ≈ oracle 88.7%).

Privacy metrics, also reported by `scripts/quick_score.py` since 2026-11-19:

- **MIA**: forget-vs-test loss-threshold attack (starter kit). Retrained ≈ 0.505.
- **MIA_eff**: logistic-regression attack on loss, entropy and confidence,
  trained on retain vs test and applied to forget. It is the fraction judged
  "non-member". Retrained ≈ 0.233.
- **ULiRA_AUC**: offline per-model U-LiRA. Each forget or test image gets a
  Gaussian N(μ_i, σ_i) of φ from retrained shadow models 16–63. The attack
  score is z², and the result is the forget-vs-test AUC averaged over models.
  0.5 means no leak; the retrained reference is 0.499.

`--fast` uses `src/unlearn/forget_quality_fast.py`, a NumPy port of the
official ε computation. It gives identical values (0.114062 vs 0.114062 for
NegGrad+) and is much faster.

## 3. The method: LASH (Layer-wise Angular SHell editing)

All parts are options of one function, so "methods we tried" below are
settings of these parts. Batch norm is frozen, so each conv/linear layer is
linear in its weights. That makes the per-layer statements below exact.

### 3.1 Geometric target: class shells

The 200 clean images (validation split, never trained on) define three
things for each class at each ResNet stage and at the logits:

- a class direction μ_c;
- the angle of a sample to it, cos(h, μ_y);
- its radius ‖h‖.

At the logits we use the competition's confidence φ = log p_y − log(1−p_y).
All are standardised to z-scores.

- **Shell matching.** The forget samples' z-scores are matched to the clean
  ones with a 1-D Wasserstein-2 loss (sorted values vs clean quantiles). In
  1-D this is the exact optimal-transport distance. Memorised samples, which
  are too confident and too close to μ_y, move out to where unseen samples
  sit.
- **Separation hinge.** No forget sample may get closer to another class's
  direction than clean samples do.
- **Keep term (optional).** KL on logits plus feature MSE on the keep images,
  so their outputs stay put.

### 3.2 Phase 1: push (weight-space PGD or C&W)

- **PGD.** Normalised gradient step per weight tensor with a cosine-decayed
  step. After each step, project onto the L2 ball ‖ΔW‖ ≤ ε‖W₀‖ per tensor.
- **C&W.** Adam on loss + c·‖ΔW‖²/‖W₀‖².
- Proposition 1: every iterate stays inside the allowed edit space and the
  ball.

### 3.3 Where the edit is allowed (the geometric restrictions)

| Option                    | What it edits                                                                                                                  | Guarantee (docs/lash.md)                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `subspace=lowkeep`      | Input directions the keep data barely uses (smallest eigenvalues of Σ_K, ≤ ρ of its energy). "Rare neurons" in input space. | Prop. 2: E_K‖Δy‖² ≤ ρ·tr(Σ_K)·ε²‖W₀‖²              |
| `subspace=geig`         | Top generalised eigenvectors of Σ_F v = γ Σ_K v: directions with the most forget energy per unit of keep energy             | Prop. 3: optimal (Ky Fan); E_K‖Δy‖² ≤ (1/γ_r)·E_F‖Δy‖² |
| `geig_min_gamma=γ_min` | Only directions with γ ≥ γ_min; layers with none are frozen                                                                 | the 1/γ_min bound holds in every layer at once                   |
| `channels=rare`         | Output channels active on forget but not keep data (backdoor-style "rare neurons")                                             | other channels exactly unchanged                                  |
| `channels=common`       | Channels most active on keep data (backdoor-style "common neurons")                                                            | other channels exactly unchanged                                  |

### 3.4 Phase 2: closed-form repair (the C&W-style reconstruction)

After a hard push, each layer is re-solved front to back:

    min_W  E_K‖Wx − W₀x₀‖² + μ·E_F‖Wx − W_p x_p‖² + λ‖W − W_p‖²

The three terms do the following:

- restore the keep outputs to the original's;
- hold the forget outputs where the push put them;
- keep the correction minimal.

Proposition 5 gives the unique closed-form minimiser
W* = (W₀C_K + μW_pC_F + λW_p)(S_K + μS_F + λI)⁻¹, with J(W*) ≤ J(W_p).

### 3.5 Randomised edit (added after the diagnosis in §5)

This adds a random edit ξ = r‖W₀‖·R(Z)/‖R(Z)‖ (Z Gaussian) in the same edit
space.

- It is applied before the push (`noise_at=start`, a PGD random start) or
  after it (`end`), optionally followed by the repair.
- Proposition 4: the noise obeys the same guarantees. In the geig subspace it
  changes keep outputs at most 1/γ times as much as forget outputs, frozen
  layers stay unchanged, and ‖ΔW‖ ≤ (ε + r)‖W₀‖.

### 3.6 Per-image random targets (copula; added after the expert analysis in §5)

`phi_target=copula`. Each run draws its own target confidence for every
forget image, and the push matches it (squared error on φ) instead of the
pooled W2 loss.

- Rank the forget images by the original model's φ, giving u⁰_i, and
  g_i = Φ⁻¹(u⁰_i).
- Draw u_i = Φ(ρ g_i + √(1−ρ²) ξ_i) with ξ_i ~ N(0,1) fresh per run.
- The target is the clean-φ quantile at u_i. With `copula_class=True` this
  uses the image's own class's clean quantiles.
- Proposition 6 (docs/lash.md):
  - the targets have exactly the clean marginal;
  - two runs' targets correlate with ρ² on the Gaussian scale, so ρ sets the
    model-to-model spread;
  - the geig 1/γ bound on keep outputs still holds.
- The clean reference was raised from 200 to all 5,000 validation images
  (`n_clean=5000`, drawn at random each batch with `clean_random=True`).

## 4. Methods tried, in order

**0. Retain fine-tuning (rejected).** The first version fine-tuned on retain
data with SGD. The user ruled it out as retraining. In BN-eval mode it also
diverged at lr 0.01 (10% accuracy). It was replaced by the weight-space
attack.

**Sweep 1** (`tasks/lash_sweep`): 8 variants × 16 seeds, ε = 0.02 unless
stated. Keep set = 200 clean + 2,000 retain.

| tag              | setting                            | FA   | RA   | TA   | F     | score |
| ---------------- | ---------------------------------- | ---- | ---- | ---- | ----- | ----- |
| hard_repair      | no keep term, ε 0.05, then repair | 96.1 | 97.1 | 86.9 | 0.048 | 0.046 |
| p_rare           | rare channels (10%)                | 96.3 | 98.0 | 87.7 | 0.038 | 0.038 |
| hard_geig_repair | geig, no keep, ε 0.05, repair     | 96.4 | 97.9 | 87.8 | 0.038 | 0.037 |
| p_keep2k         | full space, keep term on 2,200     | 93.6 | 97.6 | 87.3 | 0.028 | 0.028 |
| p_clean200       | only the 200 clean images          | 92.5 | 94.9 | 85.3 | 0.028 | 0.026 |
| p_common         | common channels (10%)              | 96.3 | 98.0 | 87.6 | 0.020 | 0.020 |
| p_geig           | geig, top 10% directions           | 97.6 | 98.6 | 88.6 | 0.019 | 0.019 |
| p_lowkeep        | lowkeep, ρ 0.01                   | 98.1 | 98.7 | 88.6 | 0.005 | 0.005 |

**Geometry diagnostic** (`scripts/lash_geometry.py`, `tasks/lash_sweep/geometry.csv`;
keep set with 2,000 retain images). The largest γ per layer (forget energy
per unit of keep energy) is:

- conv1 and layer1: 1.08–1.33. Forget and retain look the same, so there is
  no selective room.
- layer2: 1.5–1.8.
- layer3: up to 6.0. layer4.0.conv1: 11.3.

In every layer, the lowkeep subspace holds only about 1% of the forget
energy, the same share as of the keep energy. Rarely used directions are
equally rare for forget data, which is why lowkeep barely forgets.

Single-seed probes then showed that Σ_K estimated from 2,000 images is
rank-deficient in deep layers (d up to 4,608). Part of the "forget-specific"
room came only from missing keep samples. Computing Σ_K on the full retain
set fixed this. These are sweep 2's settings.

**Sweep 2** (`tasks/lash_sweep2`): geig with a γ threshold, Σ_K from all
retain images, Σ_F from 5,000 forget images; 6 variants × 16 seeds.

| tag                  | setting                            | FA   | RA   | TA   | F     | score |
| -------------------- | ---------------------------------- | ---- | ---- | ---- | ----- | ----- |
| geo_g1.2_e0.5_repair | γ ≥ 1.2, ε 0.5, no keep, repair | 95.7 | 98.0 | 87.7 | 0.043 | 0.043 |
| geo_g1.2_e0.3        | γ ≥ 1.2, ε 0.3, no keep         | 92.2 | 96.5 | 86.2 | 0.042 | 0.040 |
| geo_g1.2_e0.5_keep3  | γ ≥ 1.2, ε 0.5, keep ×3        | 92.4 | 98.0 | 87.7 | 0.038 | 0.037 |
| geo_g1.5_e0.5        | γ ≥ 1.5, ε 0.5, no keep         | 92.4 | 96.9 | 87.0 | 0.038 | 0.037 |
| geo_g1.2_e0.3_keep   | γ ≥ 1.2, ε 0.3, keep ×1        | 92.7 | 98.1 | 87.6 | 0.036 | 0.036 |
| geo_g1.5_e0.2        | γ ≥ 1.5, ε 0.2, no keep         | 96.6 | 98.6 | 88.3 | 0.014 | 0.015 |

geo_g1.2_e0.3_keep has the best accuracy balance of any LASH setting.

- It is close to the oracle on FA (92.7 vs 88.7) and loses only 0.6 points
  of RA.
- Its score is still low, for the reason in §5.

**Sweeps 3 and 4: randomised edit** (`tasks/lash_sweep3`, `tasks/lash_sweep4`).

- The base is geo_g1.2_e0.3_keep, plus noise r.
- The table shows accuracy and model-to-model spread (median over forget
  images of the std of φ across models). Retrained models have a spread of
  1.23–1.36.
- **Not yet scored with F.** Sweep 4 lost 31 of 80 runs to GPU
  out-of-memory: 6 runs per GPU was too many on the 24 GB card with the
  repair. Scoring never ran.

| tag              | setting                 | models | spread | mean φ | FA   | RA   | TA   |
| ---------------- | ----------------------- | ------ | ------ | ------- | ---- | ---- | ---- |
| retrained        | –                      | 16     | 1.36   | 5.19    | 88.7 | 99.0 | 87.9 |
| (no noise)       | geo_g1.2_e0.3_keep      | 16     | 0.29   | 5.28    | 92.7 | 98.1 | 87.6 |
| nz_e0.1          | noise 0.1, end          | 8      | 0.30   | 5.27    | 92.7 | 98.1 | 87.5 |
| nz_e0.3          | noise 0.3, end          | 8      | 0.45   | 5.20    | 92.7 | 97.9 | 87.4 |
| nz_s0.3          | noise 0.3, start        | 8      | 0.35   | 5.28    | 92.7 | 98.1 | 87.6 |
| nz_e0.6          | noise 0.6, end          | 15     | 0.82   | 4.90    | 91.9 | 97.1 | 86.7 |
| nz_e1.0          | noise 1.0, end          | 14     | 1.30   | 4.00    | 89.3 | 94.4 | 84.5 |
| nz_e1.0_rep      | noise 1.0, repair μ 1  | 16     | 0.79   | 4.97    | 94.5 | 97.6 | 87.4 |
| nz_e1.0_rep_mu5  | noise 1.0, repair μ 5  | 11     | 1.13   | 4.30    | 91.0 | 95.8 | 85.7 |
| nz_e1.5_rep_mu20 | noise 1.5, repair μ 20 | 9      | 1.68   | 1.80    | 77.4 | 84.0 | 76.4 |
| nz_e1.5_rep_mu5  | noise 1.5, repair μ 5  | 0      | –     | –      | –   | –   | –   |

**Sweeps 5 and 6: per-image random targets** (`tasks/lash_sweep5`,
`tasks/lash_sweep6`). All of these use geig with γ ≥ 1.2, full-retain Σ_K,
and 16 seeds. Scores are at N = 16 with the fast metric. c2k/c5k means
2,000/5,000 clean images, pc means per-class targets, and rep means ε 0.5
with no keep term, followed by the repair.

| tag               | ρ   | FA   | RA   | TA   | F     | score      | MIA   | MIA_eff | ULiRA_AUC |
| ----------------- | --- | ---- | ---- | ---- | ----- | ---------- | ----- | ------- | --------- |
| retrained         | –   | 88.7 | 99.0 | 87.9 | –     | –          | 0.505 | 0.233   | 0.499     |
| c5k_r0.7_pc_rep   | 0.7 | 93.7 | 98.4 | 88.2 | 0.102 | **0.1017** | 0.535 | 0.168   | 0.592     |
| cop_r0.7_c2k_rep  | 0.7 | 94.3 | 98.4 | 88.1 | 0.101 | 0.1011     | 0.539 | 0.159   | 0.596     |
| cop_r0.7          | 0.7 | 93.9 | 98.4 | 88.1 | 0.100 | 0.0998     | 0.532 | 0.187   | 0.597     |
| cop_r0.7_c2k      | 0.7 | 91.4 | 98.3 | 88.0 | 0.099 | 0.0988     | 0.523 | 0.187   | 0.597     |
| cop_r0.5          | 0.5 | 94.5 | 98.3 | 88.1 | 0.099 | 0.0987     | 0.539 | 0.172   | 0.607     |
| c5k_r0.7          | 0.7 | 91.4 | 98.3 | 88.0 | 0.099 | 0.0986     | 0.523 | 0.188   | 0.597     |
| c5k_r0.6_pc       | 0.6 | 90.8 | 98.3 | 88.1 | 0.098 | 0.0973     | 0.521 | 0.195   | 0.602     |
| c5k_r0.7_pc       | 0.7 | 90.4 | 98.3 | 88.1 | 0.097 | 0.0968     | 0.517 | 0.201   | 0.596     |
| c5k_r0.8_pc       | 0.8 | 89.9 | 98.4 | 88.2 | 0.095 | 0.0950     | 0.514 | 0.206   | 0.589     |
| c5k_r0.8_pc_rep   | 0.8 | 93.3 | 98.4 | 88.2 | 0.092 | 0.0921     | 0.532 | 0.170   | 0.584     |
| cop_r0.5_c2k_e0.6 | 0.5 | 90.4 | 98.0 | 87.7 | 0.088 | 0.0874     | 0.518 | 0.203   | 0.641     |
| cop_r0.9          | 0.9 | 93.3 | 98.4 | 88.2 | 0.084 | 0.0834     | 0.527 | 0.196   | 0.577     |

The score went from 0.046 (best before) to 0.102. This is above Sebastian
(0.098) and 12% below NegGrad+ (0.115), with no retraining. RA and TA lose
only about 0.6 points.

**Round 7–8: better rank signal + cheaper statistics** (`tasks/lash_r7`,
`tasks/lash_r8`; 2026-11-20).

- Diagnosis: LASH's per-image spread already followed difficulty like
  retraining's (hard images 2.5 vs 2.2, easy 0.7 vs 0.9). The U-LiRA leak
  came from per-image *mean* error. The mean of LASH models correlated 0.765
  with the retrained mean, no better than the original confidence (0.768).
- Probe of cheap signals from the original model only (forget images; R² is
  2-fold cross-validated against the retrained per-image mean):

  | signal | corr | R² |
  | --- | --- | --- |
  | original φ on the image | 0.768 | 0.59 |
  | weight noise 0.1, mean of 8 | 0.781 | 0.61 |
  | input-gradient norm | −0.775 | 0.60 |
  | mean φ over ±4 px shifts | 0.835 | 0.69 |
  | **mean φ over ±2 px shifts × flip (16 views)** | **0.859** | **0.74** |
  | ±8 px shifts × flip | 0.658 | 0.43 |

  Added as `rank_signal="views"` (docs/lash.md, "Rank signal"). Proposition 7
  says the optimal ρ is the signal's Gaussian-rank correlation with one
  retrained model: 0.686 for the original φ (search found 0.7) and 0.768 for
  the views.
- Cost: `moment_dtype="float32"` (batch sums in float32, accumulated in
  float64). On an idle Quadro RTX 8000 one run takes **63 s instead of
  260 s**, with the same model (accuracies within 0.1 pt, predictions agree
  on 99.7%). Phases: edit space 19 s, attack 10 s, repair 33 s. NegGrad+
  takes 118 s on the same GPU. Retraining is estimated at about 7 min there,
  so LASH is about 15% of retraining. Caching Σ_K across requests is no
  longer worth it.

  All rows: views signal, per-class targets, ε 0.5, no keep term, repair
  (except v_r0.77: ε 0.3, keep ×1). N = 16, fast metric.

  | tag | ρ | FA | RA | TA | F | score | MIA_eff | ULiRA_AUC |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | v_r0.62_rep | 0.62 | 94.5 | 98.3 | 88.1 | 0.114 | **0.1137** | 0.159 | 0.573 |
  | v_r0.55_rep | 0.55 | 94.8 | 98.3 | 88.1 | 0.114 | 0.1136 | 0.157 | 0.580 |
  | v_r0.67_rep | 0.67 | 94.3 | 98.3 | 88.2 | 0.113 | 0.1129 | 0.162 | 0.567 |
  | v_r0.72_rep | 0.72 | 94.1 | 98.3 | 88.3 | 0.112 | 0.1114 | 0.164 | 0.560 |
  | v_r0.77_rep | 0.77 | 93.9 | 98.4 | 88.3 | 0.108 | 0.1075 | 0.165 | 0.551 |
  | v_r0.77 | 0.77 | 90.5 | 98.3 | 88.3 | 0.103 | 0.1027 | 0.200 | 0.557 |
  | v_r0.82_rep | 0.82 | 93.7 | 98.4 | 88.3 | 0.101 | 0.1008 | 0.168 | 0.540 |
  | v_r0.87_rep | 0.87 | 93.6 | 98.4 | 88.3 | 0.090 | 0.0898 | 0.169 | 0.528 |
  | (before) c5k_r0.7_pc_rep | 0.7 | 93.7 | 98.4 | 88.2 | 0.102 | 0.1017 | 0.168 | 0.592 |

  The score went from 0.102 to 0.113–0.114, level with NegGrad+ (0.115) at
  N = 16. At equal ρ the leak is lower than before (0.560 vs 0.592 at ρ ≈ 0.7).
  ρ = 0.67 is the chosen balance.

**Final 64-seed run of v_r0.67_rep** (`tasks/lash_final2/`; official metric at
N = 64, U-LiRA at N = 32):

  | method | score N=64 | F | RA | FA | TA | MIA_eff | ULiRA_AUC | score N=32 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | NegGrad+ | **0.0730** | 0.0724 | 99.99 | 90.5 | 87.7 | 0.262 | 0.516 | 0.091 |
  | **LASH v_r0.67_rep** | **0.0700** | 0.0702 | 98.3 | 94.3 | 88.2 | 0.163 | 0.565 | 0.088 |
  | LASH c5k_r0.7_pc_rep (before) | 0.0601 | 0.0603 | 98.4 | 93.8 | 88.2 | 0.170 | 0.591 | 0.077 |
  | Sebastian | 0.0588 | 0.0632 | 96.6 | 87.9 | 83.9 | 0.236 | 0.497 | 0.074 |

  - LASH is 2nd of 27 methods, at 96% of NegGrad+.
  - It has the best TA of all.
  - It runs at about half NegGrad+'s cost on the same GPU, with no
    retraining.

**NGSP (dropped).** A NegGrad+ variant with shrink-perturb noise
(`tasks/ngsp_sweep*`). It was stopped and removed because it modifies a paper
method; see §1. The code is kept outside the repo.

## 5. Findings

1. **No LASH setting has beaten NegGrad+ yet, but LASH now beats
   Sebastian.** The best score at N = 16 is 0.1017 (c5k_r0.7_pc_rep, §4
   sweep 6), against 0.115 for NegGrad+ and 0.098 for Sebastian. Before the
   per-image targets it was 0.046.
2. **The main reason is a lack of model-to-model variation, not the
   geometry.**
   - Every run edits the same original model in almost the same way, so the
     16 LASH models are near-copies.
   - Median per-image spread of φ: LASH 0.02–0.45, retrained 1.36, NegGrad+
     1.07, Sebastian 1.51.
   - When one distribution is nearly a point mass, the metric's threshold
     attack separates it from the retrained spread, and the image scores
     about nothing even when its mean is right.
   - The sweep 1 scores rank in the same order as this spread: lowkeep 0.02
     spread → score 0.005; geig 0.09 → 0.019; hard_repair 0.45 → 0.046.
3. **Early layers offer no selective room.** γ ≈ 1.1–1.3 in conv1 and
   layer1, so editing them hurts retain as much as forget. This holds for
   any method, because forget and retain are drawn iid from the same data.
   This is why the full-space push, rare and common channels and lowkeep all
   lowered FA and RA together, or barely moved either.
4. **Deep layers do have forget-specific directions** (γ up to 6 in layer3
   and 11 in layer4.0), but only once Σ_K is estimated on the whole retain
   set. With γ ≥ 1.2 and full-retain statistics, LASH reaches FA 92.7 /
   RA 98.1 / TA 87.6: forgetting close to the oracle with a 0.6-point RA
   cost. This is the geometric part working as the propositions predict.
5. **Rarely used directions (lowkeep) are rare for forget data too.** They
   hold about 1% of both forget and keep energy, so the edit is safe but
   barely forgets (FA 98.1).
6. **Backdoor-style channel selection** (rare or common neurons) keeps
   accuracy (RA 98.0) but under-forgets (FA 96.3). Rare channels were better
   than common ones (0.038 vs 0.020).
7. **Hard push + closed-form repair** was the best sweep 1 setting (0.046).
   The repair recovers RA after a hard push. The push without the keep term
   also adds a little model variation.
8. **Randomised edit: the variation can be matched, at a cost.**
   - Noise 1.0 in the γ ≥ 1.2 subspace reaches the retrained spread (1.30)
     but costs 4.6 points of RA, because the selective subspace is small and
     deep.
   - The repair restores RA but removes part of the spread: μ = 1 gives
     0.79, μ = 5 gives 1.13 at RA 95.8.
   - Too much noise (1.5 with μ 20) collapses accuracy.
   - The open question is which point on this trade-off scores best. That
     needs the F scoring that has not run yet.
9. More PGD steps (500 vs 200) gave no gain (single-seed probe).
10. **What retraining looks like per image** (expert analysis of the 64
    retrained models):
    - across seeds, each image's φ is roughly N(m_i, s(m_i));
    - the spread depends on difficulty: about 2.4 for hard images and 0.85
      for easy ones;
    - images vary independently, with no shared per-model offset;
    - forget images under retraining look exactly like validation images.

    F punishes both a too-narrow spread and a wrong per-image mean. Isotropic
    weight noise got the spread wrong, so it could not reach this. Per-image
    random targets (§3.6) build exactly this structure, which is why the
    score doubled.
11. **Ceiling for an output-only target: about F 0.10 — superseded by
    finding 15.** Any target built
    from the original model's outputs predicts the retrained per-image mean
    with R² ≈ 0.49–0.59 at best:

    | Target                          | R²        |
    | ------------------------------- | --------- |
    | original φ                      | 0.49      |
    | class quantile map              | 0.48      |
    | blend of the two                | 0.53      |
    | kNN local shell                 | 0.16–0.25 |

    The kNN probe is in `tasks/lash_local/`. LASH is now at this ceiling
    (0.097–0.102 across ρ, clean-set size and per-class targets). NegGrad+
    gets its better mean by training on retain data, which is not allowed
    here.
12. **ρ trades score against leak.**
    - Lower ρ widens the spread, which helps F up to about ρ 0.7, but moves
      targets further from the retrained mean, so U-LiRA rises (0.607 at
      ρ 0.5, 0.641 at ρ 0.5 with ε 0.6).
    - Higher ρ lowers U-LiRA (0.577 at ρ 0.9) but the models get too similar
      and F drops.
    - ρ = 0.7 is the best balance. Per-class targets and 5,000 vs 2,000 clean
      images changed little (±0.002).
13. **Repair helps slightly.** Hard push (ε 0.5, no keep) followed by the
    closed-form repair is the best variant at ρ 0.7. It raises FA (93.7)
    and MIA_eff falls to about 0.16–0.17, below retrained's 0.23, meaning
    the attack calls more forget images members.
14. **The remaining weakness is the per-model U-LiRA leak** (AUC ≈ 0.59 vs
    0.50). It comes from the same cause as the F ceiling: some per-image
    targets are far from what retraining would give.
15. **Near-copies of the image break that ceiling.** The original model's
    mean confidence over 16 slightly shifted and mirrored copies predicts
    the retrained mean far better than its confidence on the image itself
    (R² 0.74 vs 0.59). Memorisation is tied to the exact pixels. Using this
    as the copula's rank signal raised the score to 0.113–0.114 and lowered
    the leak (§4, rounds 7–8).
16. **The optimal ρ is predictable** (Prop. 7). ρ* is the rank correlation
    between the signal and one retrained model. A better signal needs less
    added randomness. F peaks slightly below ρ* (0.55–0.67 for ρ* 0.77),
    because F also rewards spread; U-LiRA keeps improving towards ρ*.

## 6. Open items / next steps

- **Superseded 2026-11-20 by v_r0.67_rep (score 0.070 at N = 64; §4).**
  Earlier: **c5k_r0.7_pc_rep on all 64 seeds** (`tasks/lash_final/`,
  `n32_fast.csv`, `n64_official.csv`).

  | method | N | score | F | RA | FA | TA | MIA_eff | ULiRA_AUC (N=32) |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | NegGrad+ | 64 | **0.0730** | 0.0724 | 99.99 | 90.5 | 87.7 | 0.262 | 0.516 |
  | **LASH** | 64 | 0.0601 | 0.0603 | 98.4 | 93.8 | 88.2 | 0.170 | 0.591 |
  | Sebastian | 64 | 0.0588 | 0.0632 | 96.6 | 87.9 | 83.9 | 0.236 | 0.497 |

  At N = 32 the scores are 0.091, 0.077 and 0.074 in the same order.
  - In the 26-method table (own original, N = 64), LASH would rank **2nd**,
    behind NegGrad+ (0.073) and ahead of Sebastian (0.059) and every other
    method.
  - LASH has the best TA of the three.
  - Its U-LiRA leak (0.59) is the largest of the three; NegGrad+ is 0.52 and
    Sebastian 0.50.
- **Runtime is not yet cheap.** One run took about 1,100 s on a shared
  Quadro RTX 8000 (6 concurrent runs plus other jobs). NegGrad+ took 83 s on
  an idle RTX PRO 6000, and retraining takes about 5 min there.
  - The cost comes from Σ_K over all 45k retain images, 200 PGD steps and
    the repair.
  - The Σ_K and geig statistics are fixed per original model, so they could
    be cached and reused across removal requests.
  - This needs a clean timing on an idle GPU.
- Reduce the U-LiRA leak without retain training, for example by making
  the target spread depend on each image's difficulty, s(m_i).
- Optional appendix only: a short repair-training step on a little retain
  data, offered as an add-on for clients who can afford it.
- Sweep 4 noise runs are superseded by the copula targets and were not
  re-run.

## 7. Files

| What                | Where                                                                           |
| ------------------- | ------------------------------------------------------------------------------- |
| Method              | `src/unlearn/methods/lash.py` (registered as `lash`, group `OURS`)        |
| Proofs              | `docs/lash.md` (Propositions 1–6)                                            |
| Scorer              | `scripts/quick_score.py` (`--fast` uses `src/unlearn/forget_quality_fast.py`) |
| Geometry diagnostic | `scripts/lash_geometry.py` → `tasks/lash_sweep/geometry.csv`               |
| Sweep 1             | `tasks/lash_sweep/` (`grid.txt`, `score_n16.csv`)                         |
| Sweep 2             | `tasks/lash_sweep2/` (`grid.txt`, `scores/*.csv`)                         |
| Sweeps 3–4         | `tasks/lash_sweep3/`, `tasks/lash_sweep4/` (`grid.txt`, logs; not scored) |
| Sweeps 5–6         | `tasks/lash_sweep5/`, `tasks/lash_sweep6/` (`grid.txt`, `fast/*.csv`)      |
| Final 64-seed run   | `tasks/lash_final/`                                                           |
| Run outputs         | `artifacts/cifar10/resnet18/v1/own/runs/lash/<tag>/`                          |
