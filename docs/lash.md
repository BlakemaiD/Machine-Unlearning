# LASH: Layer-wise Angular SHell editing

Our own unlearning method (`src/unlearn/methods/lash.py`). It does no
retraining. Like an adversarial (PGD, C&W) or backdoor-injection attack, it
edits the **weights** while the inputs stay fixed, but the goal is to forget.
This note gives the construction and the guarantees, with proofs.

## Setting and notation

- F: forget set. K: keep set = 200 clean held-out images (validation split,
  never trained on) + retain images: 2,000 class-balanced, or the whole
  retain set (`n_keep_retain=-1`), which the γ-threshold variant needs (see
  below). The push's keep term uses only the first `n_keep_push` of them.
- Batch norm is frozen at its running statistics (eval mode), so between
  nonlinearities every conv/linear layer is **linear in its weights**:
  y = W x, with W ∈ R^{m×d} the reshaped kernel and x ∈ R^d an im2col patch
  (d = C_in·k_h·k_w). A weight change ΔW changes the layer's output by
  exactly Δy = ΔW x.
- Second moments of a layer's input patches:
  Σ_K = E_K[x xᵀ], Σ_F = E_F[x xᵀ] (averaged over images and positions).
- For any ΔW: E_K‖ΔW x‖² = tr(ΔW Σ_K ΔWᵀ), and likewise for F.

## The geometric target: class shells

For a monitored layer (ResNet stage outputs, average-pooled, plus the logits),
the clean images define for every class c:

- a direction μ_c = normalise(mean of normalised features of class c);
- the **angle** of a sample to its class, cos(h, μ_y), and its **radius**
  ‖h‖, each standardised per class with the clean mean and the pooled std
  (z-scores);
- at the logits, the competition's logit-scaled confidence
  φ = log p_y − log(1 − p_y), standardised the same way.

The set of (angle, radius) values that clean samples take is the class
**shell**: where unseen samples of that class sit. A model that never saw a
forget sample places it like any other unseen sample, so the target is to
make the forget samples' z-scores distributed like the clean ones.

**Matching loss.** For a batch of forget z-scores z_1..z_n and sorted clean
z-scores r, the loss is (1/n) Σ_i (z_(i) − Q_r((i − ½)/n))², with z_(i) the
sorted batch and Q_r the empirical clean quantile function.

*Fact (1-D optimal transport).* On the real line, the monotone (quantile)
coupling is optimal for the squared cost, so this loss is exactly the
squared Wasserstein-2 distance between the batch's empirical distribution
and the clean one (with Q_r interpolated). It is zero iff the two
distributions coincide. Memorised samples (too confident, too close to μ_y)
are moved out to the shell, and the hardest ones can become misclassified,
as happens for unseen data.

**Separation hinge.** With s = max_{c≠y} cos(h, μ_c), standardised by the
clean statistics, the penalty max(0, z_s − m)² stops any forget sample from
getting closer to another class than clean samples do.

## Phase 1: push (projected gradient on the weights)

Each step takes the gradient G of the objective (shell + separation +
optionally keeping the K outputs), maps it into the allowed edit space with a
linear map R, and moves each weight tensor:

    W ← W − η_t R(G)/‖R(G)‖,   then   ΔW ← ΔW · min(1, ε‖W_0‖/‖ΔW‖)

with η_t = step_frac·ε‖W_0‖·½(1 + cos πt/T).

The map R is right-multiplication by a projector P (input-side subspace)
and/or a 0/1 mask on output channels (rows). The two commute (one acts on
columns, the other on rows), and R is linear and idempotent.

**Proposition 1 (iterates stay admissible).** If ΔW_0 ∈ range(R) with
‖ΔW_0‖_F ≤ ε‖W_0‖_F (e.g. ΔW_0 = 0, or a random start, see below), then for all t,
ΔW_t ∈ range(R) and ‖ΔW_t‖_F ≤ ε‖W_0‖_F.

*Proof.* By induction. ΔW_t ∈ range(R) and R(G) ∈ range(R), so their
difference is in range(R) (a linear subspace). The ball projection only
rescales, which preserves the subspace and enforces the norm bound. ∎

### Rarely used directions (`subspace="lowkeep"`)

Let Σ_K = U Λ Uᵀ with eigenvalues λ_1 ≤ … ≤ λ_d. Take S, the largest prefix
with Σ_{i∈S} λ_i ≤ ρ·tr Σ_K, and P = U_S U_Sᵀ. This is the input-space
analogue of injecting a backdoor into rarely used neurons.

**Proposition 2 (bounded change on the keep set).** For every edit with
ΔW = ΔW P:

    E_K‖Δy‖² = tr(ΔW P Σ_K P ΔWᵀ) ≤ λ_max(S) ‖ΔW‖_F² ≤ ρ · tr(Σ_K) · ε² ‖W_0‖_F².

*Proof.* P Σ_K P = U_S Λ_S U_Sᵀ ≼ λ_max(S) P, and tr(ΔW P ΔWᵀ) = ‖ΔW‖_F² for
ΔW = ΔW P. Also λ_max(S) ≤ Σ_{i∈S} λ_i ≤ ρ tr Σ_K. The last step uses
Proposition 1. ∎

So the mean squared change of every layer's output on the keep set is at most
a ρ-fraction of the keep energy times the squared edit size. With ρ → 0 (a
true null space, Σ_K P = 0) the change is exactly zero. By induction over
layers (each layer's keep inputs are then unchanged), the network's outputs
on K are then exactly preserved.

### Forget-specific directions (`subspace="geig"`)

Let A = Σ_K + αI (α > 0, α = reg·tr Σ_K/d). Solve the generalised eigenproblem
Σ_F v = γ A v with γ_1 ≥ … ≥ γ_d. Let V_r be the top r eigenvectors and P
the orthogonal projector onto span(V_r).

**Proposition 3 (optimal selectivity).**
(a) V_r maximises tr(Vᵀ Σ_F V) subject to Vᵀ A V = I (Ky Fan / generalised
Rayleigh-Ritz): no other r-dimensional subspace carries more forget energy
per unit of keep energy.
(b) For every edit with ΔW = ΔW P:

    E_K‖Δy‖²  ≤  (1/γ_r) · E_F‖Δy‖².

*Proof.* (a) With A = LLᵀ and u = Lᵀv, this becomes the symmetric eigenproblem
for L⁻¹Σ_F L⁻ᵀ, so Ky Fan's maximum principle applies. (b) The v_i are both
A-orthogonal and Σ_F-orthogonal. Write a row w of ΔW in span(V_r) as
w = Σ c_i v_i. Then wᵀΣ_F w = Σ γ_i c_i² ≥ γ_r Σ c_i² = γ_r wᵀAw ≥ γ_r wᵀΣ_K w.
Summing over the rows gives the claim. ∎

So in every layer, the edit changes the keep outputs at most 1/γ_r times as
much as it changes the forget outputs.

**Choosing r by a selectivity threshold (`geig_min_gamma=γ_min`).** Instead
of a fixed fraction, keep exactly the directions with γ_i ≥ γ_min. Then
Proposition 3(b) holds with 1/γ_r ≤ 1/γ_min in **every** edited layer, and
a layer with no such direction is frozen (its edit is exactly zero). One
guarantee then covers the whole network: no layer is edited in a direction
that changes keep outputs more than 1/γ_min times as much as forget outputs.

*Estimation caveat.* In a deep layer d can exceed the number of keep
patches, so Σ_K estimated from 2,000 images is rank-deficient and some
"forget-specific" directions are merely directions the keep sample missed.
Estimating Σ_K on the whole retain set removes this. Measured on our model
(`scripts/lash_geometry.py`): γ_1 ≈ 1.0–1.3 in the early layers (forget and
retain look the same, so nothing there can be edited selectively) and 4–11
in layer3/layer4.0.

### Rare or common neurons (`channels="rare" | "common"`)

Backdoor-style selection of output channels (rows of W). "rare" keeps the top
fraction by mean |activation| on F divided by that on K (forget-specific
neurons); "common" keeps the top fraction by activation on K. Proposition 1
holds (a row mask is a linear idempotent R). Unselected channels' outputs are
exactly unchanged.

## Randomised edit (`noise`, `noise_at`)

The competition metric does not compare one model with one model. For each
forget example it compares the **distribution** of its confidence across N
unlearned models with that across N independently retrained models. Our
original model is the same for every run, so a deterministic edit produces
N almost identical models. Their confidences are then nearly a point mass,
which a test separates from the retrained spread, and the example scores
nothing however good its mean is. Measured median per-example std across 16
models: retrained 1.36, NegGrad+ 1.07, Sebastian 1.51, LASH without noise
0.02–0.45. F ranks the LASH variants in exactly that order.

So LASH also adds a random edit in the same edit space:

    ξ = r ‖W_0‖ · R(Z)/‖R(Z)‖_F,   Z with iid N(0, 1) entries,

either before the push (`start`, the PGD random start: the push and keep
term then shape it) or after it (`end`, optionally followed by the repair).

**Proposition 4 (the noise obeys the same guarantees).** ξ ∈ range(R), so
(a) with `geig`, E_K‖ξ x‖² ≤ (1/γ_r) E_F‖ξ x‖²: the randomness lands mainly
on the forget outputs; (b) with `lowkeep`, E_K‖ξ x‖² ≤ ρ tr(Σ_K) r²‖W_0‖²;
(c) frozen layers and unselected channels stay unchanged; (d) the total
edit satisfies ‖ΔW‖_F ≤ ε‖W_0‖ (`start`, as the ball projection follows) or
≤ (ε + r)‖W_0‖ (`end`, triangle inequality). With the repair after it,
Proposition 5 applies with W_p the noised weights: keep outputs are
restored while forget outputs keep their randomness.

*Proof.* R is linear, so R(Z) ∈ range(R), and scaling preserves this. Then
apply Propositions 1–3 to ΔW = ξ; (d) follows from Proposition 1 and the
triangle inequality. ∎

## Phase 2: closed-form repair (C&W-style trade-off)

After a hard push (no keep term, larger ε), the layers are repaired front to
back. With x the layer input under the current network, x_0 under the
original and x_p under the pushed one, each layer's new weights minimise

    J(W) = E_K‖W x − W_0 x_0‖² + μ E_F‖W x − W_p x_p‖² + λ‖W − W_p‖_F².

- The first term restores the keep outputs.
- The second keeps the forget outputs where the push put them.
- The third keeps the correction minimal (the C&W norm term).

**Proposition 5 (exact repair).** For λ > 0, J is strictly convex and its
unique minimiser is

    W* = (W_0 C_K + μ W_p C_F + λ W_p)(S_K + μ S_F + λ I)⁻¹,

with S_K = E_K[x xᵀ], C_K = E_K[x_0 xᵀ], S_F = E_F[x xᵀ] and
C_F = E_F[x_p xᵀ]. In particular J(W*) ≤ J(W_p) and J(W*) ≤ J(W_0).

*Proof.* J is a quadratic in W. Its Hessian, 2(S_K + μS_F + λI) ⊗ I, is
positive definite for λ > 0. Setting the gradient
2[(W S_K − W_0 C_K) + μ(W S_F − W_p C_F) + λ(W − W_p)] to zero gives the
formula. Any other point, W_p or W_0 included, has J at least J(W*). ∎

For the first repaired layer, x = x_0 = x_p, so
J(W_p) = E_K‖(W_p − W_0)x‖²: the repair can only reduce the push's damage on
the keep set while holding the forget outputs.

## What the guarantees do and do not cover

- They are **per layer** and stated for the second moments of the measured
  keep and forget images. For unseen retain images they hold in expectation,
  up to the estimation error of Σ_K. Between layers there are ReLUs, so there
  is no global bound beyond the exact null-space case (ρ → 0).
- They bound how much the keep data **changes**. Whether the forget data
  then looks unseen is measured by the competition metric F.
- Forget and retain images are drawn from the same distribution, so in
  expectation Σ_F ≈ Σ_K over a large sample, and γ_r is close to 1 except
  where few patches per image exist (the deep layers). This limits how
  selective any fixed linear subspace can be. The experiments measure how
  much remains.
