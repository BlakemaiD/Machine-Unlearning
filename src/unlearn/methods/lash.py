"""OURS: LASH - Layer-wise Angular SHell editing (weight-space attack + repair).

Idea (user's proposal, 2026-11-17). No retraining. Like an adversarial (PGD /
C&W) or backdoor-injection attack, but aimed at forgetting and applied to
the *weights* while the inputs stay fixed. Notation and proofs:
``docs/lash.md``.

Data: the forget set F, ``n_clean`` clean held-out images (default 200, from
the validation split, never trained on) and ``n_keep_retain`` retain images
(default 2,000). Clean + retain images form the keep set K. Batch norm is
frozen (eval mode); conv and linear layers are linear in their weights:
y = W x, with x an im2col patch.

Phase 1, push (projected gradient on the weights).
  Objective: move each forget sample, in every monitored layer, into the
  "shell" where clean samples of its class lie (angle to the class
  direction, radius, and output confidence; 1-D Wasserstein match of the
  standardised values to the clean ones), keep it no closer to other classes
  than clean samples are, and (``w_keep``) keep the outputs of K. Each step
  is a normalised gradient step on each weight tensor followed by
  projection onto the ball ||delta|| <= eps ||W_0||.
  Edit-space restriction (``subspace``), per layer, on the input side
  (delta <- delta P, P an orthogonal projector):
    "lowkeep": span of the eigenvectors of Sigma_K = E_K[x x^T] holding
               at most a fraction ``rho`` of the keep energy ("rarely used
               directions"). Guarantee: E_K ||delta x||^2 <= lambda_max(P) ||delta||_F^2.
    "geig":    span of the top generalised eigenvectors of
               Sigma_F v = gamma (Sigma_K + reg I) v ("forget-specific
               directions"). Guarantee: every edit direction has at least
               gamma_r times more forget energy than (regularised) keep energy.
               With ``geig_min_gamma`` the span is all directions with
               gamma >= geig_min_gamma, and layers without any are frozen.
  Neuron restriction (``channels``, backdoor-style), on the output side:
    "rare":   only output channels whose mean |activation| is highest on F
              relative to K (forget-specific neurons);
    "common": only the channels most used by K.
Phase 2, repair (``repair=True``; closed-form, C&W-style trade-off).
  Layer by layer, front to back, the new weights are the unique minimiser of
      E_K ||W x - W_0 x_0||^2 + mu E_F ||W x - W_p x_p||^2 + lam ||W - W_p||_F^2
  (x: input under the current model, x_0 under the original, x_p under the
  pushed model): keep outputs return to the original, forget outputs stay
  where the push put them, and the change from the pushed weights is
  minimal. Closed form:
      W = (W_0 C_K + mu W_p C_F + lam W_p) (S_K + mu S_F + lam I)^-1,
  S_K = E_K[x x^T], C_K = E_K[x_0 x^T], S_F = E_F[x x^T], C_F = E_F[x_p x^T].
"""

import copy
import math
import time
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..device import DEVICE
from ..models import get_classifier
from ..utils import merge_hparams

INFO = {"rank": None, "team": "ours",
        "family": "weight-space attack: class-shell push in rare subspaces + closed-form repair"}

DEFAULTS = {
    "n_clean": 200,          # clean held-out images (balanced across classes; val has 5,000)
    "clean_random": False,   # draw the clean images at random per seed (else the first ones)
    "n_keep_retain": 2000,   # retain images added to the keep set (balanced); -1 = all
    "n_keep_push": 2200,     # keep images whose outputs the push term preserves (first n)
    "n_stat_forget": 2000,   # forget images used for covariance / channel statistics
    # phase 1: push
    "attack": "pgd",         # "pgd" (projected) or "cw" (norm-penalised, Adam)
    "steps": 200,
    "eps": 0.02,             # per-tensor L2 radius relative to ||W_0||
    "step_frac": 0.1,        # step = step_frac * eps * ||W_0|| (cosine-decayed)
    "cw_c": 10.0,
    "cw_lr": 1e-4,
    "forget_batch": 512,
    "w_logit": 1.0,          # confidence shell at the logits
    "phi_target": "w2",      # "w2": match the batch's confidence distribution (monotone
                             # coupling); "copula": per-image random target, see _copula_targets
    "copula_rho": 0.9,       # copula: correlation with the image's own rank (1 = w2-like, 0 = iid)
    "copula_class": False,   # copula: rank within the class and use that class's clean quantiles
    "rank_signal": "orig",   # copula ranks from: "orig" = the original model's phi on the image,
                             # "views" = its mean phi over 16 shifted/flipped copies (see _view_phi)
    "view_shift": 2,         # views: shift in pixels (8 directions x {as is, flipped})
    "w_feat": 1.0,           # angle / radius shells, averaged over layers
    "w_sep": 1.0,            # other-class separation hinge
    "sep_margin": 1.0,
    "w_keep": 1.0,           # preservation of keep outputs during the push
    "layers": "all",         # loss taps: "all" stages + logits, "deep", "logits"
    "subspace": "none",      # "none", "lowkeep", "geig"
    "rho": 0.01,             # lowkeep: max keep-energy fraction of the edit subspace
    "geig_frac": 0.1,        # geig: fraction of input dims kept (top generalised eigvecs)
    "geig_reg": 1e-3,        # geig: ridge on Sigma_K, relative to tr(Sigma_K)/d
    "geig_min_gamma": 0.0,   # geig: if > 0, keep exactly the directions with gamma >= this
                             #       (layers without any are frozen); replaces geig_frac
    "channels": "all",       # "all", "rare", "common"
    "channel_frac": 0.1,     # fraction of output channels editable
    # randomised edit: models that unlearn the same forget set must vary as much
    # as independently retrained ones, so add a random edit inside the edit space
    "noise": 0.0,            # per-tensor L2 radius of the random edit, relative to ||W_0||
    "noise_at": "end",       # "start" (PGD random start, then pushed) or "end" (after push)
    # phase 2: closed-form repair
    "repair": False,
    "repair_mu": 1.0,        # weight of keeping the pushed forget outputs
    "repair_lam": 1e-2,      # ridge toward the pushed weights, relative to tr(S_K)/d
    # cost: second moments are summed per batch in this dtype and accumulated in
    # float64 (solves / eigendecompositions are always float64). float32 is ~25x
    # faster on GPUs with slow float64; the batch sums are well conditioned.
    "moment_dtype": "float64",
}


# ---------------------------------------------------------------- helpers

def _stage_modules(net):
    if all(hasattr(net, f"layer{i}") for i in range(1, 5)):
        return [getattr(net, f"layer{i}") for i in range(1, 5)]
    feats = getattr(net, "features", None)
    if feats is None:
        return []
    blocks = list(feats.children())
    idx = sorted({round(i * (len(blocks) - 1) / 3) for i in range(4)})
    return [blocks[i] for i in idx]


def _logit_conf(logits, y):
    logp = F.log_softmax(logits.float(), dim=1)
    lp = logp.gather(1, y[:, None]).squeeze(1)
    return lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))


class _Taps:
    def __init__(self, modules):
        self.out = [None] * len(modules)
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(modules)]

    def _hook(self, i):
        def fn(_m, _inp, o):
            self.out[i] = o.flatten(2).mean(2) if o.dim() == 4 else o
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def _balanced(loader, n, k):
    per = math.ceil(n / k)
    xs, ys, cnt = [], [], [0] * k
    for x, y in loader:
        for xi, yi in zip(x, y):
            c = int(yi)
            if cnt[c] < per:
                xs.append(xi)
                ys.append(c)
                cnt[c] += 1
        if min(cnt) >= per:
            break
    return torch.stack(xs), torch.tensor(ys)


def _first(loader, n):
    xs, n_got = [], 0
    for x, _ in loader:
        xs.append(x)
        n_got += len(x)
        if n_got >= n:
            break
    return torch.cat(xs)[:n]


def _forward(net, taps, x):
    logits = net(x)
    return logits, [t.float() for t in taps.out]


def _patches(mod, x, aug, dtype=torch.float64):
    """Layer input as rows of the linear map y = W x (im2col for conv)."""
    if isinstance(mod, nn.Conv2d):
        p = F.unfold(x, mod.kernel_size, dilation=mod.dilation, padding=mod.padding,
                     stride=mod.stride)
        p = p.transpose(1, 2).reshape(-1, p.shape[1])
    else:
        p = x.reshape(len(x), -1)
    p = p.to(dtype)
    if aug:
        p = torch.cat([p, torch.ones(len(p), 1, dtype=p.dtype, device=p.device)], 1)
    return p


def _editable_modules(net):
    return [(n, m) for n, m in net.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "groups", 1) == 1]


class _Stop(Exception):
    pass


def _module_inputs(net, mod, x, bs=500):
    """Yield the input of ``mod`` for batches of x (forward stops at mod)."""
    box = {}

    def pre(_m, inp):
        box["x"] = inp[0]
        raise _Stop

    h = mod.register_forward_pre_hook(pre)
    try:
        for xb in x.split(bs):
            try:
                net(xb)
            except _Stop:
                pass
            yield box["x"]
    finally:
        h.remove()


@torch.no_grad()
def _shell_stats(net, taps, x, y, k):
    logits, feats = _forward(net, taps, x)
    stats = []
    for h in feats:
        hn = F.normalize(h, dim=1)
        mu = torch.stack([F.normalize(hn[y == c].mean(0), dim=0) for c in range(k)])
        cos_all = hn @ mu.T
        cos_own = cos_all.gather(1, y[:, None]).squeeze(1)
        cos_oth = cos_all.masked_fill(F.one_hot(y, k).bool(), -2).max(1).values
        s = {"mu": mu, "oth_m": cos_oth.mean(), "oth_s": cos_oth.std().clamp_min(1e-4)}
        for name, v in (("cos", cos_own), ("rad", h.norm(dim=1))):
            m = torch.stack([v[y == c].mean() for c in range(k)])
            sd = (v - m[y]).std().clamp_min(1e-4)
            s.update({name + "_m": m, name + "_s": sd, name + "_z": ((v - m[y]) / sd).sort().values})
        stats.append(s)
    phi = _logit_conf(logits, y)
    pm = torch.stack([phi[y == c].mean() for c in range(k)])
    ps = (phi - pm[y]).std().clamp_min(1e-3)
    z = (phi - pm[y]) / ps
    return stats, {"m": pm, "s": ps, "z": z.sort().values,
                   "zc": [z[y == c].sort().values for c in range(k)]}


def _quantile(ref, u):
    """Linear-interpolated quantile of the sorted 1-D tensor ref at levels u in (0, 1)."""
    pos = u.clamp(0, 1) * (len(ref) - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=len(ref) - 1)
    return ref[lo] + (ref[hi] - ref[lo]) * (pos - lo)


def _copula_targets(z0, ref, rho, gen=None):
    """Per-image random targets with exactly the clean marginal (Gaussian copula).

    u0_i = (rank_i + 1/2)/n is image i's quantile under the original model and
    g_i = Phi^-1(u0_i). With xi_i ~ N(0, 1) iid, rho*g_i + sqrt(1 - rho^2)*xi_i is
    N(0, 1) whenever g_i is, so u_i = Phi(.) is uniform and the target
    Q_ref(u_i) follows the clean distribution (the same target as "w2"). Across
    seeds the targets of one image have correlation rho^2 in the Gaussian scale:
    rho = 1 recovers the monotone (w2) coupling, rho < 1 adds a per-image,
    rank-preserving-on-average spread.
    """
    n = len(z0)
    u0 = (z0.argsort().argsort().double() + 0.5) / n
    nd = torch.distributions.Normal(0.0, 1.0)
    g = nd.icdf(u0)
    xi = torch.randn(n, generator=gen, dtype=torch.float64, device=z0.device)
    u = nd.cdf(rho * g + math.sqrt(max(0.0, 1 - rho ** 2)) * xi)
    return _quantile(ref.double(), u).float()


@torch.no_grad()
def _view_phi(net, x, y, shift, bs=500):
    """Mean logit confidence of the original model over 16 near-copies of each image
    (cyclic shifts by +-shift px in 8 directions, each also mirrored). The exact
    training image carries the memorised confidence; its near-copies were never
    seen pixel for pixel, so their mean confidence ranks images closer to how a
    model that never saw them would (docs/lash.md, "Rank signal")."""
    s = shift
    offs = [(s, 0), (-s, 0), (0, s), (0, -s), (s, s), (-s, -s), (s, -s), (-s, s)]
    tot = torch.zeros(len(x), device=x.device)
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        for xv in (xb, xb.flip(3)):
            for dx, dy in offs:
                tot[i:i + bs] += _logit_conf(net(torch.roll(xv, (dy, dx), (2, 3))), yb)
    return tot / (2 * len(offs))


def _w1(z, ref):
    """Squared 1-D Wasserstein-2 distance between batch values z and the sorted reference."""
    q = (torch.arange(len(z), device=z.device, dtype=torch.float32) + 0.5) / len(z)
    pos = q * (len(ref) - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=len(ref) - 1)
    target = ref[lo] + (ref[hi] - ref[lo]) * (pos - lo)
    return (z.sort().values - target).pow(2).mean()


# ------------------------------------------------- edit-space restrictions

@torch.no_grad()
def _second_moments(net, mods, x, bs=500, dtype=torch.float64):
    """E[x x^T] of every module's input patches over images x (one forward pass)."""
    acc = {name: [0, 0] for name, _ in mods}
    hooks = []
    for name, m in mods:
        def pre(mod, inp, name=name):
            p = _patches(mod, inp[0], aug=False, dtype=dtype)
            acc[name][0] = acc[name][0] + (p.T @ p).double()
            acc[name][1] += len(p)
        hooks.append(m.register_forward_pre_hook(pre))
    try:
        for xb in x.split(bs):
            net(xb)
    finally:
        for h in hooks:
            h.remove()
    return {name: s / n for name, (s, n) in acc.items()}


def _lowkeep_projector(sk, rho):
    """Projector onto the smallest-eigenvalue directions holding <= rho of keep energy."""
    ev, u = torch.linalg.eigh(sk)
    ev = ev.clamp_min(0)
    keep = torch.cumsum(ev, 0) <= rho * ev.sum()
    if not keep.any():
        return None
    q = u[:, keep]
    return (q @ q.T).float()


def _geig_projector(sf, sk, frac, reg, min_gamma=0.0):
    """Orthogonal projector onto the span of the top generalised eigenvectors of (S_F, S_K + reg I).

    With min_gamma > 0 the span is exactly the eigenvectors with gamma >= min_gamma
    (None if there are none), so every edit direction has forget/keep energy ratio
    >= min_gamma (docs/lash.md, Prop. 3).
    """
    d = sk.shape[0]
    a = sk + reg * torch.trace(sk) / d * torch.eye(d, dtype=sk.dtype, device=sk.device)
    lc = torch.linalg.cholesky(a)
    li = torch.linalg.inv(lc)
    ev, w = torch.linalg.eigh(li @ sf @ li.T)       # symmetric reduction, ascending
    r = int((ev >= min_gamma).sum()) if min_gamma > 0 else max(1, int(round(frac * d)))
    if r == 0:
        return None
    v = li.T @ w[:, -r:]                             # generalised eigenvectors (top r)
    q, _ = torch.linalg.qr(v)
    return (q @ q.T).float()


@torch.no_grad()
def _channel_masks(net, mods, xk, xf, which, frac):
    acts = {}
    hooks = []
    for name, m in mods:
        def fn(_m, _i, o, name=name):
            a = o.abs().flatten(2).mean(2) if o.dim() == 4 else o.abs()
            acts.setdefault(name, []).append(a.sum(0))
        hooks.append(m.register_forward_hook(fn))

    def mean_act(x):
        acts.clear()
        for xb in x.split(500):
            net(xb)
        return {n: torch.stack(v).sum(0) / len(x) for n, v in acts.items()}

    ak, af = mean_act(xk), mean_act(xf)
    for h in hooks:
        h.remove()
    masks = {}
    for name, m in mods:
        score = af[name] / (ak[name] + 1e-6) if which == "rare" else ak[name]
        n = max(1, int(round(frac * len(score))))
        mask = torch.zeros_like(score)
        mask[score.topk(n).indices] = 1
        masks[name] = mask
    return masks


# ---------------------------------------------------------------- repair

@torch.no_grad()
def _repair(net, orig, pushed, mods, xk, xf, mu, lam, dtype=torch.float64):
    """Closed-form layer-wise minimiser (see module docstring), front to back."""
    mods_o = dict(orig.named_modules())
    mods_p = dict(pushed.named_modules())
    for name, m in mods:
        aug = m.bias is not None
        mo, mp = mods_o[name], mods_p[name]

        def moments(x, net_t, mod_t):
            s, c, n = 0, 0, 0
            for xin, xt in zip(_module_inputs(net, m, x), _module_inputs(net_t, mod_t, x)):
                p, pt = _patches(m, xin, aug, dtype), _patches(m, xt, aug, dtype)
                s = s + (p.T @ p).double()
                c = c + (pt.T @ p).double()
                n += len(p)
            return s / n, c / n

        s_k, c_k = moments(xk, orig, mo)
        s_f, c_f = moments(xf, pushed, mp)

        def mat(mod):
            w = mod.weight.detach().reshape(mod.weight.shape[0], -1).double()
            return torch.cat([w, mod.bias.detach().double()[:, None]], 1) if aug else w

        w0, wp = mat(mo), mat(mp)
        d = s_k.shape[0]
        lam_abs = lam * torch.trace(s_k) / d
        a = w0 @ c_k + mu * (wp @ c_f) + lam_abs * wp
        b = s_k + mu * s_f + lam_abs * torch.eye(d, dtype=s_k.dtype, device=s_k.device)
        w = torch.linalg.solve(b, a.T).T             # b symmetric
        if aug:
            m.bias.copy_(w[:, -1].to(m.bias.dtype))
            w = w[:, :-1]
        m.weight.copy_(w.reshape(m.weight.shape).to(m.weight.dtype))


# ---------------------------------------------------------------- method

class _Clock:
    """Wall time per phase (GPU synchronised), printed at the end of a run."""
    def __init__(self):
        self.t, self.parts = time.perf_counter(), []

    def mark(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.parts.append((name, now - self.t))
        self.t = now

    def report(self):
        tot = sum(s for _, s in self.parts)
        print("[lash timing] " + "  ".join(f"{n} {s:.1f}s" for n, s in self.parts)
              + f"  total {tot:.1f}s", flush=True)


def unlearning(net, retain, forget, validation, **hparams):
    hp = merge_hparams(DEFAULTS, hparams, "lash")
    clock = _Clock()
    mdt = {"float64": torch.float64, "float32": torch.float32}[hp["moment_dtype"]]
    net.to(DEVICE).eval()   # inputs fixed, batch norm frozen: only weights change
    for p in net.parameters():
        p.requires_grad_(False)
    orig = copy.deepcopy(net)
    k = get_classifier(net).out_features
    stages = _stage_modules(net)
    if hp["layers"] == "deep":
        stages = stages[-1:]
    elif hp["layers"] == "logits":
        stages = []
    taps = _Taps(stages)

    clean_src = (torch.utils.data.DataLoader(validation.dataset, batch_size=500, shuffle=True)
                 if hp["clean_random"] else validation)
    xc, yc = _balanced(clean_src, hp["n_clean"], k)
    xc, yc = xc.to(DEVICE), yc.to(DEVICE)
    stats, lstats = _shell_stats(net, taps, xc, yc, k)
    xk = xc
    if hp["n_keep_retain"] != 0:
        xr = (_first(retain, 10 ** 9) if hp["n_keep_retain"] < 0
              else _balanced(retain, hp["n_keep_retain"], k)[0])
        xk = torch.cat([xc, xr.to(DEVICE)])
    # statistics (covariances, channel scores, repair) use all of xk; the push's
    # keep term uses its first n_keep_push images (the clean ones first)
    xk_push = xk[: hp["n_keep_push"]] if hp["w_keep"] > 0 else xk[:0]
    xf_stat = _first(torch.utils.data.DataLoader(forget.dataset, batch_size=500, shuffle=True),
                     hp["n_stat_forget"]).to(DEVICE)
    with torch.no_grad():
        keep_logits, keep_feats = _forward(net, taps, xk_push) if len(xk_push) else _forward(net, taps, xc)
        keep_logp = F.log_softmax(keep_logits.float(), 1)
        keep_var = [f.var(0).mean().clamp_min(1e-6) for f in keep_feats]

    clock.mark("load+shells")
    mods = _editable_modules(net)
    # edit-space restrictions, computed on the original model
    proj, masks = {}, {}
    if hp["subspace"] != "none":
        sk = _second_moments(net, mods, xk, dtype=mdt)
        sf = _second_moments(net, mods, xf_stat, dtype=mdt) if hp["subspace"] == "geig" else None
        for name, _ in mods:
            proj[name] = (_lowkeep_projector(sk[name], hp["rho"]) if hp["subspace"] == "lowkeep"
                          else _geig_projector(sf[name], sk[name], hp["geig_frac"], hp["geig_reg"],
                                          hp["geig_min_gamma"]))
        del sk, sf
    if hp["channels"] != "all":
        masks = _channel_masks(net, mods, xk, xf_stat, hp["channels"], hp["channel_frac"])

    clock.mark("edit-space")
    names = [n for n, _ in mods]
    params = [m.weight for _, m in mods]
    theta0 = {n: p.detach().clone() for n, p in zip(names, params)}
    norm0 = {n: theta0[n].norm() for n in names}
    frozen = {n for n in names if hp["subspace"] != "none" and proj[n] is None}
    # whole forget set in memory, batches by index (per-image targets need indices)
    fx, fy = [], []
    for x, y in torch.utils.data.DataLoader(forget.dataset, batch_size=500, shuffle=False):
        fx.append(x)
        fy.append(y)
    fx, fy = torch.cat(fx).to(DEVICE), torch.cat(fy).to(DEVICE)
    phi_tgt = None
    if hp["phi_target"] == "copula":
        with torch.no_grad():
            if hp["rank_signal"] == "views":
                phi0 = _view_phi(net, fx, fy, hp["view_shift"])
            else:
                phi0 = _logit_conf(torch.cat([net(fx[i:i + 500]) for i in range(0, len(fx), 500)]), fy)
            z0 = (phi0 - lstats["m"][fy]) / lstats["s"]
        if hp["copula_class"]:
            phi_tgt = torch.empty_like(z0)
            for c in range(k):
                m = fy == c
                if m.any():
                    phi_tgt[m] = _copula_targets(z0[m], lstats["zc"][c], hp["copula_rho"])
        else:
            phi_tgt = _copula_targets(z0, lstats["z"], hp["copula_rho"])

    def forget_batches():
        while True:
            perm = torch.randperm(len(fx), device=DEVICE)
            for i in range(0, len(perm) - hp["forget_batch"] + 1, hp["forget_batch"]):
                yield perm[i:i + hp["forget_batch"]]

    forget_ld = forget_batches()
    n_layers = max(len(stages), 1)

    def restrict(n, g):
        """Map a weight-shaped tensor into the allowed edit space (linear, idempotent)."""
        if n in frozen:
            return torch.zeros_like(g)
        if n in proj:
            g = (g.reshape(g.shape[0], -1) @ proj[n]).reshape(g.shape)
        if n in masks:
            g = g * masks[n].view(-1, *([1] * (g.dim() - 1)))
        return g

    def objective():
        idx = next(forget_ld)
        xf, yf = fx[idx], fy[idx]
        nf = len(xf)
        use_keep = len(xk_push) > 0
        logits, feats = _forward(net, taps, torch.cat([xf, xk_push]) if use_keep else xf)
        lf, lk = logits[:nf], logits[nf:]
        z_phi = (_logit_conf(lf, yf) - lstats["m"][yf]) / lstats["s"]
        loss = hp["w_logit"] * (_w1(z_phi, lstats["z"]) if phi_tgt is None
                                else (z_phi - phi_tgt[idx]).pow(2).mean())
        keep = (F.kl_div(F.log_softmax(lk.float(), 1), keep_logp, log_target=True,
                         reduction="batchmean") if use_keep else 0.0)
        shell, sep = 0.0, 0.0
        for h_all, s, f0, v0 in zip(feats, stats, keep_feats, keep_var):
            h = h_all[:nf]
            hn = F.normalize(h, dim=1)
            cos_all = hn @ s["mu"].T
            cos_own = cos_all.gather(1, yf[:, None]).squeeze(1)
            shell = (shell + _w1((cos_own - s["cos_m"][yf]) / s["cos_s"], s["cos_z"])
                     + _w1((h.norm(dim=1) - s["rad_m"][yf]) / s["rad_s"], s["rad_z"]))
            cos_oth = cos_all.masked_fill(F.one_hot(yf, k).bool(), -2).max(1).values
            sep = sep + F.relu((cos_oth - s["oth_m"]) / s["oth_s"] - hp["sep_margin"]).pow(2).mean()
            if use_keep:
                keep = keep + (h_all[nf:] - f0).pow(2).mean() / v0
        if stages:
            loss = loss + (hp["w_feat"] * shell + hp["w_sep"] * sep) / n_layers
        return loss + hp["w_keep"] * keep

    def add_noise():
        """Random edit xi = r * R(Z)/||R(Z)||, Z iid Gaussian: lies in the edit space."""
        with torch.no_grad():
            for n, p in zip(names, params):
                z = restrict(n, torch.randn_like(p))
                zn = z.norm()
                if zn > 0:
                    p += hp["noise"] * norm0[n] * z / zn

    if hp["noise"] > 0 and hp["noise_at"] == "start":
        add_noise()
    clock.mark("targets")
    # phase 1: push
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=hp["cw_lr"]) if hp["attack"] == "cw" else None
    for t in range(hp["steps"]):
        loss = objective()
        if hp["attack"] == "cw":
            loss = loss + hp["cw_c"] * sum((p - theta0[n]).pow(2).sum() / norm0[n] ** 2
                                           for n, p in zip(names, params))
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            if hp["attack"] == "cw":
                for n, p, g in zip(names, params, grads):
                    p.grad = restrict(n, g)
                opt.step()
                for n, p in zip(names, params):  # keep delta in the edit space
                    p.copy_(theta0[n] + restrict(n, p - theta0[n]))
            else:
                scale = 0.5 * (1 + math.cos(math.pi * t / hp["steps"]))
                for n, p, g in zip(names, params, grads):
                    g = restrict(n, g)
                    gn = g.norm()
                    if gn == 0:
                        continue
                    r = hp["eps"] * norm0[n]
                    p -= hp["step_frac"] * r * scale * g / gn
                    d = p - theta0[n]
                    dn = d.norm()
                    if dn > r:
                        p.copy_(theta0[n] + d * (r / dn))
    for p in params:
        p.requires_grad_(False)
        p.grad = None
    taps.remove()
    clock.mark("push")
    if hp["noise"] > 0 and hp["noise_at"] == "end":
        add_noise()

    # phase 2: closed-form repair
    if hp["repair"]:
        pushed = copy.deepcopy(net)
        _repair(net, orig, pushed, mods, xk, xf_stat, hp["repair_mu"], hp["repair_lam"], mdt)
        del pushed
        clock.mark("repair")
    clock.report()
    del orig
    for p in net.parameters():
        p.requires_grad_(True)
    net.eval()
    return net
