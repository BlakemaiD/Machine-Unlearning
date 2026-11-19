#!/usr/bin/env python
"""Per-layer geometry behind LASH's guarantees (docs/lash.md), on the original model.

For every conv/linear layer: input dim d, number of keep patch vectors,
top generalised eigenvalues gamma of (Sigma_F, Sigma_K + reg I) (forget
energy per unit keep energy; Prop. 3) and the share of forget energy inside
the low-keep-energy subspace with rho = 0.01 (Prop. 2; ~0.01 if forget and
keep look alike there).

    python scripts/lash_geometry.py --csv out.csv
"""
import argparse, sys
import pandas as pd, torch
from common import add_common_args, exp_dir
from unlearn.data import get_loaders
from unlearn.device import DEVICE
from unlearn.models import load_checkpoint
from unlearn.methods import lash


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    L = get_loaders(a.dataset, a.data_root, a.forget_idx, batch_size=64, seed=42, shuffle_seed=0)
    net = load_checkpoint(exp_dir(a.out, a.dataset, a.arch, a.version) / "reference" / "original" / "seed_0000.pt")
    xc, _ = lash._balanced(L["val"], 200, 10)
    xr, _ = lash._balanced(L["retain"], 2000, 10)
    xk = torch.cat([xc, xr]).to(DEVICE)
    xf = lash._first(L["forget"], 2000).to(DEVICE)
    mods = lash._editable_modules(net)
    with torch.no_grad():
        sk, sf = lash._second_moments(net, mods, xk), lash._second_moments(net, mods, xf)
    rows = []
    for name, m in mods:
        K, Fm = sk[name], sf[name]
        d = K.shape[0]
        A = K + 1e-3 * torch.trace(K) / d * torch.eye(d, dtype=K.dtype, device=K.device)
        Li = torch.linalg.inv(torch.linalg.cholesky(A))
        g = torch.linalg.eigvalsh(Li @ Fm @ Li.T).flip(0)
        r10 = max(1, round(0.1 * d))
        P = lash._lowkeep_projector(K, 0.01)
        f_share = float(torch.trace(P.double() @ Fm) / torch.trace(Fm)) if P is not None else 0.0
        rows.append({"layer": name, "d": d, "gamma_1": float(g[0]), "gamma_r(10%)": float(g[r10 - 1]),
                     "forget_energy_in_lowkeep(rho=.01)": f_share})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    sys.exit(main())
