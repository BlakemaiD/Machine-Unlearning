#!/usr/bin/env python
"""Run one unlearning method once (one seed) and save its logits, cost and model.

    python scripts/run_unlearning.py --method fanchuan_1 --seed 0
    python scripts/run_unlearning.py --method kookmin_2 --seed 5 --orig challenge
    python scripts/run_unlearning.py --method neggrad_plus --seed 0 --tag a0.95 --hp alpha=0.95

Every run starts from a fresh copy of the same original model (Reuse-N-1
protocol of the competition): ``--orig own`` = our reference/original/seed_0000.pt,
``--orig challenge`` = the challenge's provided checkpoint (ResNet-18 / CIFAR-10).
Skips if the output .npz already exists (safe to resubmit).
"""

import argparse
import json
import sys
import traceback

import numpy as np

from common import (CHALLENGE_WEIGHTS, add_common_args, environment_info, parse_kv,
                    reference_path, run_path, save_model)
from unlearn.data import get_loaders
from unlearn.evaluation import collect_outputs, run_metrics, save_run
from unlearn.methods import MODULES
from unlearn.models import load_checkpoint
from unlearn.utils import seed_everything, track_cost


def original_checkpoint(a):
    if a.orig == "challenge":
        key = (a.dataset, a.arch)
        if key not in CHALLENGE_WEIGHTS:
            raise SystemExit(f"no challenge checkpoint for {key}; use --orig own")
        return CHALLENGE_WEIGHTS[key]
    path = reference_path(a.out, a.dataset, a.arch, a.version, "original", a.orig_seed, "pt")
    if not path.exists():
        raise SystemExit(f"{path} missing: train it with train_reference.py --role original")
    return path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--method", required=True, choices=sorted(MODULES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--orig", default="own", choices=["own", "challenge"])
    p.add_argument("--orig-seed", type=int, default=0)
    p.add_argument("--tag", default="default", help="name for this hyperparameter setting")
    p.add_argument("--hp", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--batch-size", type=int, default=64,
                   help="loader batch size handed to methods (competition: 64)")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    npz = run_path(a.out, a.dataset, a.arch, a.version, a.orig, a.method, a.tag, a.seed)
    if npz.exists() and not a.force:
        print(f"[skip] {npz} exists")
        return 0

    hp = parse_kv(a.hp)
    # split seed fixed (same test/val for all runs); shuffle order differs per run
    loaders = get_loaders(a.dataset, a.data_root, a.forget_idx, batch_size=a.batch_size,
                          seed=42, shuffle_seed=10_000 + a.seed)
    ckpt = original_checkpoint(a)
    num_classes = int(loaders["test"].dataset.tensors[1].max()) + 1
    net = load_checkpoint(ckpt, arch=a.arch, num_classes=num_classes)

    module = MODULES[a.method]
    seed_everything(a.seed)
    try:
        with track_cost() as cost:
            net = module.unlearning(net, loaders["retain"], loaders["forget"], loaders["val"], **hp)
    except Exception:
        traceback.print_exc()
        return 1

    outputs = collect_outputs(net, loaders)
    meta = {"kind": "unlearning", "method": a.method, "tag": a.tag, "info": module.INFO,
            "hparams_override": hp, "defaults": getattr(module, "DEFAULTS", {}),
            "dataset": a.dataset, "arch": a.arch, "orig": a.orig, "orig_checkpoint": str(ckpt),
            "seed": a.seed, "batch_size": a.batch_size, "seconds": cost["seconds"],
            "peak_mem_mb": cost["peak_mem_mb"], "env": environment_info()}
    run = dict(outputs)
    run.update({k: v.astype("float32") for k, v in outputs.items() if k.endswith("logits")})
    run.update({k: v.astype("int64") for k, v in outputs.items() if k.endswith("targets")})
    diverged = not all(np.isfinite(v).all() for k, v in run.items() if k.endswith("logits"))
    if diverged:
        print(f"WARNING: {a.method} seed {a.seed} produced non-finite outputs (diverged)")
        for k in list(run):
            if k.endswith("logits"):
                run[k] = np.nan_to_num(run[k], nan=0.0, posinf=0.0, neginf=0.0)
    m = run_metrics(run)
    meta.update({"metrics": m, "diverged": diverged, "version": a.version})
    save_run(npz, outputs, meta)
    if not a.no_save_model:
        save_model(net, run_path(a.out, a.dataset, a.arch, a.version, a.orig, a.method, a.tag,
                                 a.seed, ext="pt"), meta)
    print(json.dumps({"method": a.method, "seed": a.seed, "seconds": round(cost["seconds"], 1),
                      **{k: round(v, 4) for k, v in m.items()
                         if k.endswith("acc") or k.startswith("mia")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
