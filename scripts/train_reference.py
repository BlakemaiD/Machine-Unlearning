#!/usr/bin/env python
"""Train one reference model from scratch: the original (all training data) or a
retrain-from-scratch oracle (retain set only), then save its checkpoint and logits.

    python scripts/train_reference.py --role retrain --seed 3
    python scripts/train_reference.py --role original --seed 0 --arch mobilenet_v2
    python scripts/train_reference.py --role retrain --seed 0 --recipe-override epochs=40

Skips if the output .npz already exists (safe to resubmit).
"""

import argparse
import json
import sys

from common import (add_common_args, environment_info, parse_kv, reference_path,
                    save_model)
from unlearn.data import get_loaders, get_train_arrays
from unlearn.evaluation import collect_outputs, run_metrics, save_run
from unlearn.train import RECIPES, train_from_scratch
from unlearn.utils import track_cost


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--role", required=True, choices=["original", "retrain"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--recipe", default=None, help="key of unlearn.train.RECIPES (default: dataset)")
    p.add_argument("--recipe-override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    npz = reference_path(a.out, a.dataset, a.arch, a.version, a.role, a.seed, "npz")
    if npz.exists() and not a.force:
        print(f"[skip] {npz} exists")
        return 0

    arrays = get_train_arrays(a.dataset, a.data_root, a.forget_idx)
    indices = arrays["retain"] if a.role == "retrain" else arrays["train"]
    recipe = a.recipe or a.dataset
    # distinct seed streams for originals vs retrains
    seed = a.seed + (0 if a.role == "retrain" else 100_000)

    with track_cost() as cost:
        net, cfg = train_from_scratch(arrays["x"], arrays["y"], indices, seed, recipe=recipe,
                                      overrides=parse_kv(a.recipe_override), arch=a.arch,
                                      num_classes=arrays["num_classes"], log_every=5)

    loaders = get_loaders(a.dataset, a.data_root, a.forget_idx)
    outputs = collect_outputs(net, loaders)
    meta = {"kind": "reference", "role": a.role, "dataset": a.dataset, "arch": a.arch,
            "seed": a.seed, "torch_seed": seed, "recipe_name": recipe, "recipe": cfg,
            "n_train": int(len(indices)), "seconds": cost["seconds"],
            "peak_mem_mb": cost["peak_mem_mb"], "env": environment_info()}
    run = dict(outputs)
    run.update({k: v.astype("float32") for k, v in outputs.items() if k.endswith("logits")})
    run.update({k: v.astype("int64") for k, v in outputs.items() if k.endswith("targets")})
    m = run_metrics(run)
    meta["metrics"] = m
    meta["version"] = a.version
    save_run(npz, outputs, meta)
    if not a.no_save_model:
        save_model(net, reference_path(a.out, a.dataset, a.arch, a.version, a.role, a.seed, "pt"),
                   {**meta, "arch": a.arch, "state_dict": "torch.load(...) -> load_state_dict"})
    print(json.dumps({"role": a.role, "seed": a.seed, "seconds": round(cost["seconds"], 1),
                      **{k: round(v, 4) for k, v in m.items() if k.endswith("acc")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
