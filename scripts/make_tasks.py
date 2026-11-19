#!/usr/bin/env python
"""Turn configs/experiments.yaml into ordered task files for scripts/pool.py.

    python scripts/make_tasks.py smoke
    python scripts/make_tasks.py cifar10_resnet18 cifar10_mobilenet_v2 --out-dir tasks/main

Writes ``1_reference.txt`` (original + oracle models) and ``2_unlearn.txt``
(every method x seed x original). One shell command per line; every command
skips itself if its output already exists, so the task files can be re-run.
"""

import argparse
import shlex
import sys
from pathlib import Path

import yaml

from common import REPO
from unlearn.methods import BASELINES, COMPETITION, MODULES, SOTA


def method_list(spec):
    if spec == "all":
        return list(BASELINES) + list(COMPETITION) + list(SOTA)
    if spec == "sota":
        return list(SOTA)
    if spec == "competition":
        return list(COMPETITION)
    if spec == "baselines":
        return list(BASELINES)
    unknown = set(spec) - set(MODULES)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    return list(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiments", nargs="+")
    p.add_argument("--config", default=str(REPO / "configs" / "experiments.yaml"))
    p.add_argument("--out-dir", default=None, help="default: tasks/<first experiment>")
    p.add_argument("--python", default="python")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out_dir = Path(a.out_dir or REPO / "tasks" / a.experiments[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_tasks, unl_tasks = [], []

    for name in a.experiments:
        e = {**cfg["defaults"], **cfg["experiments"][name]}
        if not e.get("enabled", True):
            raise SystemExit(f"experiment {name} is disabled in {a.config}")
        common = ["--dataset", e["dataset"], "--arch", e["arch"], "--version", str(e["version"])]
        if e.get("data_root"):
            common += ["--data-root", e["data_root"]]
        if e.get("out"):
            common += ["--out", str(REPO / e["out"])]
        recipe = (["--recipe-override", *e["recipe_override"]]
                  if e.get("recipe_override") else [])
        nosave = [] if e.get("save_models", True) else ["--no-save-model"]

        def cmd(script, *args):
            return " ".join(shlex.quote(x) for x in
                            [a.python, str(REPO / "scripts" / script), *common, *args])

        if "own" in e["origs"]:
            # always saved: every "--orig own" unlearning run loads it
            ref_tasks.append(cmd("train_reference.py", "--role", "original", "--seed", "0", *recipe))
        for s in range(e["n_retrain"]):
            ref_tasks.append(cmd("train_reference.py", "--role", "retrain", "--seed", str(s), *recipe,
                                 *nosave))

        extra = e.get("hparams") or {}
        for orig in e["origs"]:
            for m in method_list(e["methods"]):
                settings = {"default": {}, **extra.get(m, {})}
                for tag, hp in settings.items():
                    hp_args = ["--hp", *[f"{k}={v}" for k, v in hp.items()]] if hp else []
                    for s in range(e["n_runs"]):
                        unl_tasks.append(cmd("run_unlearning.py", "--method", m, "--seed", str(s),
                                             "--orig", orig, "--tag", tag,
                                             "--batch-size", str(e["batch_size"]), *nosave, *hp_args))

    # Interleave methods so slow and fast methods share GPUs evenly.
    unl_tasks.sort(key=lambda c: (int(c.split("--seed ")[1].split()[0]), c))
    (out_dir / "1_reference.txt").write_text("\n".join(ref_tasks) + "\n")
    (out_dir / "2_unlearn.txt").write_text("\n".join(unl_tasks) + "\n")
    print(f"{len(ref_tasks)} reference tasks, {len(unl_tasks)} unlearning tasks -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
