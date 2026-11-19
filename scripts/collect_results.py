#!/usr/bin/env python
"""Gather one run's results into a single versioned folder: results/<version>_<date>_<time>/.

    python scripts/collect_results.py --run-id v1_2026-11-09_1931 \
        --experiments cifar10_resnet18 --job 10709

Copies the small, shareable outputs (tables, plots, per-example epsilons, task
status) and writes ``manifest.json``. Large artifacts (model checkpoints and
per-run logits) stay in ``artifacts/<dataset>/<arch>/<version>/``; the manifest
records where. ``finding_summary.md`` (written by hand) goes next to it.

Layout::

    results/<run-id>/
        manifest.json
        <experiment>/
            reference_models.csv      every original / retrained model
            task_status.tsv           exit code + runtime of every task
            <orig>/                   own | challenge
                summary.csv .md .json, per_run.csv, eps/, plots/
"""

import argparse
import json
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from common import REPO, code_hash, exp_dir


def reference_table(base):
    rows = []
    for card in sorted((base / "reference").glob("*/seed_*.json")):
        m = json.loads(card.read_text())
        rows.append({"role": m["role"], "seed": m["seed"],
                     **{k: m["metrics"][k] for k in
                        ["retain_acc", "forget_acc", "test_acc", "val_acc",
                         "mia_forget_vs_test", "mia_efficacy"]},
                     "train_minutes": m["seconds"] / 60, "gpu": m["env"].get("gpu"),
                     "code_sha": m["env"].get("code_sha"),
                     "checkpoint": str(card.with_suffix(".pt").relative_to(REPO))})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="e.g. v1_2026-11-09_1931")
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--job", default=None, help="Slurm job id")
    p.add_argument("--config", default=str(REPO / "configs" / "experiments.yaml"))
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dest = REPO / "results" / a.run_id
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": a.run_id, "collected": datetime.now().isoformat(timespec="seconds"),
                "slurm_job": a.job, "host": socket.gethostname(), "code_sha": code_hash(),
                "experiments": {}}

    for name in a.experiments:
        e = {**cfg["defaults"], **cfg["experiments"][name]}
        out = REPO / e.get("out", "artifacts")
        base = exp_dir(out, e["dataset"], e["arch"], str(e["version"]))
        edest = dest / name
        edest.mkdir(exist_ok=True)

        ref = reference_table(base)
        ref.to_csv(edest / "reference_models.csv", index=False)
        status = REPO / "tasks" / name / "logs" / "status.tsv"
        if status.exists():
            shutil.copy(status, edest / "task_status.tsv")

        origs = {}
        for orig in e["origs"]:
            src = base / orig / "results"
            if not src.exists():
                print(f"[warn] no results for {name}/{orig}")
                continue
            shutil.copytree(src, edest / orig, dirs_exist_ok=True)
            origs[orig] = {"results": str((edest / orig).relative_to(REPO)),
                           "runs": str((base / orig / "runs").relative_to(REPO)),
                           "models": str((base / orig / "models").relative_to(REPO))}

        manifest["experiments"][name] = {
            "dataset": e["dataset"], "arch": e["arch"], "version": str(e["version"]),
            "n_retrain": e["n_retrain"], "n_runs": e["n_runs"], "methods": e["methods"],
            "extra_hparams": e.get("hparams"), "batch_size": e["batch_size"],
            "artifacts_dir": str(base.relative_to(REPO)),
            "reference_models": int(len(ref)), "origs": origs,
        }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"collected into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
