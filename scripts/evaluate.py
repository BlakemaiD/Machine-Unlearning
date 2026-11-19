#!/usr/bin/env python
"""Aggregate every saved run into the full metric table (no GPU training needed).

    python scripts/evaluate.py --dataset cifar10 --arch resnet18 --orig own
    python scripts/evaluate.py --orig challenge --methods fanchuan_1 kookmin_2

Writes to ``<out>/<dataset>/<arch>/<orig>/results/``:

* ``summary.csv`` / ``summary.json`` - one row per (method, tag), every metric
  (mean and std over runs, gaps to the retrained oracle, competition F and
  final score, divergences, cost);
* ``summary.md``  - the main comparison table, sorted by competition score;
* ``per_run.csv`` - per-run scalar metrics of every unlearned and retrained model;
* ``eps/<method>__<tag>.npy`` - per-example epsilon over the forget set;
* ``plots/`` - score bars, epsilon histograms, forget-vs-test loss histograms,
  confidence distributions vs retrain for a few forget examples.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from common import add_common_args, exp_dir
from unlearn.evaluation import load_runs, run_metrics, summarize

MAIN_COLUMNS = [
    ("competition_score", "Score"), ("forget_quality_F", "F"),
    ("retain_acc", "RA"), ("forget_acc", "FA"), ("test_acc", "TA"),
    ("gap_forget_acc", "|dFA|"), ("avg_gap", "AvgGap"),
    ("mia_forget_vs_test", "MIA"), ("mia_efficacy", "MIA-Eff"),
    ("js_forget", "JS(f)"), ("agreement_forget", "Agree(f)"),
    ("seconds", "Time(s)"), ("rte_speedup_vs_retrain", "Speedup"),
    ("n_unlearned", "N"), ("n_diverged", "Diverged"),
]


def plots(results_dir, runs_by_name, retrain_runs, eps_by_name, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from unlearn.forget_quality import compute_logit_scaled_confidence
    from unlearn.metrics import losses_from_logits

    pdir = results_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    df = summary.sort_values("competition_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
    ax.barh(df["name"], df["competition_score"])
    ax.set_xlabel("Competition score  F x RA-ratio x TA-ratio")
    fig.tight_layout()
    fig.savefig(pdir / "competition_score.png", dpi=150)
    plt.close(fig)

    names = list(runs_by_name)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.hist(eps_by_name[name], bins=40)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("per-example epsilon")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(pdir / "epsilon_histograms.png", dpi=130)
    plt.close(fig)

    show = names + ["retrain (oracle)"]
    fig, axes = plt.subplots(int(np.ceil(len(show) / cols)), cols,
                             figsize=(4 * cols, 3 * int(np.ceil(len(show) / cols))), squeeze=False)
    for ax, name in zip(axes.flat, show):
        r = retrain_runs[0] if name == "retrain (oracle)" else runs_by_name[name][0]
        ax.hist(losses_from_logits(r["test_logits"], r["test_targets"]), bins=50, density=True,
                alpha=0.5, label="test")
        ax.hist(losses_from_logits(r["forget_logits"], r["forget_targets"]), bins=50,
                density=True, alpha=0.5, label="forget")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
    axes.flat[0].legend(frameon=False)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(pdir / "loss_histograms_run0.png", dpi=130)
    plt.close(fig)

    # logit-scaled confidence distributions over models, for 4 forget examples
    r_conf = np.stack([compute_logit_scaled_confidence(r["forget_logits"], r["forget_targets"])[1]
                       for r in retrain_runs])
    examples = np.linspace(0, r_conf.shape[1] - 1, 4).astype(int)
    for name in names:
        u_conf = np.stack([compute_logit_scaled_confidence(r["forget_logits"],
                                                           r["forget_targets"])[1]
                           for r in runs_by_name[name]])
        fig, axes = plt.subplots(1, 4, figsize=(14, 2.8))
        for ax, e in zip(axes, examples):
            ax.hist(r_conf[:, e], bins=20, alpha=0.5, label="retrain", density=True)
            ax.hist(u_conf[:, e], bins=20, alpha=0.5, label="unlearned", density=True)
            ax.set_title(f"forget ex {e}  eps={eps_by_name[name][e]:.2f}", fontsize=9)
        axes[0].legend(frameon=False)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(pdir / f"conf_{name}.png", dpi=110)
        plt.close(fig)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--orig", default="own", choices=["own", "challenge"])
    p.add_argument("--methods", nargs="*", default=None, help="default: every method found")
    p.add_argument("--min-runs", type=int, default=3)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    base = exp_dir(a.out, a.dataset, a.arch, a.version)
    retrain_runs = load_runs(base / "reference" / "retrain", "seed_*.npz")
    print(f"retrained oracle runs: {len(retrain_runs)}")
    r_per = [run_metrics(r, seed=i) for i, r in enumerate(retrain_runs)]

    runs_root = base / a.orig / "runs"
    results_dir = base / a.orig / "results"
    (results_dir / "eps").mkdir(parents=True, exist_ok=True)

    rows, per_run_rows, runs_by_name, eps_by_name = [], [], {}, {}
    for i, m in enumerate(r_per):
        per_run_rows.append({"name": "retrain", "method": "retrain", "tag": "-", "run": i, **m})

    for method_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        if a.methods and method_dir.name not in a.methods:
            continue
        for tag_dir in sorted(p for p in method_dir.glob("*") if p.is_dir()):
            runs = load_runs(tag_dir)
            if len(runs) < a.min_runs:
                print(f"[skip] {method_dir.name}/{tag_dir.name}: only {len(runs)} runs")
                continue
            name = method_dir.name if tag_dir.name == "default" else f"{method_dir.name}[{tag_dir.name}]"
            print(f"evaluating {name} ({len(runs)} runs)", flush=True)
            s, eps = summarize(runs, retrain_runs, retrained_per_run=r_per)
            info = runs[0]["meta"].get("info", {})
            rows.append({"name": name, "method": method_dir.name, "tag": tag_dir.name,
                         "rank": info.get("rank"), "team": info.get("team"),
                         "family": info.get("family"),
                         "hparams_override": json.dumps(runs[0]["meta"].get("hparams_override", {})),
                         **s})
            np.save(results_dir / "eps" / f"{method_dir.name}__{tag_dir.name}.npy", eps)
            for j, r in enumerate(runs):
                per_run_rows.append({"name": name, "method": method_dir.name, "tag": tag_dir.name,
                                     "run": j, **run_metrics(r, seed=j)})
            runs_by_name[name] = runs
            eps_by_name[name] = eps

    if not rows:
        print("no method runs found")
        return 1
    summary = pd.DataFrame(rows).sort_values("competition_score", ascending=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    summary.to_json(results_dir / "summary.json", orient="records", indent=1)
    pd.DataFrame(per_run_rows).to_csv(results_dir / "per_run.csv", index=False)

    table = summary[["name"] + [c for c, _ in MAIN_COLUMNS]].rename(columns=dict(MAIN_COLUMNS))
    ref = {"name": "retrain (oracle)", "RA": summary["retrain_retain_acc"].iloc[0],
           "FA": summary["retrain_forget_acc"].iloc[0], "TA": summary["retrain_test_acc"].iloc[0],
           "MIA": summary["retrain_mia_forget_vs_test"].iloc[0],
           "MIA-Eff": summary["retrain_mia_efficacy"].iloc[0],
           "Time(s)": summary["retrain_seconds"].iloc[0], "N": len(retrain_runs)}
    table = pd.concat([table, pd.DataFrame([ref])], ignore_index=True)
    with open(results_dir / "summary.md", "w") as f:
        f.write(f"# {a.dataset} / {a.arch} / {a.version} / original = {a.orig}\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not a.no_plots:
        plots(results_dir, runs_by_name, retrain_runs, eps_by_name, summary)
    print(f"\nresults in {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
