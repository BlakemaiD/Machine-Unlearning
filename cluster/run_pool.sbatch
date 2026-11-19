#!/usr/bin/env bash
# One job = one whole node (8 GPUs). Runs every task of the given experiments
# with several tasks per GPU. Resumable: resubmit after the 24 h limit.
#
#   cd ~/Projects/MachineUnlearning
#   sbatch -J mu-smoke cluster/run_pool.sbatch smoke
#   sbatch -J mu-c10   cluster/run_pool.sbatch cifar10_resnet18
#   SLOTS=8 sbatch -J mu-arch cluster/run_pool.sbatch cifar10_mobilenet_v2 cifar10_efficientnet_b0
#
# Pin a node with -w g3 (the other node stays free for other users).
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:rtx_pro_6000:8
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -t 24:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
PROJ=${PROJ:-$HOME/Projects/MachineUnlearning}
SLOTS=${SLOTS:-7}            # tasks per GPU (8 GPUs x 7 = 56 concurrent runs)
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearn

TAG=$(IFS=+; echo "$*")
START=$(date +%Y-%m-%d_%H%M)   # run id = <version>_<job start>, e.g. v1_2026-11-09_1931
TASKS=tasks/$TAG
echo "node $(hostname)  experiments: $*  slots/gpu: $SLOTS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

CUDA_VISIBLE_DEVICES=0 python scripts/check_metric_port.py
python scripts/make_tasks.py "$@" --out-dir "$TASKS"
python scripts/pool.py "$TASKS/1_reference.txt" "$TASKS/2_unlearn.txt" \
    --slots-per-gpu "$SLOTS" --log-dir "$TASKS/logs" || echo "some tasks failed (see $TASKS/logs/status.tsv)"

# Evaluate everything that finished (CPU + one GPU for the epsilon metric).
python - "$@" <<'PY'
import subprocess, sys, yaml
cfg = yaml.safe_load(open("configs/experiments.yaml"))
procs = []
for name in sys.argv[1:]:
    e = {**cfg["defaults"], **cfg["experiments"][name]}
    for orig in e["origs"]:
        cmd = ["python", "scripts/evaluate.py", "--dataset", e["dataset"], "--arch", e["arch"],
               "--orig", orig, "--version", str(e["version"])]
        if e.get("out"): cmd += ["--out", e["out"]]
        if e.get("data_root"): cmd += ["--data-root", e["data_root"]]
        print(" ".join(cmd), flush=True)
        log = open(f"tasks/{name}_eval_{orig}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
for p_ in procs:  # evaluations of different originals run in parallel
    p_.wait()
PY

# Gather tables / plots / manifest into results/<version>_<start>/ (small; synced back).
VERSION=$(python -c "import yaml,sys; c=yaml.safe_load(open('configs/experiments.yaml')); \
print({**c['defaults'], **c['experiments'][sys.argv[1]]}['version'])" "$1")
PREFIX=""; case "$1" in smoke*) PREFIX="smoke_" ;; esac   # smoke checks are labelled as such
python scripts/collect_results.py --run-id "${PREFIX}${VERSION}_${START}" --experiments "$@" \
    --job "${SLURM_JOB_ID:-none}"
