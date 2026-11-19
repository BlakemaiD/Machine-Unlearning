#!/usr/bin/env bash
# Pipeline check + recipe calibration, in one single-node job.
#   cd ~/Projects/MachineUnlearning && sbatch -w g4 -J mu-smoke cluster/smoke_calibrate.sbatch
#
# 1. metric-port check against the official metric.py
# 2. calibration: own original + 2 retrained ResNet-18s with the FULL recipe,
#    written to artifacts/ (reused by the main cifar10_resnet18 run)
# 3. smoke: tiny 2-epoch reference models, every method x 3 seeds x both
#    originals, then evaluate.py -> artifacts_smoke/.../results
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:rtx_pro_6000:8
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -t 04:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
PROJ=${PROJ:-$HOME/Projects/MachineUnlearning}
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearn
echo "node $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

CUDA_VISIBLE_DEVICES=0 python scripts/check_metric_port.py

T=tasks/smoke_calibrate
python scripts/make_tasks.py smoke --out-dir "$T"
python scripts/make_tasks.py cifar10_resnet18 --out-dir "$T/main"
# calibration = the first 3 reference tasks of the main run (original + retrain 0, 1)
head -n 3 "$T/main/1_reference.txt" > "$T/0_calibration_and_smoke_reference.txt"
cat "$T/1_reference.txt" >> "$T/0_calibration_and_smoke_reference.txt"

python scripts/pool.py "$T/0_calibration_and_smoke_reference.txt" "$T/2_unlearn.txt" \
    --slots-per-gpu 7 --log-dir "$T/logs" || echo "some tasks failed (see $T/logs/status.tsv)"

for orig in own challenge; do
  python scripts/evaluate.py --out artifacts_smoke --orig $orig || true
done
echo "=== calibration models (full recipe) ==="
grep -h '"role"' "$T"/logs/0_calibration_and_smoke_reference/0000[0-2].log || true
