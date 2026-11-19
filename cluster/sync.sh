#!/usr/bin/env bash
# Copy the code (not data / artifacts) to the cluster, or results back.
#   bash cluster/sync.sh push            # local -> cluster
#   bash cluster/sync.sh pull-results    # cluster results/<run-id>/ folders -> local results/
#   bash cluster/sync.sh pull-all        # also every run .npz (no model .pt files)
set -euo pipefail
REMOTE=${REMOTE:-head0}
RDIR=${RDIR:-Projects/MachineUnlearning}
cd "$(dirname "$0")/.."
case "${1:-push}" in
  push)
    rsync -av --exclude '.git' --exclude 'data/' --exclude 'artifacts*/' --exclude 'tasks/' \
      --exclude '__pycache__' --exclude 'cluster/logs/' --exclude 'results/' --exclude '*.pth' ./ "$REMOTE:$RDIR/"
    ssh "$REMOTE" "mkdir -p $RDIR/cluster/logs" ;;
  pull-results)
    # results/<version>_<date>_<time>/ folders made by scripts/collect_results.py
    rsync -av "$REMOTE:$RDIR/results/" results/ ;;
  pull-all)
    rsync -av --exclude '*.pt' "$REMOTE:$RDIR/artifacts/" artifacts/ ;;
  *) echo "usage: $0 push|pull-results|pull-all"; exit 1 ;;
esac
