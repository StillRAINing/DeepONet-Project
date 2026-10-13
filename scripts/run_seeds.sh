#!/usr/bin/env bash
# Step 2 of the selection rule (docs/EXPERIMENTS.md): re-train ONE chosen setting with extra seeds.
#   bash scripts/run_seeds.sh <model> <input> <lr> <schedule> [seeds...]
#   bash scripts/run_seeds.sh deeponet samples 1e-3 cosine            # seeds 1 and 2 (default)
#   bash scripts/run_seeds.sh mlp coef 3e-3 none 1 2 3                 # any seed list
# Same budget as run_sweep.sh (200 epochs, --match-params 171396). An existing checkpoint is
# reloaded, not retrained: seed 3001 runs from before 2026-10-13 15:30 (unseeded init, see
# EXPERIMENTS.md reproducibility note) are only redone if you delete them or call train_quick with
# --retrain yourself. Afterwards:  py -3.10 -m scripts.summarise_runs
set -u
if [ $# -lt 4 ]; then sed -n 2,8p "$0"; exit 1; fi
PY="${PY:-py -3.10}"
MATCH=171396
MODEL=$1; INPUT=$2; LR=$3; SCHED=$4; shift 4
SEEDS="${*:-1 2}"
mkdir -p logs
for seed in $SEEDS; do
  name="${MODEL}_${INPUT}_lr${LR}_${SCHED}_seed${seed}"
  echo "$(date +%H:%M) start $name"
  $PY -u -m scripts.train_quick --model "$MODEL" --input "$INPUT" --lr "$LR" --schedule "$SCHED" \
      --epochs 200 --match-params $MATCH --seed "$seed" > "logs/${name}.log" 2>&1
  echo "$(date +%H:%M) done  $name: $(tail -n 2 "logs/${name}.log" | head -n 1)"
done
