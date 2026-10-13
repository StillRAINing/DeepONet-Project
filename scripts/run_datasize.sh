#!/usr/bin/env bash
# Data-size learning curve (docs/EXPERIMENTS.md, "Not yet run"): does the DeepONet's relative
# standing change with fewer training trajectories? Every model sees the SAME first n samples.
#   bash scripts/run_datasize.sh <lr> <schedule> [seed]
# n = 7000 is the full-data run the sweep already made (reloaded, not retrained).
# Epochs are kept at 200, so smaller n means fewer gradient steps; note that when reading the curve.
set -u
if [ $# -lt 2 ]; then sed -n 2,8p "$0"; exit 1; fi
PY="${PY:-py -3.10}"
MATCH=171396
LR=$1; SCHED=$2; SEED=${3:-3001}
mkdir -p logs
for n in 500 1000 2000 3500 7000; do
  for cfg in "deeponet samples" "deeponet coef" "mlp samples" "mlp coef"; do
    set -- $cfg
    name="${1}_${2}_lr${LR}_${SCHED}_n${n}_seed${SEED}"
    echo "$(date +%H:%M) start $name"
    $PY -u -m scripts.train_quick --model "$1" --input "$2" --lr "$LR" --schedule "$SCHED" \
        --epochs 200 --match-params $MATCH --n-train "$n" --seed "$SEED" > "logs/${name}.log" 2>&1
    echo "$(date +%H:%M) done  $name: $(tail -n 2 "logs/${name}.log" | head -n 1)"
  done
done
echo "$(date +%H:%M) all done"
