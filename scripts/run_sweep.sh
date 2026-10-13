#!/usr/bin/env bash
# Fair comparison sweep (run from the repo root:  bash scripts/run_sweep.sh).
# Every model gets the SAME budget: ~171k parameters (--match-params, DESIGN_LOG W1), the same
# lr grid, 200 epochs, batch 100, seed 3001, Adam, train-only normalisation, val-only selection.
# Finished runs are skipped (train_quick reloads an existing checkpoint), so the script can be resumed.
# Logs go to logs/, one per run. Afterwards:  py -3.10 -m scripts.summarise_runs
set -u
PY="${PY:-py -3.10}"
MATCH=171396   # parameter count of the reference DeepONet (samples input, p 64, width 128)
mkdir -p logs

run () {   # run <model> <input> <lr> <schedule>
  local name="${1}_${2}_lr${3}_${4}"
  echo "$(date +%H:%M) start $name"
  $PY -u -m scripts.train_quick --model "$1" --input "$2" --lr "$3" --schedule "$4" \
      --epochs 200 --match-params $MATCH > "logs/${name}.log" 2>&1
  echo "$(date +%H:%M) done  $name: $(tail -n 2 "logs/${name}.log" | head -n 1)"
}

# 1. Cosine schedule on the reference DeepONet (compare with the constant-lr run already logged).
run deeponet samples 1e-3 cosine

# 2. Baselines, constant lr, same lr grid as the DeepONet had.
for lr in 1e-3 3e-4 3e-3; do
  for cfg in "deeponet coef" "mlp samples" "mlp coef"; do
    set -- $cfg; run "$1" "$2" "$lr" none
  done
done

# 3. Cosine schedule for every model, at the two higher starting lrs.
for lr in 1e-3 3e-3; do
  for cfg in "deeponet samples" "deeponet coef" "mlp samples" "mlp coef"; do
    set -- $cfg; run "$1" "$2" "$lr" cosine
  done
done
echo "$(date +%H:%M) all done"
