#!/usr/bin/env bash
# Candidate improvements, each given to BOTH model families so the comparison stays fair
# (docs/EXPERIMENTS.md, E4). Same budget as run_sweep.sh: ~171k parameters, 200 epochs, batch 100.
#   bash scripts/run_improve.sh <lr> <schedule> [seed]       e.g.  bash scripts/run_improve.sh 1e-3 cosine
# Use the (lr, schedule) the selection rule picked; each variant changes ONE thing from that base.
# Finished runs are skipped (train_quick reloads an existing checkpoint). Afterwards:
#   py -3.10 -m scripts.summarise_runs
set -u
if [ $# -lt 2 ]; then sed -n 2,8p "$0"; exit 1; fi
PY="${PY:-py -3.10}"
MATCH=171396
LR=$1; SCHED=$2; SEED=${3:-3001}
mkdir -p logs

run () {   # run <model> <label> <extra train_quick args...>
  local model=$1 label=$2; shift 2
  local name="${model}_samples_lr${LR}_${SCHED}_${label}_seed${SEED}"
  echo "$(date +%H:%M) start $name"
  $PY -u -m scripts.train_quick --model "$model" --input samples --lr "$LR" --schedule "$SCHED" \
      --epochs 200 --match-params $MATCH --seed "$SEED" "$@" > "logs/${name}.log" 2>&1
  echo "$(date +%H:%M) done  $name: $(tail -n 2 "logs/${name}.log" | head -n 1)"
}

# Variants outer, models inner: if time runs out, both families have finished the same variants.
# VARIANTS selects/reorders them, e.g.  VARIANTS="ff6" bash scripts/run_improve.sh 1e-3 cosine
for variant in ${VARIANTS:-ff6 cons ff6cons}; do
  case $variant in
    ff6)     extra="--fourier 6" ;;                      # Fourier time features (spectral bias)
    cons)    extra="--consistency 0.1" ;;                # qdot = dq/dt self-consistency term
    ff6cons) extra="--fourier 6 --consistency 0.1" ;;    # both
    *) echo "unknown variant $variant"; exit 1 ;;
  esac
  for model in deeponet mlp; do
    run "$model" "$variant" $extra
  done
done
echo "$(date +%H:%M) all done"
