#!/usr/bin/env bash
# p sweep at FIXED width 128 (no --match-params): parameters grow with p. DeepONet, coef input,
# lr 3e-3 cosine, seed 3001, 200 epochs. Companion to the equal-budget sweep in run_ff6_psweep.sh.
set -u
PY="${PY:-py -3.10}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
for p in 8 16 32 64 128 256; do
  $PY -u -m scripts.train_quick --model deeponet --input coef --lr 3e-3 --schedule cosine \
      --epochs 200 --width 128 --p $p --seed 3001 > "logs/psweep_w128_p$p.log" 2>&1 &
done
wait; echo all done
