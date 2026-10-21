#!/usr/bin/env bash
# E4 follow-up (AI-written glue): Fourier time features + qdot=dq/dt consistency term (weight 0.1),
# samples input, lr 1e-3 cosine, both families, seeds 3001 (reloaded if present), 1, 2.
# 4 jobs in parallel (seeds 1 and 2, both families), 4 torch threads each.  PY=python bash scripts/run_cons_seeds.sh
set -u
PY="${PY:-py -3.10}"; M=171396
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
for s in 3001 1 2; do
  for m in deeponet mlp; do
    $PY -u -m scripts.train_quick --model $m --input samples --lr 1e-3 --schedule cosine --epochs 200 \
        --match-params $M --fourier 6 --consistency 0.1 --seed $s > "logs/ffcons_${m}_samples_s$s.log" 2>&1 &
  done
done
wait
echo all done
