#!/usr/bin/env bash
# E4 follow-up (AI-written glue): (a) Fourier time features (--fourier 6) with 3 seeds for both
# families and both inputs, at each model's selected lr; (b) DeepONet p sweep at a fixed parameter
# budget (--match-params, so width shrinks as p grows), coef input, selected lr, seed 3001.
# Runs everything in parallel, one torch thread each, so train_s is NOT comparable with earlier runs.
#   PY=python bash scripts/run_ff6_psweep.sh ; then  python -m scripts.summarise_runs
set -u
PY="${PY:-py -3.10}"; M=171396
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
go () { local name=$1; shift
  $PY -u -m scripts.train_quick --epochs 200 --match-params $M "$@" > "logs/${name}.log" 2>&1 & }
for s in 3001 1 2; do
  for m in deeponet mlp; do
    go ff6_${m}_samples_s$s --model $m --input samples --lr 1e-3 --schedule cosine --fourier 6 --seed $s
    go ff6_${m}_coef_s$s    --model $m --input coef    --lr 3e-3 --schedule cosine --fourier 6 --seed $s
  done
done
for p in 8 16 32 128 256; do
  go psweep_p$p --model deeponet --input coef --lr 3e-3 --schedule cosine --p $p --seed 3001
done
wait; echo all done
