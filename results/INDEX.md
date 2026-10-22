# What is in `results/`

Grouped by the report section each file supports (section numbers follow `docs/RESULTS_SUMMARY.md`).
Figure numbers (#) are those in `docs/FIGURE_PLAN.md`. The script that writes each file is in brackets.

## 2. Inputs, outputs and data  [`scripts/data_figures.py`, `scripts/smooth_family_check.py`]

| File | What it shows |
|---|---|
| `data_trajectories.png` | #5 sample trajectories, train vs OOD |
| `data_channel_hist.png` | #6 channel histograms (why per-channel z-score) |
| `data_param_coverage.png` | #7 parameter coverage of the splits (what OOD means) |
| `data_peak_vs_inputs.png` | #8 peak \|q2\| vs inputs, regime switching (sine family) |
| `smooth_family_check.png` | the smooth torque family is effectively 24 numbers |
| `torque_family_prototype.png` | working figure from choosing the torque family [`scripts/torque_family_prototype.py`] |

## 4. Validation: solver  [`scripts/solver_figures.py`]

| File | What it shows |
|---|---|
| `solver_checks.md` | all solver-check numbers (order, energy drift, RK4 vs DOP853) |
| `solver_convergence.png` | #2 RK4 order 4 |
| `solver_energy.png` | #3 energy conservation (a) and energy balance (b) |
| `solver_vs_dop853.png` | #4 stored labels vs DOP853 |

## 4. Validation: code

| File | What it shows |
|---|---|
| `test_report.xml` | pytest results (table c) [`pytest --junitxml=results/test_report.xml`] |
| `mutation_check.md` | 10 deliberate bugs and which tests caught them (table d) [`scripts/mutation_check.py`] |

## 4. Validation: surrogate on test and OOD  [`scripts/final_eval.py`, `scripts/plot_eval.py`, `scripts/final_analysis.py`]

| File | What it shows |
|---|---|
| `final_eval.md` / `.csv` | **headline table**: rule-selected four models, 3 seeds, test + OOD |
| `final_analysis.md` | every follow-up table: initial condition, naive baselines, consistency, OOD, sensitivity, E4, speed |
| `final_overview.png` | #17 four-model comparison |
| `evalfig_test_final4_cases.png` | #10 best / median / worst test cases |
| `evalfig_test_final4_time.png` | #11 error vs time per channel |
| `evalfig_test_final4_cdf.png` | #12 error CDF |
| `evalfig_test_final4_regime.png`, `_inputs.png` | #13 error by regime and by input |
| `evalfig_ood_amp1_final4_*.png`, `evalfig_ood_smooth_final4_*.png` | the same five plots on the two harder OOD sets |
| `final_consistency.png` | #14 qdot vs dq/dt and energy-balance excess |
| `final_ood_degradation.png` | #15 OOD degradation curve |
| `final_seeds.png` | #18 seed variance |
| `final_sensitivity.png` | extra: input sensitivity of the true map vs surrogate error |
| `final_persample.npz` | per-sample errors behind the figures (not committed) |

## 5. Comparison and optimisation

| File | What it shows |
|---|---|
| `runs.csv` | every training run, one row each, validation metrics [`scripts/summarise_runs.py`] |
| `runs_agg.csv` | the same aggregated over seeds (mean +- std) |
| `analysis_seeds.md` | gap between models vs seed noise (validation) [`scripts/analyse_seeds.py`] |
| `e4_variants.png`, `e4_datasize.png`, `e4_psweep.png` | E4: Fourier features / penalty, data size, p sweep |
| `final_eval_ff6.md` / `.csv`, `final_eval_ff6cons.md` / `.csv` | E4 variants on test, evaluated **after** the selection rule (follow-up, not headline) |
| `benchmark.md` / `.csv` / `.json` | speed, FLOPs, memory, break-even (EXPERIMENTS B2) [`scripts/run_benchmark.py`] |
| `bench_pareto.png`, `bench_batch.png`, `bench_breakeven.png` | #16 the speed figures [`scripts/final_analysis.py`] |

## Training curves  [`scripts/plot_training.py`]

`training_curves/`: one `loss_<run>.png` per checkpoint, plus `loss_by_model.png` (#9, the one to use in the
report) and `loss_all_runs.png`.

## Working files (development history, not for the report)

| File | What it was |
|---|---|
| `evalfig_val_E1_E2_*.png`, `evalfig_val_E2_vs_coef_*.png` | validation figures used while running E1 / E2 |
| `eval_deeponet_*_val.npz` | cached validation predictions from E1 (not committed) |
