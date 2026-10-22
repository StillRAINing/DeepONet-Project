# Figure and table plan for the report

Each figure should be explainable by you without help. "Status" shows what exists today. Rule: every
figure states its split (train / val / test / OOD), its metric definition, and what a reader should
conclude. Never show a mean without a spread or worst case.

Status updated 2026-10-15: every figure exists. "AI-made" means the plotting code was written by the AI
(AI_LOG rows 37 to 41); you still have to be able to explain every curve. Numbers behind each figure:
`results/final_analysis.md`, `results/solver_checks.md`, `results/benchmark.md`.

| # | Figure / table | Purpose | File | Status | You must be able to explain |
|---|---|---|---|---|---|
| 1 | Arm schematic with conventions (`q1` from horizontal, `q2` relative, payload at tip) | Define the problem | `src/viz.py` screenshot (`P` key) or hand sketch | **to do (you)** | the conventions, and why gravity enters as `cos q1` |
| 2 | RK4 convergence: error vs `h`, log-log, slope 4 | Trust the ground truth | `results/solver_convergence.png` (`scripts/solver_figures.py`) | done (AI-made) | local vs global error; why the violent sample is not asymptotic at large h and shows order ~5 at mid h |
| 3 | Energy conservation (`b = 0`, `tau = 0`) and energy balance with torque/friction | Trust the physics | `results/solver_energy.png` | done (AI-made) | why raw energy is not conserved with friction; why the 20 ms residual is larger (quadrature, not RK4) |
| 4 | RK4 labels vs DOP853 on a calm, a violent test and a violent OOD sample | Cross-check, show where truth degrades | `results/solver_vs_dop853.png` | done (AI-made) | why the error grows after the elbow starts spinning |
| 5 | Sample trajectories, train vs OOD | Show the task difficulty | `results/data_trajectories.png` | done | symlog axis, why OOD looks calmer |
| 6 | Channel histograms | Justify per-channel normalisation | `results/data_channel_hist.png` | done | heavy tails, units differ |
| 7 | Parameter coverage of the splits | Define OOD | `results/data_param_coverage.png` | done | what OOD does and does not test |
| 8 | Peak `|q2|` vs `A2` and `b` | Show regime switching / hard cases | `results/data_peak_vs_inputs.png` | done (sine family; still valid as motivation) | bimodality (`REVIEW_NOTES` 2.5) |
| 9 | Training and validation loss curves | Convergence and overfitting | `results/training_curves/loss_by_model.png`, `results/training_curves/loss_<run>.png` | done | where train and val diverge, what you selected on |
| 10 | Best / median / worst test cases, all four channels | Make accuracy tangible | `results/evalfig_test_final4_cases.png` (MLP coef; also `_ood_amp1_`, `_ood_smooth_`) | done | why the worst case fails (elbow flip predicted wrongly) |
| 11 | Error vs time, per channel, with percentile band | Where it goes wrong | `results/evalfig_test_final4_time.png` | done | definition (divided by the whole-trajectory rms), why it grows |
| 12 | Error CDF over test (and OOD) | Spread, not just mean | `results/evalfig_test_final4_cdf.png`, `_ood_amp1_`, `_ood_smooth_` | done | which fraction exceeds a tolerance |
| 13 | Error by regime and by inputs | Explain failures | `results/evalfig_test_final4_regime.png`, `_inputs.png` | done | link to the spinning regime |
| 14 | Physical consistency (`qdot` vs `dq/dt`) and energy-balance excess | Does the output obey physics | `results/final_consistency.png` | done (AI-made) | why neither is in the data loss; what the penalty term does |
| 15 | OOD degradation: amplitude curve, matched m_p/b comparison, rougher torque by input type | Honest extrapolation | `results/final_ood_degradation.png` | done (AI-made) | why it is a degradation curve; why m_p/b OOD is easy |
| 16 | Speed: accuracy vs cost, batch-size sweep, break-even | The speed claim | `results/bench_pareto.png`, `bench_batch.png`, `bench_breakeven.png`; table `results/benchmark.md` | done (AI-made) | why batched RK4 is the fair rival; why the cached trunk is a real DeepONet advantage |
| 17 | Four-model comparison at equal tuning budget | Fair baseline | `results/final_overview.png` + `evalfig_test_final4_cdf.png` | done | why compact input is included |
| 18 | Seed variance | Reproducibility | `results/final_seeds.png` | done (AI-made) | why one run is an anecdote |
| extra | Input sensitivity of the true map vs surrogate error | Why the tail is hard | `results/final_sensitivity.png` | done (AI-made) | correlation, not proof |
| extra | E4: Fourier features / consistency penalty; data size; p sweep | Fair-fight improvements, W2 | `results/e4_variants.png`, `e4_datasize.png`, `e4_psweep.png` | done (AI-made) | which differences are beyond seed noise |

Colour convention: MLP coef `#0072B2`, MLP samples `#56B4E9`, DeepONet coef `#D55E00`, DeepONet samples
`#E69F00` (colour-blind safe, Okabe-Ito); data figures use train `#0072B2`, OOD `#D55E00`, val `#009E73`,
test `#CC79A7`.

Tables: (a) parameter ranges (README), (b) results (`results/final_eval.md`, `final_analysis.md`),
(c) test report (`results/test_report.xml`), (d) mutation check (`results/mutation_check.md`), (e) AI-use
summary from `AI_LOG.md`.
