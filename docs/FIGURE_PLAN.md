# Figure and table plan for the report

Each figure should be explainable by you without help. "Status" shows what exists today. Rule: every
figure states its split (train / val / test / OOD), its metric definition, and what a reader should
conclude. Never show a mean without a spread or worst case.

| # | Figure / table | Purpose | Produced by | Status | You must be able to explain |
|---|---|---|---|---|---|
| 1 | Arm schematic with conventions (`q1` from horizontal, `q2` relative, payload at tip) | Define the problem | `src/viz.py` screenshot (`P` key) or hand sketch | viz exists | the conventions, and why gravity enters as `cos q1` |
| 2 | RK4 convergence: error vs `h`, log-log, slope 4 | Trust the ground truth | **you** (`tests/test_solvers.py`) | not started | local vs global error, why the slope flattens at tiny `h` |
| 3 | Energy conservation (`b = 0`, `tau = 0`) and energy balance with torque/friction | Trust the physics | **you** | not started | why raw energy is not conserved with friction |
| 4 | RK4 vs `solve_ivp` agreement on a mild and a spinning sample | Cross-check, show where truth degrades | **you** | not started | the unconverged tail (`REVIEW_NOTES` 2.2) |
| 5 | Sample trajectories, train vs OOD | Show the task difficulty | `scripts/data_figures.py` -> `results/data_trajectories.png` | done | symlog axis, why OOD looks calmer |
| 6 | Channel histograms | Justify per-channel normalisation | `results/data_channel_hist.png` | done | heavy tails, units differ |
| 7 | Parameter coverage of the splits | Define OOD | `results/data_param_coverage.png` | done | what OOD does and does not test |
| 8 | Peak `|q2|` vs `A2` and `b` | Show regime switching / hard cases | `results/data_peak_vs_inputs.png` | done | bimodality (`REVIEW_NOTES` 2.5) |
| 9 | Training and validation loss curves (log y) for every model | Show convergence and overfitting | **you** (`train.py`) | not started | where train and val diverge, what you selected on |
| 10 | Predicted vs true trajectories: best, median, worst test cases (all four channels) | Make accuracy tangible | `evaluate.py` + plotting helper | not started | why the worst case fails |
| 11 | Relative L2 vs time, per channel, with percentile band | Where it goes wrong | **you** (`evaluate.py`) | not started | definition of the error, why it grows |
| 12 | Error histogram / CDF over test trajectories (and OOD) | Spread, not just mean | **you** | not started | which fraction exceeds a tolerance |
| 13 | Error split by regime (flipped vs not) and by `A`, `b` | Explain failures | **you** | not started | link to the bimodality |
| 14 | Physical consistency: `qdot` vs `d q/dt`; energy-balance residual | Does the output obey physics | **you** | not started | why neither is enforced by the loss |
| 15 | OOD degradation curve: error vs distance beyond range (`m_p`, `b`, and `A`/`f`) | Honest extrapolation | **you** | not started | why this is a degradation curve, not a pass/fail |
| 16 | Speed table: scalar RK4, vectorised RK4, `solve_ivp` (matched tolerance), MLP, DeepONet; params, FLOPs, memory | The speed claim | `src/benchmark.py` | scaffold done (initial numbers in `REVIEW_NOTES` 5) | what each row's tolerance is, break-even queries incl. data + training |
| 17 | Four-model comparison (MLP 304-d, MLP compact, DeepONet 304-d, DeepONet compact) at equal tuning budget | Fair baseline | **you** | not started | why compact input is included |
| 18 | Seed variance: mean and std of test error over >= 3 seeds | Reproducibility | **you** | not started | why one run is an anecdote |

Colour convention used so far: train `#0072B2`, OOD `#D55E00`, val `#009E73`, test `#CC79A7` (colour-blind safe).

Tables: (a) parameter ranges (README), (b) results (README planned table, extended with p95/max error),
(c) test report (`pytest --junitxml=results/test_report.xml`), (d) AI-use summary from `AI_LOG.md`.
