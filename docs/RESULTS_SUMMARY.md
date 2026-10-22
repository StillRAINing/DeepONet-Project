# Results summary: the evidence for each report section

Written by the AI on 2026-10-15 (AI_LOG row 42) after all runs, the benchmark and the final analysis were
finished. It is a **map**, not report text: each claim points to the number, the file it comes from and the
figure that shows it. Check every number against its file before you use it, and write the report in your own
words. Sections follow the brief's "Written Project Documentation" list (5 to 10 pages).

Headline metric everywhere: per-sample relative L2 over (time, channel), channels scaled by the train std.
Test n = 1500, OOD sets n = 200 each. "3 seeds" = mean +- std over training seeds 3001, 1, 2.

## The story in five sentences (draft facts, not prose)

1. At equal parameters (171k) and equal tuning, a plain MLP beats the DeepONet in distribution: test median
   0.071 vs 0.109, beyond seed noise (gaps 8 to 12 seed std on validation).
2. The DeepONet is more physically self-consistent (qdot vs dq/dt 0.10 to 0.15 vs 0.22 to 0.26) and far cheaper
   per query once its trunk is cached (0.02 ms vs 0.27 ms at batch 64); a consistency penalty gives both
   families consistency without costing accuracy.
3. Every model fails on the same 5 to 10 % of trajectories, where the elbow spins: p95 0.60 to 0.65, close to
   the "predict the mean trajectory" baseline (0.87). The true map is itself highly sensitive there (a 1 %
   input change moves those trajectories by 4 to 8 %, Spearman +0.81 with the error).
4. Out of range: torque amplitude beyond 6 N m breaks every model (median 0.53 to 0.62, then saturating);
   heavier payload and friction are a mild shift; a rougher torque family is handled better when the torque
   is given as samples than as coefficients (the one place the "function input" pays off).
5. The surrogate pays for its training after about 70k to 205k queries against `solve_ivp`, but only after
   about 0.8 to 3.0 million against an exact, batched NumPy RK4, which is the honest rival (B2 quiet rerun).

## 1. Engineering problem

| Claim | Evidence |
|---|---|
| Context, intended use (fast sweeps over payload / friction / torque profiles), scope limits | `README.md` Purpose, Scope limits |
| Why it matters: one ODE solve per query; design studies need thousands to millions | benchmark (section 5) gives the actual numbers |
| Scope: planar 2-link, viscous friction, fixed x0 and 3 s horizon, simulation only | README Scope, CLAUDE.md weak point 7 |

Figure to make yourself: arm schematic (FIGURE_PLAN #1).

## 2. Inputs, outputs and data

| Claim | Number | Source | Figure |
|---|---|---|---|
| Inputs m_p in [0, 1] kg, b in [0.05, 0.5] N m s/rad, torque = smooth random function, peak in [0, 6] N m per joint; output [q1, q2, q1dot, q2dot] on 151 times | | README table, `src/data.py` | `results/data_param_coverage.png` |
| The torque is effectively 24 numbers (6 modes x 2 x 2), fed as 302 samples or as coefficients | | README, DESIGN_LOG A6 | `results/smooth_family_check.png` |
| Dataset: 7000 / 1500 / 1500 + four OOD sets of 200, seed 3001 | | `src/data.py` | `results/data_trajectories.png` |
| Channels differ in scale and have heavy tails -> per-channel z-score from train only | | DESIGN_LOG A8 | `results/data_channel_hist.png` |
| Invalid / unsupported inputs: ValueError vs OutOfRangeWarning | | README table, `src/validate.py`, `tests/test_validate.py` | |
| Label accuracy: stored RK4 vs DOP853 median 1e-8 on every split | | EXPERIMENTS 'Truth check', `results/solver_checks.md` #4 | `results/solver_vs_dop853.png` |

## 3. Computational approach

| Claim | Evidence |
|---|---|
| RK4 (fixed step, 2.5 ms, 8 substeps per output step) generates the truth; why fixed step | `src/solvers.py`, DESIGN_LOG |
| DeepONet: branch(u) . trunk(t) per channel, p = 64, 171k parameters | `src/model.py`, `tests/test_model.py::test_einsum_equals_explicit_loop` |
| Loss: per-sample relative L2 on channel-scaled data (every trajectory counts equally) | `src/train.py`, DESIGN_LOG A8 |
| Training: Adam, batch 100, 200 epochs, cosine lr, best-validation checkpoint | EXPERIMENTS "Fixed protocol" |
| p = 64 is on a plateau (16 to 128 equivalent; 8 clearly worse) | `results/final_analysis.md` p table, `results/e4_psweep.png` |

## 4. Validation method and results

What was validated, how, and what it cannot show (the brief asks for all three):

| Level | Method | Result | Source / figure |
|---|---|---|---|
| Solver | order on y' = -y and on the arm vs DOP853 | order 4.00 to 4.12 (scalar); approaches 4 on a calm arm sample | `results/solver_checks.md`, `results/solver_convergence.png` |
| Solver | energy conservation (b = 0, tau = 0) | drift 4.3e-5 J at the dataset step (relative 2e-6) | `results/solver_energy.png` (a) |
| Solver | energy balance with torque and friction | residual 1.2e-3 J (2.5 ms grid) | `results/solver_energy.png` (b) |
| Code | 115 unit tests; 10 deliberate bugs | `results/test_report.xml`; `results/mutation_check.md` | |
| Surrogate | withheld test set, used once after a pre-written selection rule | table below | `results/final_eval.md`, `results/final_overview.png` |
| Surrogate | physical constraints: qdot = dq/dt, energy balance | DeepONet more consistent; energy excess follows accuracy | `results/final_consistency.png` |
| Surrogate | out-of-range robustness (4 OOD sets) | degradation curve | `results/final_ood_degradation.png` |
| Surrogate | repeatability: 3 seeds | seed std of test median 0.000 to 0.007 | `results/final_seeds.png` |
| Surrogate | naive yardsticks | mean trajectory 0.43, 1-NN 0.42 (test median) | `results/final_analysis.md` |

Test results, rule-selected models, 3 seeds (`results/final_eval.md`):

| Model | Input | median | p95 | under 10 % | qdot vs dq/dt |
|---|---|---|---|---|---|
| MLP | coef | **0.071 +- 0.000** | **0.600** | **58.2 %** | 0.244 |
| MLP | samples | 0.080 +- 0.001 | 0.603 | 55.8 % | 0.223 |
| DeepONet | coef | 0.109 +- 0.001 | 0.653 | 47.3 % | **0.109** |
| DeepONet | samples | 0.130 +- 0.007 | 0.630 | 39.3 % | 0.147 |

Where the error lives (`results/final_analysis.md`, `evalfig_test_final4_*.png`): by regime, median 0.05 to
0.09 when peak |q| < pi (67 % of test) vs 0.37 to 0.54 above 2 pi; by time, the median grows ~4x from 1 s to 3 s;
by channel, q1 0.03 vs q2dot 0.19 to 0.35.

Did validation influence selection? Yes, by design: validation loss picked lr and schedule per model; the
test split was evaluated once afterwards (EXPERIMENTS "Selection rule", written before results). Two E4 variants
were evaluated on test **after** that, as reported follow-ups (`final_eval_ff6*.md`); say so.

What the validation cannot establish: agreement with a physical arm; anything beyond 3 s or other initial
conditions; behaviour on other torque families beyond the one rougher set; statistics beyond 3 seeds and one
data split.

## 5. Comparison and optimisation

| Claim | Number | Source / figure |
|---|---|---|
| Baselines: MLP with the same inputs, MLP and DeepONet with compact coefficients, equal parameters and equal lr grid | 4 models x 5 settings | EXPERIMENTS E3 |
| MLP wins in distribution, beyond seed noise | gap / seed std 7.7 to 12 (val) | `results/analysis_seeds.md` |
| Fourier time features: help sample-input models, not compact ones; gap not closed | DeepONet samples test 0.130 -> 0.105 | `results/e4_variants.png`, EXPERIMENTS E4 |
| Consistency penalty: consistency 0.071 -> 0.008 (DeepONet), 0.30 -> 0.12 (MLP), no accuracy cost | | same |
| Best configuration found: MLP, samples, ff6 + cons | test median 0.066 +- 0.002 | `results/final_eval_ff6cons.md` |
| More data helps the median as ~n^-0.5, the tail much less | | `results/e4_datasize.png` |
| Speed (B2 quiet rerun): surrogates 0.14 to 0.28 ms / traj (batch 64), DeepONet cached 0.02 ms; vectorised RK4 4.4 ms (0.73 ms at batch 1500); `solve_ivp` at matched error ~7.5 to 8 ms and fails at loose rtol | | `results/benchmark.md`, `results/bench_pareto.png`, `bench_batch.png` |
| Profile: the scalar solver spends its time in Python calls to `state_derivative` (4800 per trajectory) -> vectorising RK4 over the batch gives a 27x speed-up with identical labels | 121 ms -> 4.4 ms | `results/benchmark.md` (cProfile section) |
| FLOPs per trajectory: MLP 5.2e7, DeepONet 2.0 to 2.5e7, cached 2.6e5 to 2.9e5; parameters 0.69 MB | | `results/benchmark.md` |
| Break-even incl. data + training | 70k to 205k vs `solve_ivp`; 0.8M to 3.0M vs batched RK4; ~5x more with the lr grid | `results/final_analysis.md` "Speed", `results/bench_breakeven.png` |

## 6. Limitations and lessons learned

Evidence-backed limitations (README Limitations has the full list):
- The spinning tail is not learnt (p95 ~0.6, worst cases 1.4 to 2.1); the map is highly sensitive there
  (`final_sensitivity.png`).
- No extrapolation in amplitude; the m_p/b OOD set is a mild shift, not a hard test.
- The torque input is low-dimensional (24 numbers): the operator framing is only partly earned; the
  ood_smooth result is the evidence for sampled input.
- Initial condition not enforced: |x_pred(0)| 1 to 6 % of the velocity scale (`final_analysis.md`).
- One split, 3 seeds; size and p sweeps single-seed; timings on one laptop with background load.

Lessons (for you to phrase): equal-effort baselines change the conclusion; physical consistency and accuracy
are different axes; the fair speed comparison is against the best classical method, not the slowest.

## 7. AI-use reflection

Material: `AI_LOG.md` rows 1 to 42, especially the "Errors" column (rows 37 to 41 list the overclaims the AI
made and corrected today). The "Checked how" and "My decision" columns are yours to fill before you write this.

## 8. References

Lu et al. 2021 (DeepONet, Nature Machine Intelligence; arXiv 1910.03193); SciPy `solve_ivp` (Dormand-Prince
RK45, DOP853); Spong / Siciliano for the 2-link arm equations (whichever textbook you derived from).

## Decisions still yours (write them in EXPERIMENTS / DESIGN_LOG before the report)

1. Headline: the rule-selected models (recommended, clean test) or also quote the ff6 + cons MLP as "best
   found" (validation-chosen, but test already seen for it)?
2. How to frame accuracy (MLP) vs consistency and per-query cost (DeepONet).
3. ~~Whether to rerun the benchmark with desktop programs closed.~~ Resolved 2026-10-21: rerun done (EXPERIMENTS B2); quote the B2 numbers.
4. Whether to add a phase-coverage test (the one missed mutation; sine family only) or state that the sine
   family is diagnostics only.
5. The E1 / E2 / E3 / E4 "Decision: [fill in]" lines in `docs/EXPERIMENTS.md`.

## Questions to rehearse (new analyses; add to STUDY_GUIDE section 7)

1. Why is batched RK4 a fairer rival than `solve_ivp`, and why does the speed-up shrink with batch size?
2. What does "trunk cached" mean, and why can't the MLP do the same?
3. Why does the violent sample show observed order ~5, not 4, in the convergence plot?
4. Why is the energy residual on the 20 ms grid larger than on the 2.5 ms grid, and why is it subtracted?
5. The OOD m_p/b error is lower than the test error. Does that mean the model extrapolates well?
6. What does the sensitivity figure show, and what does it not prove?
7. Why does the consistency penalty not cost accuracy, and why did Fourier features make the MLP less consistent?
8. What does a "caught" mutation prove about a test, and why can test_benchmark catch dynamics bugs?
