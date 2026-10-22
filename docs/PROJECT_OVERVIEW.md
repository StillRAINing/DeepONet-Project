# Project overview: everything at the high level (as of 2026-10-22)

The one document to read before the report and the interview. It states **what was built, what was
decided, what came out, and why**, with every number traced to the file it comes from. It replaces
`docs/DeepONet_Project_Explained.pdf` (written 2026-10-12, before the smooth torque family, the
models and every result existed; it is now wrong on the dataset, the inputs and the status).

It is a map, not report text, and it deliberately gives no derivations for the crucial modules
(dynamics, RK4 order, the DeepONet contraction, the loss, the metrics). For those, use
`docs/STUDY_GUIDE.md` and be able to do them with this page closed. AI-written, 2026-10-22
(AI_LOG row 52). Check every number against its source file before you quote it.

---

## 1. The project in six sentences

1. A 2-link planar arm in a vertical plane, with a tip payload `m_p`, viscous joint friction `b` and two
   joint torques `tau_1(t), tau_2(t)`, is simulated with a hand-written RK4 solver (the ground truth).
2. A DeepONet and an MLP are trained to map `(m_p, b, torque functions)` to the 3 s trajectory
   `[q1, q2, q1dot, q2dot]` on 151 times, from 7,000 RK4 trajectories (seed 3001).
3. Under an equal-parameter (about 171k) and equal-tuning protocol, **the MLP is more accurate**
   (test median 0.071 vs 0.109 for the best DeepONet), beyond seed noise.
4. **The DeepONet is more physically self-consistent** (`qdot` vs `dq/dt` 0.11 to 0.15 vs 0.22 to 0.24)
   and **far cheaper per query once its trunk is cached** (0.02 ms vs 0.27 ms per trajectory at batch 64).
5. **Every model fails on the same hard 5 to 10 %** (the spinning-elbow regime: p95 about 0.6) and on
   torque amplitudes above the training range (median 0.53 to 0.83).
6. Counting data generation and training, the surrogates pay off after **70k to 205k queries** against
   `solve_ivp` at matched error, but only after **0.8 to 3.0 million** against an exact batched NumPy RK4,
   the honest rival.

---

## 2. The physical system (`src/dynamics.py`, yours to derive)

| Item | Value / form |
|---|---|
| State | `x = [q1, q2, q1dot, q2dot]`; `q1` from the horizontal, `q2` relative to link 1 |
| Equation | `M(q) qddot + C(q, qdot) qdot + G(q) + b qdot = tau` |
| Solved as | `qddot = M^-1 (tau - C qdot - G - b qdot)` with a linear solve (`np.linalg.solve`), not an inverse |
| Constants (`ArmParams`) | `l1 = 0.5`, `l2 = 0.4` m; `m1 = 2.0`, `m2 = 1.5` kg; `lc = l/2`; `I = m l^2 / 12`; `g = 9.81` |
| Payload | point mass at the tip of link 2, folded into link 2's mass, COM and inertia (`combine_payload`, parallel-axis theorem: two correction terms) |
| Gravity | enters as `cos q1` and `cos(q1 + q2)` because `q1` is measured from the horizontal |
| Initial condition | fixed: `x0 = 0` (arm horizontal, at rest); gravity alone swings it, so no trajectory is trivial |
| Energy | `energy()`; with torque and friction `dE/dt = tau . qdot - b |qdot|^2`, so raw energy is NOT conserved; the *balance* is the check |
| Gravity torque at start | about 15 N m (`m_p = 0`) to 24 N m (`m_p = 1`) on joint 1, so a 6 N m torque cannot hold the arm up, it perturbs the swing |

Verification evidence: `tests/test_dynamics.py` (payload vs brute-force geometry, Coriolis identity,
gravity = gradient of potential, energy balance, small oscillations about the hanging equilibrium).
The 2026-10-12 audit also matched `M`, `C`, `G`, `energy` against an independent geometric reference to
about 1e-9 (`REVIEW_NOTES.md` section 1). **Still owed by you:** re-derive `M`, `C`, `G` and the payload
terms by hand (AI_LOG item 3, DESIGN_LOG E). `docs/physics/*.pdf` are reading material, not your derivation.

Examiner hooks: why `M` depends on `q2` only; why `M` is symmetric positive definite; units of every
term; what changes if the payload sat on link 1.

---

## 3. Ground truth and solver (`src/solvers.py`, yours to defend)

| Item | Value |
|---|---|
| Scheme | classical fixed-step RK4 (`rk4_step`, `rk4_integrate`); adaptive RK45 also exists (`rk45_*`) |
| Step | output grid 151 points over 3 s (20 ms); **8 substeps** per output interval, so `h = 2.5 ms`, 1,200 steps, 4,800 calls to `state_derivative` per trajectory |
| Why fixed step | same scheme for every label, simple to verify (order test), cheap to vectorise; accuracy checked against DOP853 |
| Order | 4.00 to 4.12 on `y' = -y`; on a calm arm sample 3.83, 3.92, 3.96 at the smallest steps; the most violent sample shows about 5 at mid `h` (pre-asymptotic, not a contradiction) |
| Energy check (b = 0, tau = 0) | drift 4.3e-5 J at 2.5 ms (relative 2e-6); falls 35x, 19x, 29x per halving (at least 16x, not clean) |
| Energy balance (torque + friction) | residual 1.2e-3 J on the 2.5 ms grid, 7.4e-2 J on the 20 ms output grid (quadrature, not RK4; the evaluation subtracts this floor) |
| Label accuracy | stored labels vs DOP853 rtol 1e-10: median about 1e-8 on every split; worst in range 5e-5; worst ood_amp2 sample 2.4e-2 (still about 30x below the surrogates' 0.8 there) |

Sources: `results/solver_checks.md`, `results/solver_convergence.png`, `solver_energy.png`,
`solver_vs_dop853.png`, EXPERIMENTS "Truth check", `tests/test_solvers.py`.

Why "16x per halving" can't be read from one run, why the arm order test uses a mild sample (`MILD` in `tests/test_solvers.py`), and
why the violent one shows order 5: STUDY_GUIDE section 2 and RESULTS_SUMMARY rehearsal question 3.

---

## 4. Data (`src/data.py`)

### 4.1 History: why the torque family changed

The first family was `tau_j = A_j sin(2 pi f_j t + phi_j)` with `A` up to 15 N m. The 2026-10-12 review
(`REVIEW_NOTES.md`) found: 72 % of trajectories spun past 2 pi (q2 up to 131 rad), RK4 labels were
unconverged on the worst (up to 13 % error), the OOD set was calmer than train, and the 304-number input
was really 6 torque numbers (CLAUDE.md weak point 1). Decisions D1 and D2 (DESIGN_LOG): adopt a
**smooth random torque family** as the main experiment; keep the sine family for diagnostics only.

### 4.2 The smooth family (main experiment, `data/dataset_smooth.npz`)

| Setting | Value | Reason (DESIGN_LOG A6, A7) |
|---|---|---|
| Form | `tau_j(t) = sum_{k=1..6} c_jk cos(2 pi k t / P) + s_jk sin(2 pi k t / P)` | evaluable at any `t` (RK4 needs `t + h/2`), fixed-meaning coefficients for the compact baseline |
| Period | `P = 2T = 6 s` | avoids forcing `tau(0) = tau(T)` |
| Spectrum | coefficient std `exp(-(omega_k ell)^2 / 4)`, `ell = 0.2` in normalised time (0.6 s) | approximates an RBF-kernel GP; `ell = 0.2` is Lu et al.'s setting |
| Modes | `K = 6` per joint, so 24 coefficients in total | variance beyond K = 6 is 4.4e-5 |
| Amplitude | each joint's shape rescaled to a peak `A_j ~ U(0, 6)` N m | caps spinning; 95th percentile of peak `|q|` about 14 rad |
| Effective dimension | 12 components for 99 % of the variance, 14 for 99.9 % | input is richer than the sine family (6) but still low-dimensional |
| `m_p`, `b` | `[0, 1]` kg, `[0.05, 0.5]` N m s/rad | unchanged from the sine family (change one thing at a time) |
| Splits | 7,000 train / 1,500 val / 1,500 test, seed 3001 | Lu et al. use 10,000; about 19 % of in-range trajectories pass 2 pi |
| Generation | 5.5 min with 8 workers (scalar RK4); 8 s with the vectorised RK4 | |

### 4.3 Out-of-range (OOD) sets, 200 each, never used for selection

| Set | What changes | What it really tests |
|---|---|---|
| `ood` | `m_p` in [1, 1.5], `b` in [0.5, 0.8] | a **mild** shift: heavier and more damped means calmer; no trajectory passes 2 pi |
| `ood_amp1` | peak torque 6 to 9 N m | amplitude extrapolation, and a regime change (73 % spin past 2 pi) |
| `ood_amp2` | peak torque 9 to 12 N m | the same, further out (94 % spin) |
| `ood_smooth` | rougher torque: `ell = 0.1`, K = 13 modes | a different input function family; the only test where "function as samples" input can matter |

Shells rather than caps, so no OOD sample lies inside the training amplitude range. Framing everywhere:
**how error degrades**, not pass/fail (weak point 4).

---

## 5. Inputs, outputs and the two input encodings

| | Shape | Notes |
|---|---|---|
| `params` | `(B, 2)` = `[m_p, b]` | |
| torque **samples** input | `(B, 2, 151)`, flattened: branch input `d_u = 2 + 302 = 304` | the standard DeepONet "sensor" representation |
| torque **compact (coef)** input | `d_u = 2 + 2 x 2 x 13 = 54` | the 24 coefficients, zero-padded to 13 modes so `ood_smooth` fits; modes 7 to 13 are always zero in training, so `ood_smooth` is pure extrapolation for coef models |
| time | `t / T` in [0, 1] | trunk (DeepONet) or extra input (MLP) |
| output | `(B, 151, 4)`, channels `[q1, q2, q1dot, q2dot]` | raw units after decoding; no angle wrapping |

Honest wording (D11): the torque is a random band-limited function, finite-dimensional in effect (24
numbers), presented to the network as 151 samples per joint; **no claim about arbitrary input functions**.
The map is parameters-to-one-trajectory (fixed `x0`, fixed 3 s), not a general dynamics operator (weak point 7).

Input validation (`src/validate.py`, `src/surrogate.py`): invalid input (wrong shape, NaN or inf,
`m_p < 0`, `b < 0`, `t < 0`, batch mismatch) raises `ValueError`; valid but untrained input (outside the
trained range, peak torque above 6 N m, `t > 3 s`) gives `OutOfRangeWarning` and still returns a
prediction (`strict=True` raises). The simulators check physical validity only, because the OOD sets are
generated with them.

---

## 6. The models (`src/model.py`, the DeepONet forward is yours to explain)

| | DeepONet (unstacked, with bias) | MLP baseline |
|---|---|---|
| Idea | branch(u) gives `p` coefficients per channel; trunk(t) gives `p` basis functions per channel; output = their dot product + bias | one network on the concatenation `[u, t]`, evaluated at each of the 151 times |
| Contraction | `einsum("bkp,btkp->btk")`: branch `(B, 4, p)`, trunk `(B, n_t, 4, p)`, sum over `p` | none |
| Depth / width | branch 3 and trunk 3 hidden layers, tanh; width 128 (samples) or 147 (coef); `p = 64` | 4 hidden layers, tanh; width 192 (samples) or 229 (coef) |
| Parameters | 171,396 (samples), 171,183 (coef) | 170,692 (samples), 171,754 (coef) |
| Trunk caching | the trunk never sees `u`, so on the fixed grid it can be evaluated once and reused: one branch pass per query | impossible: `t` enters the first layer together with `u` |
| Activation | tanh (smooth, so `dq/dt` by autodiff is meaningful; ReLU would give piecewise-constant `dq/dt`) | same |

Four models in the comparison: {DeepONet, MLP} x {samples, coef}, all about 171k parameters
(`--match-params 171396`, fixes weakness W1). Why the compact input exists: CLAUDE.md weak point 2. A
304-input MLP alone would be a handicapped baseline.

Not enforced by any model: `x(0) = x0` (median `|x_pred(0)|` is 0.3 to 1.3 % of channel rms for
positions, 1.3 to 6.3 % for velocities; `results/final_analysis.md`) and `qdot = dq/dt` (measured, see 8).

---

## 7. Training (`src/train.py`, yours to defend; most of it is AI-written, AI_LOG rows 16, 17, 20, 29)

| Item | Setting | Why |
|---|---|---|
| Output scaling | per-channel z-score, **train statistics only** (`ChannelNormaliser`) | channels differ in units and size; raw `q2dot` would carry about half of `sum(x^2)` |
| Input scaling | per-feature z-score, train only (`InputNormaliser`); constant features get std 1 | std 1, not a tiny floor, so the zero-padded modes don't explode on `ood_smooth` |
| Loss | per-sample relative L2 over (time, channel) on scaled data, batch mean (`relative_l2_loss`); denominator = size of the (scaled, uncentred) true trajectory | every trajectory counts equally; never near 0 (smallest trajectory norm 31) |
| Optimiser | Adam, no weight decay (your decision 2026-10-13), batch 100, 200 epochs | |
| Schedule | cosine from `lr` to `lr/100`, chosen per model on validation | helped or was neutral in all 8 pairs (E3) |
| lr grid | 3e-4, 1e-3, 3e-3 constant; 1e-3, 3e-3 cosine (5 settings per model) | same grid for every model (equal tuning) |
| Selection | weights at the epoch of lowest **validation** loss | the test split is never passed to `fit` |
| Seeds | data seed 3001 fixed; training seeds 3001, 1, 2 for the selected settings | one run is an anecdote |
| Reproducibility bug | until 15:30 on 2026-10-13 the initial weights were not seeded (`build_model` ran before `torch.manual_seed`); fixed in `train_quick.py`, flagged as `init_seeded` in `runs.csv` | early E1/E2 runs are not bit-reproducible; the 3-seed retrain covers the finalists |
| Optional E4 terms | `--fourier K`: time features `[t, sin k pi t, cos k pi t]`; `--consistency w`: penalty on `dq/dt - qdot` | validation and selection always use the plain relative L2 |

**Selection rule** (written before any E2/E3 result, EXPERIMENTS): (1) per model, the (lr, schedule) with
the lowest val loss, seed 3001; (2) retrain with seeds 1 and 2; (3) only then evaluate the 12 checkpoints
once on test and all OOD sets. Selected: MLP coef 3e-3 cosine; MLP samples 1e-3 cosine; DeepONet coef 3e-3
cosine; DeepONet samples 1e-3 cosine.

---

## 8. Evaluation (`src/evaluate.py`, defines what "accurate" means)

| Metric | Definition |
|---|---|
| Headline error | per-sample relative L2 over (time, channel), channels scaled by the train std; report **median, p95, max, fraction under 5 % / 10 %**, never a mean alone |
| Per channel | each channel on its own scale (`rel_l2_per_channel`) |
| Error vs time | `|error(t)|` divided by the whole-trajectory rms (a pointwise relative error is undefined at `t = 0`, where `x0 = 0`) |
| Consistency | rms(`dq/dt` - `qdot`) / rms(true `qdot`), `dq/dt` by autograd of the q outputs, times `std_q / T` for units |
| Energy balance | residual `E(t) - E(0) - integral(tau . qdot - b |qdot|^2)` on the prediction, minus the truth's quadrature floor, as a fraction of the energy swing |
| Yardsticks | predict the mean training trajectory: test median 0.43 (p95 0.87); 1-nearest-neighbour in coef space: 0.42 (p95 1.01) |

---

## 9. Results: the numbers to know

### 9.1 Headline (test, n = 1500, rule-selected, mean +- std over 3 seeds, `results/final_eval.md`)

| Model | Input | Median | p95 | Max | Under 10 % | Consistency |
|---|---|---|---|---|---|---|
| MLP | coef | **0.071 +- 0.000** | **0.600** | 1.52 | **58.2 %** | 0.244 |
| MLP | samples | 0.080 +- 0.001 | 0.603 | 1.57 | 55.8 % | 0.223 |
| DeepONet | coef | 0.109 +- 0.001 | 0.653 | 1.54 | 47.3 % | **0.109** |
| DeepONet | samples | 0.130 +- 0.007 | 0.630 | 2.06 | 39.3 % | 0.147 |

- The validation ranking survives on test (medians within 0.003 of validation): selection did not overfit val.
- MLP-vs-DeepONet val-loss gaps are 7.7 to 12 seed standard deviations (`results/analysis_seeds.md`).
- Compact input beats samples input in distribution for both families (about 2 std for the MLP, 4 for the DeepONet).

### 9.2 Where the error lives (`results/final_analysis.md`, seed 3001)

- **Regime** (peak `|q|`): below pi (67 % of test) median 0.045 to 0.089; pi to 2 pi 0.12 to 0.18; above 2 pi 0.37 to 0.54.
- **Time**: median grows about 4x from 1 s to 3 s; p90 grows 6 to 9x.
- **Channel**: q1 about 0.03; q2 0.16 to 0.21; q1dot 0.08 to 0.15; q2dot 0.19 to 0.35. The elbow and the velocities are hard.
- **Sensitivity of the true map**: a 1 % input change moves calm trajectories 0.6 %, spinning ones 4 to 8 %
  (p90 32 to 56 %); Spearman +0.81 with the MLP's error. The tail is hard because the map is hard
  there (correlation, not proof).

### 9.3 Out of range (median, 3 seeds)

| Model | Input | `ood` (m_p, b) | `ood_amp1` (6 to 9) | `ood_amp2` (9 to 12) | `ood_smooth` |
|---|---|---|---|---|---|
| MLP | coef | 0.052 | 0.53 | 0.70 | 0.220 |
| MLP | samples | 0.052 | 0.54 | 0.78 | 0.167 |
| DeepONet | coef | 0.060 | 0.62 | 0.83 | 0.212 |
| DeepONet | samples | 0.065 | 0.61 | 0.81 | **0.155** |

- `ood` error is *lower* than test because the set is calm; matched on trajectory size, it is within about 25 % of in-range error. **Not evidence of extrapolation.**
- Amplitude: no cliff at 6 N m. Error already grows about 20x inside the range, roughly doubles across the edge, then saturates at 0.6 to 0.8.
- `ood_smooth`: samples input beats coef input for both families, the reverse of in-distribution. This is the
  one place the function-as-samples encoding pays off. DeepONet vs MLP on samples (0.155 vs 0.167) is about 1 std apart: **unresolved, don't claim a DeepONet win**.

### 9.4 E4 fair-fight follow-ups (validation-chosen; test seen afterwards, so post hoc)

- Fourier time features (ff6): help the samples-input models (DeepONet test median 0.130 to 0.105), not the coef ones; **the gap is not closed**.
- Consistency penalty (weight 0.1, added on top of ff6, samples input): DeepONet consistency 0.071 (ff6) to 0.008 (ff6 + cons), MLP 0.30 to 0.12, **no accuracy cost**.
  Fourier features alone make the MLP *less* consistent (0.22 to 0.30 samples, 0.24 to 0.42 coef).
- Best configuration found: **MLP, samples, ff6 + cons, test median 0.066 +- 0.002**. It is not the headline,
  because the test set was seen for it.
- Data size (n = 500 to 7,000, one seed): median falls about as n^-0.5 (slopes -0.43 to -0.55), no saturation; p95 only 0.77-0.82 to 0.64-0.68. MLPs lead at every n >= 1000; the DeepONet does not overtake at small n.
- Basis size p (one seed): p = 8 clearly worse; 16 to 128 a plateau; p = 256 at equal budget worse (width 71). p = 64 is defensible (W2).

### 9.5 Speed (`results/benchmark.md`, B2 quiet rerun 2026-10-21, 64 val trajectories)

| Method | ms / trajectory |
|---|---|
| scalar Python RK4 (made the labels) | 121 |
| vectorised NumPy RK4, batch 64 / batch 1500 (exact, same scheme) | 4.4 / 0.73 |
| `solve_ivp` RK45 at the surrogates' median error (rtol 0.03 to 0.05) | 7.4 to 8.1; fails on 1 to 2 of 64 at rtol 0.08 to 0.1 |
| MLP coef / samples, batch 64 | 0.27 / 0.28 |
| DeepONet coef / samples, batch 64 | 0.18 / 0.14 |
| DeepONet, trunk cached, batch 64 | 0.02 (0.007 on one thread) |

- Speed-up vs vectorised RK4 shrinks with batch size: 137 to 161x at batch 1, 17 to 35x at 64, 3 to 7x at 1500 (un-cached). Cached DeepONet: 201 to 359x at every batch.
- Why the scalar solver is slow: 4,800 Python calls to `state_derivative` per trajectory (cProfile). Vectorising over the batch gives 27x with identical labels.
- FLOPs per trajectory: MLP 5.2e7, DeepONet 2.0 to 2.5e7, cached 2.6e5 to 2.9e5. Parameter memory 0.69 MB each.
- Up-front cost: data 8 s (vectorised) or 1,206 s (scalar); training 556 to 1,490 s per run (idle, scaled from 3 epochs).
- **Break-even**: 70k to 205k queries vs matched `solve_ivp`; **0.8M to 3.0M vs batched RK4 at batch 1500**; 3.8M to 14.9M if the 5-run lr grid is counted.
- The "about 850x" against the scalar loop is the number **not** to headline (weak point 3).
- B1 (2026-10-15) had 30 to 35 % background load; it had flattered the surrogates (solvers sped up about 2x in B2, surrogates only 1.2 to 1.5x).

### 9.6 Code verification

- 117 tests, 0 skipped, 0 failed (`results/test_report.xml`, 2026-10-22).
- Mutation check (`results/mutation_check.md`): 10 deliberate bugs, 9 caught by a targeted test. The three
  survivors of the 2026-10-12 review (two parallel-axis terms, a perturbed RK4 stage) are now caught.
  Still missed: phase range halved in the **sine** family sampler (diagnostics only; the smooth family has no phase).
- Most of those tests were AI-written (AI_LOG row 36), so be ready to explain what each targeted test checks.

---

## 10. How each known weak point was handled (CLAUDE.md, 2026-10-12 review)

| # | Weak point | What was done | What to say |
|---|---|---|---|
| 1 | Input is really a few numbers | Smooth random family (24 numbers), `ood_smooth` set | Still low-dimensional; "operator framing only partly earned"; `ood_smooth` is the evidence for sampled input |
| 2 | Unfair MLP baseline | Compact-input MLP and DeepONet, matched 171k parameters, same lr grid, pre-written selection rule | The MLP wins; legitimate, reportable |
| 3 | Inflated speed claim | Vectorised RK4, `solve_ivp` at matched error, batch sweep, break-even incl. data + training, quiet rerun | Fair rival is batched RK4; only the cached DeepONet stays far ahead |
| 4 | OOD barely tests anything | Amplitude shells and a rougher family added; magnitude-matched comparison | Degradation curve; `ood` is a mild shift |
| 5 | Sensitivity and scaling | Amplitude capped at 6 N m; per-channel scaling; per-channel, per-time, per-regime errors, worst cases | Tail remains unlearnt; the true map is sensitive there |
| 6 | Outputs not consistent | Consistency metric; energy-balance excess; optional penalty | DeepONet more consistent by architecture; penalty fixes both |
| 7 | Fixed IC and horizon | Stated as scope | Do not over-claim |
| 8 | Leakage | Train-only statistics, test once, OOD never used for selection, seed 3001 recorded | Two post-hoc variants seen on test: say so |

---

## 11. Limitations (say them before the examiner does)

- Simulation only; the network matches the *model*, not a robot. Planar 2-link, viscous friction, fixed `x0` and 3 s horizon.
- Torque input is about 24 numbers; no claim about arbitrary functions.
- Spinning tail unlearnt (p95 about 0.6, worst cases 1.4 to 2.1, worse than predicting zero); no amplitude extrapolation.
- `ood` (m_p, b) is a mild shift.
- One data split, 3 training seeds, one model size; data-size and p sweeps single-seed.
- Timings: one laptop, one run; run-to-run variation about 10 to 20 %.
- Test set also used to report two post-hoc variants (ff6, ff6 + cons); the headline stays rule-selected.
- Initial condition not enforced (velocity error at `t = 0` up to about 6 % of the channel rms).

---

## 12. Where everything is

| Need | File |
|---|---|
| Headline tables | `results/final_eval.md` (+ `_ff6`, `_ff6cons` for post-hoc variants) |
| Every follow-up table | `results/final_analysis.md` |
| Speed | `results/benchmark.md` |
| Solver checks | `results/solver_checks.md` |
| Test quality | `results/test_report.xml`, `results/mutation_check.md` |
| Which figure shows what | `docs/FIGURE_PLAN.md`, `results/INDEX.md` |
| Claim -> number -> file per report section | `docs/RESULTS_SUMMARY.md` |
| What was run, when, with what outcome | `docs/EXPERIMENTS.md` |
| Why each design choice | `docs/DESIGN_LOG.md` |
| Interview questions (no answers) | `docs/STUDY_GUIDE.md` |
| AI use | `AI_LOG.md` |
| Report draft | `report/MMA3001_Report_DRAFT.docx` (scaffold; the reasoning sections are yours) |
| Live demo | `python -m src.demo` (RK4 vs the four models; N = test sample, O = OOD sample) |

---

## 13. Your 60-second spoken summary (draft facts; say it in your own words)

Problem, then method, then the three findings (MLP more accurate; DeepONet more consistent and cheaper per
cached query; everyone fails on the spinning tail and on stronger torques), then the honest speed answer
(break-even 0.8 to 3 million queries against batched RK4), then one limitation (low-dimensional input,
simulation only). If you can't say this without notes, go back to sections 1 and 9.
