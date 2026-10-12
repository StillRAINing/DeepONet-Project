# Review notes (AI-assisted audit, 2026-10-12)

Written while you were away. **Nothing here changes your crucial code** (`dynamics.py`,
`solvers.py`, `model.py`). It is evidence for you to act on. Numbers were produced by
`scripts/data_stats.py` and throwaway scripts; reproduce them before quoting any in the report.

Verdict in one paragraph: `dynamics.py` is physically correct. The *dataset design* is the
problem: with `A` up to 15 N m the arm spins wildly, the ground truth is not converged on the worst
trajectories, and the OOD set is tamer than training. Fix these before training anything.

---

## 1. `dynamics.py`: verified correct

Independent check (not copied from your code): I built T and V from raw link geometry
(COM positions, point-mass payload, finite-difference velocities) and compared with your functions
at 5 random `(q, qdot)` with `m_p = 0.7`:

| Quantity | How compared | Max discrepancy |
|---|---|---|
| `energy()` | T + V from geometry | 5e-10 |
| `mass_matrix()` | Hessian of T w.r.t. `qdot` | 4e-10 |
| `gravity_vector()` | gradient of V w.r.t. `q` | 2e-9 |
| `coriolis_vector()` | power identity `qdot . C qdot = 0.5 qdot . Mdot qdot` | 5e-11 |

So the maths is right. That does **not** mean you can defend it: AI_LOG item 3 (re-derive by hand)
is still open. About `docs/physics/*.pdf` and `docs/DeepONet_Project_Explained.pdf`: I earlier
assumed these were your derivations, but on reading them I cannot tell who made them, and their style
(identical "set-up" boilerplate on every PDF, "derived from scratch") reads like generated explainers.
If AI produced them, they are **not logged in `AI_LOG.md`** (rows 1-7 do not cover them): add a row.
I did check their worked-example numbers against the code (combine_payload, M, Coriolis, G, state
derivative, energy): they all reproduce exactly, so the PDFs are numerically consistent with `dynamics.py`.
Either way they are reading material, not evidence that you re-derived anything. Your own handwritten
derivation is what the interview will test.

### Your tests do not catch everything (mutation test)
I copied the repo to a scratch folder, broke one thing at a time, and ran the suite:

| Mutation | Caught? |
|---|---|
| Coriolis `2 q1d q2d` -> `1 q1d q2d` | yes |
| Coriolis row-2 sign flipped | yes |
| Gravity row 2 uses `cos(q1)` instead of `cos(q1+q2)` | yes |
| Friction sign flipped | yes |
| `M12` coupling factor 2 | yes |
| **Parallel-axis: drop `m2 (lc2e - lc2)^2`** | **NO** |
| **Parallel-axis: drop `m_p (l2 - lc2e)^2`** | **NO** |
| **RK4 `k2` perturbed by 1% (order drops)** | **NO** |
| **Phase range in `data.py` halved to [0, pi)** | **NO** |

(Some "yes" cases were caught incidentally, e.g. by a blow-up, not by a targeted assertion.)

Why the survivors survive: `test_zero_payload_...` and `test_huge_payload_...` only check `m_p = 0`
and `m_p -> infinity`, where the extra inertia terms vanish or are negligible. Nothing checks a
mid-range payload against a reference. **Tests for you to write** (guide in `STUDY_GUIDE.md`):
- payload inertia against a brute-force reference (point masses summed directly);
- RK4 order test (halve `h`, error falls ~16x) on `y' = -y` and on the arm;
- a statistical test that sampled `phi` covers [0, 2 pi).

There was **no `tests/test_solvers.py`** at all, the largest gap in the unit's core numerical content.
I added `tests/test_solvers.py` as a *skeleton*: six skipped tests with docstrings saying what to check
and nothing else. The same goes for three skipped tests in `test_dynamics.py` and four in
`test_model.py`. Writing them is your job (and the mutation table is the acceptance test).

---

## 2. Dataset: findings (the part that most needs your attention)

### 2.1 The arm spins out (CLAUDE.md weak point 5, confirmed and worse than expected)
Train split (1400 trajectories, `A` in [0, 15]):

| Channel | std | min | max | share of `sum(x^2)` |
|---|---|---|---|---|
| q1 | 3.7 | -38.7 | 27.2 | 5.5% |
| q2 | 10.0 | -84.3 | **131.5** | 27.1% |
| q1dot | 5.9 | -49.5 | 40.3 | 9.9% |
| q2dot | 14.5 | -149.3 | **196.5** | **57.5%** |

- 71.7% of trajectories have `|q| > 2 pi` at some time. `q2 = 131 rad` is about 21 revolutions.
- A joint-2 torque of 15 N m on a 1.5 kg, 0.4 m link (`I ~ 0.02 kg m^2` about its own COM, ~0.08 about joint 2) is a huge acceleration.
  `A` is not scaled to the link inertia.
- A plain relative L2 over all channels is more than half decided by `q2dot`. A network can look
  good while `q1` is poor. Normalise per channel, report per channel.

Effect of one common `A` upper bound (60 samples each, `substeps = 8`; indicative only, sampling noise
explains small differences from the 1400-sample train numbers above):

| A max | frac with `|q| > 2 pi` | p99 `|q2|` | max `|q2dot|` |
|---|---|---|---|
| 2 | 0.00 | 1.5 | 22 |
| 3 | 0.00 | 1.8 | 25 |
| 5 | 0.13 | 8.1 | 48 |
| 8 | 0.38 | 22.7 | 62 |
| 15 | 0.60 | 33.1 | 99 |

A single cap of 2-3 N m removes the wrapping, but a per-joint cap looks better. Measured with `python -m scripts.torque_range_sweep` (40 samples per row; "influence" = relative L2
distance between a trajectory and the same arm with torque switched off, so it says how much the
function input matters; "rk4-err" = substeps 8 vs 32):

| A1 max | A2 max | frac `|q|>2pi` | p99 `|q|` | influence | rk4-err |
|---|---|---|---|---|---|
| 15 | 15 | 0.70 | 25.8 | 3.48 | 5.3e-3 |
| 8 | 8 | 0.35 | 17.8 | 1.95 | 2.2e-5 |
| 8 | 2 | 0.03 | 3.1 | 0.83 | 7e-8 |
| 5 | 5 | 0.15 | 7.7 | 1.22 | 4e-7 |
| 5 | 2 | 0.00 | 3.0 | 0.64 | 3e-8 |
| 3 | 3 | 0.05 | 3.2 | 0.75 | 6e-8 |
| 2 | 2 | 0.00 | 2.9 | 0.51 | 2e-8 |

So lowering `A` does not make the torque irrelevant: even at `A <= 2` it changes the trajectory by
~50% against the no-torque arm, while the motion stays within a few radians and the RK4 ground truth is
converged (error ~1e-8). The `A2` cap is what matters most (joint 2 is the light, fast one);
`(A1, A2) = (8, 2)` or `(5, 2)` look like sensible candidates. Your decision and your justification;
note that this is an indicative 40-sample screen, not a tuned choice.
Full-size check of `(A1, A2) = (8, 2)` (2000 samples drawn with seed 3001, first 1400 as "train", scratch
only, nothing in `data/` changed; generation took 191 s): std of `q1, q2, q1dot, q2dot` = 1.2, 1.3, 3.2, 4.2
(min/max within about +-17 rad, +-42 rad/s), 7.6% of trajectories beyond `2 pi`, peak `|q2|` median 1.3 /
p90 2.4 / p99 10.4 / max 17.8 rad. RK4 at `substeps = 8` against `substeps = 32` on the 10 worst plus 20 random
samples: worst relative error 8e-6, median 6e-8 (so the ground truth is converged). `q2dot` still carries
52% and `q1dot` 31% of `sum(x^2)`, so per-channel normalisation is still needed.
Keeping `A = 15` is defensible only if you *want* chaotic
spinning; then say so and expect large errors at late times.
Note the comment in `data.py` ("gravity torque at the start pose is ~12 N m") is stale: it is about
**15 N m at `m_p = 0` and 24 N m at `m_p = 1`** (joint 1), which is a separate reason `A` must be
able to compete with gravity. Re-derive and fix the comment.

### 2.2 The ground truth is not converged on the worst trajectories
`simulate(..., substeps=8)` (h = 0.0025 s) against `substeps = 64` on three of the five largest-`|q2|`
train samples (the other two behave similarly):

| Sample | max `|q2|` | rel. L2 at substeps 4 / 8 / 16 |
|---|---|---|
| 458 | 131 | 2.6e-1 / **1.3e-1** / 8e-3 |
| 107 | 110 | 1.2e-1 / 9.5e-3 / 3e-4 |
| 605 | 84 | 1.6e-1 / 1.8e-2 / 7e-4 |

Random samples are fine (rel. L2 ~1e-7 at substeps 8). Against `substeps = 32` on 150 random train
samples: median 7e-7, 99th percentile 1.5e-3, 1.3% above 1e-3, none above 1e-2. So the problem is
confined to the spinning tail, but a surrogate evaluated on that tail is being compared with a
*wrong* answer. Options: lower `A` (2.1 fixes this too), raise `substeps` for the dataset, or
measure and report the RK4 reference error as a floor on every surrogate error. This is also a
good place to apply your halving-`h` convergence test to the real dataset, not just a toy ODE.

### 2.3 The OOD set is not "harder", it is different
OOD (heavier payload, higher friction) is *calmer*: q2 std 2.5 vs 10.0 in train, only 27.5% of
trajectories exceed `2 pi` vs 71.7%. Extra friction damps the motion. Consequences:
- A low OOD relative error can come from easy trajectories, not from extrapolation skill.
- Compare OOD error against an **in-range subset matched on trajectory magnitude**, or report the
  absolute RMSE too.
- Torque amplitude and frequency are never out of range (weak point 4). Add an OOD set with
  `A` and/or `f` outside the training range; that is the extrapolation that actually bites.

### 2.4 Input redundancy (weak point 1)
Branch input is 304 numbers generated from 8 (2 params + 6 torque coefficients). The 302 torque
samples have numerical rank 38 over the training set (tolerance 1e-6), far below 302 but
not 6, because `sin` is nonlinear in `f` and `phi`. The honest statement for the report is
"8-parameter surrogate". See section 4 for the baseline this implies.

### 2.5 Regime switching: peak `|q2|` is bimodal (`results/data_peak_vs_inputs.png`)
Plotting each trajectory's peak `|q2|` against `A2` and `b` shows two populations: one that stays
within about 1-3 rad and one clustered in bands near 8-9 rad and 14-15 rad, even for small `A2`. So the
input-to-output map is close to *discontinuous*: a small change in `(m_p, b, A, f, phi)` can tip the
elbow over into a full flip, after which the trajectory is qualitatively different. That is a much
harder regression target than "smooth in the inputs" (neural nets smooth over jumps) and will likely
dominate your worst-case errors. What I checked: the populations differ mainly in `A2`
(mean `A2` is 3.6 N m in the low group, 7.9 in the 6-11 rad group, 10.6 in the group above 11 rad;
`A1`, `b` and `m_p` barely differ), and in the 6-11 rad group the elbow first passes `pi` anywhere
between 0.4 s and 2.8 s. So the likely story is "strong enough joint-2 torque pushes the elbow past
the folded-back configuration and it winds on", with a timing that depends sensitively on the phase
and frequency. That is my interpretation of a 40-ish-sample look, not a proof: plot a trajectory from
each group and test it (e.g. does `|q2|` crossing `pi` predict the group?). Report it as an
observation, and look at the error split by regime (flipped vs not flipped). With a lower `A2` cap (2.1) the upper population should mostly disappear.

### 2.6 Reference floors on the validation split (context, not results)
Throwaway script, **validation split only**, normalisation statistics from train only. Relative L2 of the
prediction against the truth on the 300 val trajectories (current dataset, `A` up to 15):

| Predictor | pooled | per-trajectory mean / median / p95 | per channel (q1, q2, q1dot, q2dot) |
|---|---|---|---|
| zeros | 1.000 | 1.00 / 1.00 / 1.00 | 1, 1, 1, 1 |
| train-mean trajectory | 0.986 | 0.96 / 0.98 / 1.01 | 0.82, 1.00, 0.94, 1.00 |
| ridge, 10 compact features | 0.965 | 1.00 / 0.98 / 1.24 | 0.77, 0.97, 0.91, 0.98 |
| ridge, quadratic in those features | 0.911 | 1.03 / 0.94 / 1.58 | 0.69, 0.90, 0.89, 0.94 |

Reading: on this dataset trivial and (near-)linear predictors are at ~1.0, i.e. the map is strongly
nonlinear and `q2`, `q2dot` are close to unpredictable for them. Any network result should be judged
against these floors and against the RK4 reference error (2.2), not against 0. It also shows that
"pooled" and "per-trajectory" relative L2 differ (and the ridge p95 exceeds 1), so define yours
precisely. If you lower `A` (2.1) these floors will change; rerun them. I did one such check with the
candidate cap `(A1, A2) = (8, 2)` (400 train / 150 val samples drawn fresh, scratch only): median peak
`|q2|` 1.2 rad (p90 2.4, max 14), 6% of trajectories beyond `2 pi` (versus 65% at 15/15 on the same
protocol); floors (pooled / per-trajectory median): train-mean 0.83 / 0.72, ridge linear 0.79 / 0.66,
ridge quadratic 0.78 / 0.67 (versus 0.99 / 0.98, 0.98 / 0.99, 0.97 / 1.06 at 15/15). So that regime is
tame but still clearly nonlinear (simple regressors remove only ~20% of the error), which is a better
place for a network comparison than either "trivially smooth" or "chaotic". One screen, small samples:
confirm before you commit to it. Do not use the test or OOD
split for this kind of exploration.

### 2.7 Input sensitivity: how predictable is the target at late times?
`python -m scripts.input_sensitivity`: perturb *every* input (`m_p`, `b`, `A`, `f`, `phi`) by a relative
0.1%, re-simulate, and measure `||dx|| / std(x)` per trajectory (median / 90th percentile, 40 samples):

| `A1 max`, `A2 max` | t = 0.5 s | t = 1 s | t = 2 s | t = 3 s |
|---|---|---|---|---|
| 15, 15 | 3e-3 / 2e-2 | 6e-3 / 4e-2 | 2.5e-2 / 0.12 | **4.7e-2 / 0.58** |
| 8, 2 | 1.5e-3 / 3.5e-3 | 3.5e-3 / 7.5e-3 | 7e-3 / 2.7e-2 | 1.5e-2 / 0.11 |
| 5, 2 | 1.2e-3 / 2.7e-3 | 3.1e-3 / 7e-3 | 6.7e-3 / 1.6e-2 | 1.3e-2 / 0.05 |

Reading: a 0.1% change of the inputs produces a ~5% median (58% at p90) change of the output at `T = 3 s`
in the current regime, i.e. the input-to-trajectory map amplifies relative changes by ~50x median and
hundreds of times in the tail. A surrogate trained on 1400 samples has an input-space resolution far
coarser than 0.1%, so its late-time error on the tail is bounded below by this sensitivity, not by your
architecture. With the per-joint caps (8, 2) or (5, 2) the amplification is ~13-15x median and ~50-110x at
p90. Use this when you choose `T` and `A`, and when you explain why errors grow with time. (Linear-ish
growth with `t` in the median and an exploding tail is also the signature of the regime switching in 2.5.)

---

## 3. `model.py`: review (read, not rewritten)

What is right: the shape contract `(B, n_t, 4)`; the einsum `"bkp,btkp->btk"` is the correct
contraction; a per-channel `p`-basis is a valid unstacked DeepONet; bias is shared; the two models
share one interface so train/evaluate/benchmark treat them alike. Existing shape tests are good.

Things to question or fix:

1. **The trunk is recomputed for every sample.** `_as_time_batch` expands a shared `(n_t,)` grid to
   `(B, n_t)`, so the trunk MLP runs `B x n_t` times though it never sees `u`. With 1-D `t` it only
   needs `n_t` evaluations, then broadcast. Same output, roughly `B`x less trunk work. Not a bug,
   but it inflates your timing/FLOP numbers, and an examiner may ask "why does the trunk depend on
   the batch?" Your answer should be "it doesn't, and here is the saving".
2. **No test pins down the einsum semantics.** Shape tests pass for a *wrong* contraction as well
   (e.g. `"bkp,btkp->bkt"` transposed then reshaped). Write the loop version
   `out[b,t,k] = sum_p branch[b,k,p] * trunk[t,k,p] + bias[k]` and assert equality.
3. **Time scaling is implicit.** The docstring says the caller normalises `t`. Training and
   benchmark must use the *same* scaling (`benchmark.run` assumes `t / T`). Put the scaling in one
   place and test it.
4. **Phase `phi` is periodic.** If you build a compact input (section 4), feeding `phi` in [0, 2 pi)
   puts `phi = 0.01` and `phi = 6.27` at opposite ends although the torque is almost identical.
   Use `(sin phi, cos phi)`.
5. README promises NaN / negative-mass / wrong-shape rejection. Only shape is checked. Either
   implement it (where? at the model, or at a public `predict()` wrapper) or drop the claim.
6. float32 model vs float64 data: fine, but state it; relative errors below ~1e-6 are noise.

### Overfit-5 smoke test of the existing models (throwaway loop, one seed)
Five train samples (moderate peak `|q|`), channels divided by their std, inputs standardised, `t / T`,
full-batch Adam, `DeepONet(p=32, width=64)` vs `MLPBaseline(width=64, depth=3)`:

| Setting | DeepONet loss / per-sample rel. L2 | MLP loss / per-sample rel. L2 |
|---|---|---|
| lr 1e-3, 3000 steps | 1.6e-2 / 0.06-0.12 | 2.8e-2 / 0.03-0.17 |
| lr 3e-3, 6000 steps | 1.8e-3 / 0.02-0.04 | 3.2e-2 / 0.02-0.24 (loss rose between steps 3000 and 6000) |

Reading: both models run, backpropagate, and reduce the loss, so there is no sign of a structural bug
in `model.py`. But overfitting five samples is *slow* (thousands of steps, loss ~1e-3 not 1e-6), and the MLP
was unstable at lr 3e-3. This is one seed and one setting, so do not read "DeepONet beats MLP" into it.
It does tell you the pass criterion for your own overfit test must be chosen deliberately (steps, lr,
target loss), and that learning-rate choice will matter.

---

## 4. The baseline problem (weak point 2) and what to add

Current `MLPBaseline` eats `2 + 2*151 = 304` redundant inputs plus `t`. A fair report needs, at
minimum, four rows, not two:

| Model | Branch / input | Question it answers |
|---|---|---|
| MLP, 304-d | full torque samples | the "naive" baseline you already have |
| **MLP, 8-d (+ sin/cos phi)** | compact coefficients | **is DeepONet needed at all?** |
| DeepONet, 304-d | full samples | operator framing |
| **DeepONet, compact** | compact coefficients | does the branch/trunk split help when the input is small? |

Give all four the same tuning budget (same number of validation-selected trials). It is a
legitimate and reportable result if the compact MLP wins. Harder torque family (random smooth /
piecewise / GP) is what would make the operator framing earned; it is optional, but decide
consciously: either do it or write "the input is 8-dimensional" in the limitations.

---

## 5. Speed claim: first real numbers (weak point 3)

New `src/benchmark.py` (assistant-mode scaffolding, tested in `tests/test_benchmark.py`). One
trajectory, 151 output times, this machine (CPU, Python 3.10, NumPy 1.26), **untrained** networks
(timing does not depend on weights), 64 test samples:

| Method | ms / trajectory |
|---|---|
| Scalar Python-loop RK4 (`data.simulate`) | ~103 |
| `solve_ivp` RK45, rtol 1e-3 | ~14 |
| `solve_ivp` RK45, rtol 1e-6 | ~44 |
| **Vectorised NumPy RK4, batch of 64** | **~3.9** |
| DeepONet (171k params), batch of 64 | ~0.12 |
| MLP baseline (89k params), batch of 64 | ~0.14 |
| DeepONet, single query | ~0.78 |

The ratio depends strongly on batch size, because NumPy RK4 is dominated by per-step Python overhead
(1200 steps x 4 stages) and amortises over the batch, while the network's cost scales with the batch:

| Batch | Vectorised RK4, ms / traj | DeepONet, ms / traj | Speed-up (RK4 / DeepONet) |
|---|---|---|---|
| 1 | 231 | 1.3 | ~180x (but scalar RK4 is 103 and `solve_ivp` 14-44 ms, so vs those: 80x / 11-35x) |
| 8 | 25.9 | 0.29 | ~90x |
| 64 | 3.7 | 0.11 | ~34x |
| 300 | 1.14 | 0.093 | ~12x |

So: ~850x against the scalar loop is the number you must **not** headline; ~10-35x against batched
vectorised RK4 is the honest range at realistic batch sizes (and it keeps shrinking as the batch grows
or if the RK4 steps/sub-steps are reduced, so state the batch size and `substeps`). A fair single-query
rival is `solve_ivp` (11-35x here). Report these with the break-even count:
`break_even_queries(t_data, t_train, t_solver, t_surrogate)`. Data generation alone is
~0.1 s x 2000 samples = a few minutes, so break-even is in the thousands of queries before you
add training time. Also match `solve_ivp`'s tolerance to the surrogate's *achieved error*, not to
1e-6: a surrogate at 1% error should be compared with `rtol` ~1e-2. Timings are from one run on a
10-thread CPU with a noisy desktop; repeat them under a fixed thread count (`torch.set_num_threads`)
before quoting.

Memory and FLOPs: analytic counts are in `benchmark.py` (`deeponet_flops`, `mlp_baseline_flops`).
One sample, 151 times: DeepONet ~20 MFLOP, MLP baseline ~27 MFLOP (with the un-cached trunk, see
3.1). `peak_memory_mb` only sees the Python heap; PyTorch allocations need RSS from a separate
process if you want to claim them.

---

## 6. Suggested order of work (my recommendation, argue with it)

1. **Decide the data regime** (2.1-2.3): `A` range, `substeps`, extra OOD set. Update README table.
   Regenerate `data/dataset.npz`; rerun `python -m scripts.data_stats`.
2. Write `tests/test_solvers.py` (RK4 order) and the missing payload test; re-run the mutation
   table above and confirm the survivors are now caught.
3. `train.py`: normalisation from train only, per-channel scaling, loss, loop, model selection.
4. `evaluate.py`: per-channel and per-time relative L2, worst-case trajectories, `qdot` vs `d q/dt`
   consistency, energy *balance*.
5. Overfit-5-samples sanity run for every model before any real training.
6. Baselines (section 4), benchmark table, report.

Do not skip 5; it separates "bug" from "model too weak" in minutes.
