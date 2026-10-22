# DeepONet Surrogate for a 2-Link Planar Robot Arm

MMA3001 (Numerical Methods and Machine Learning), individual project, Monash University, 2026.

## Purpose

A manipulator's dynamics change with its payload and joint friction. Simulating them with a
numerical integrator is accurate but needs one full ODE solve per query. This project trains a
**DeepONet** to act as a fast surrogate for a whole *family* of 2-link planar arms, then checks
honestly how accurate and how fast it is compared with:

- the ground truth: a hand-written **RK4** solver, cross-checked against SciPy `solve_ivp`
- a baseline surrogate: a plain **MLP**

Intended use: fast parameter sweeps (design studies, controller tuning) on a 2-link arm.
This links to the Glados / KUKA arm work as future work.

**Scope limits**
- Planar rigid 2-link arm, vertical plane, gravity on.
- Viscous joint friction only (Coulomb friction is a stretch goal).
- Fixed horizon and fixed initial condition (see below).
- Simulation only. Validation shows the network matches the *model*, not a physical robot.

## Status

Work in progress (final report due Fri 23 Oct 2026). `AI_LOG.md` is the running AI-use log.

- [x] Dynamics + RK4 solver (`src/dynamics.py`, `src/solvers.py`) with unit tests for the arm model
- [x] Solver verification: RK4 convergence order, exact-solution and `solve_ivp` agreement tests
      (`tests/test_solvers.py`); figures and numbers in `results/solver_*.png`, `results/solver_checks.md`
- [x] Dataset generation (seeded, 3001): sine family and the main smooth random-torque family
- [x] DeepONet + MLP baselines (same parameter budget, same tuning protocol)
- [x] Evaluation: validation, test and four out-of-range sets, per-channel / per-time errors
- [x] Input validation (`src/validate.py`, `src/surrogate.py`)
- [x] Benchmark (`results/benchmark.md`, 2026-10-21, EXPERIMENTS B2): quiet rerun with desktop programs closed
- [x] Fair-fight follow-ups (E4): Fourier time features, consistency penalty, data size, basis size p
- [x] Final analysis and every report figure (`results/final_analysis.md`, `docs/FIGURE_PLAN.md`)
- [x] Test report (`results/test_report.xml`, 117 tests), mutation check (`results/mutation_check.md`),
      HTML documentation (`docs/html/`)
- [ ] Written report and AI-use reflection

## Problem definition

### System model

State `x = [q1, q2, q1dot, q2dot]`. `q1` is measured from the horizontal axis, `q2` is relative
to link 1. Euler-Lagrange form:

    M(q) qddot + C(q, qdot) qdot + G(q) + b qdot = tau

Payload `m_p` is a point mass at the tip of link 2, folded into link 2's mass, centre of mass and
inertia (parallel-axis theorem).

### Inputs and outputs

**Main experiment (smooth random-torque family, `data/dataset_smooth.npz`).** Locked decisions are
in `docs/DESIGN_LOG.md` (A6). The torque is a random band-limited function (6 Fourier modes per
joint, correlation length 0.2 in normalised time, rescaled to a random peak). It is finite
dimensional in effect (24 numbers) but is given to the network as 2 x 151 samples; the report makes no
claim about arbitrary input functions. The earlier sinusoidal family (`data/dataset.npz`) is kept for
diagnostics.

| | Definition | Units | Trained domain |
|---|---|---|---|
| Input | Payload mass `m_p` (point mass at the tip of link 2) | kg | [0, 1] |
| Input | Viscous joint friction `b` | N m s/rad | [0.05, 0.5] |
| Input | Joint torques `tau_1(t_i), tau_2(t_i)` sampled on the fixed 151-point grid | N m | peak of `abs(tau)` in [0, 6] N m |
| Input | Query time `t` | s | [0, 3] |
| Output | Joint angles `q1(t), q2(t)` (`q1` from the horizontal, `q2` relative to link 1) | rad | unbounded, no angle wrapping |
| Output | Joint velocities `q1dot(t), q2dot(t)` | rad/s | unbounded |

Array shapes: `params (B, 2)` = `[m_p, b]`; `torque (B, 2, 151)`; output `(B, n_t, 4)` with channels
`[q1, q2, q1dot, q2dot]`. The initial condition is fixed (arm horizontal and at rest, `x0 = 0`) and the
horizon is fixed (3 s), so the map is parameters-to-one-trajectory, not a general dynamics operator.
A "compact" model variant takes the 24 torque coefficients (zero-padded to 52 slots so the rougher `ood_smooth` set fits) instead of the 302 torque samples.

Fixed constants: `l1 = 0.5 m`, `l2 = 0.4 m`, `m1 = 2.0 kg`, `m2 = 1.5 kg`, `lc = l/2`,
`I = m l^2 / 12`, `g = 9.81 m/s^2`.

**Invalid, missing and unsupported inputs** (`src/validate.py`, used by `src/surrogate.py` and by the
simulators `data.simulate` / `data.simulate_smooth`):

| Case | Examples | Behaviour |
|---|---|---|
| Invalid | wrong shape, non-numeric, `None`, NaN or infinite values, `m_p < 0`, `b < 0`, negative torque amplitude or frequency, params and torque with different batch sizes, query time `t < 0` | `ValueError` naming the argument |
| Unsupported (valid physics, outside the trained range) | `m_p` or `b` outside the trained range, torque peak above 6 N m, query time above 3 s | `OutOfRangeWarning` and the prediction is still returned (it is an extrapolation); `strict=True` raises `ValueError` instead |

The simulators only check physical validity (not the trained range) because the out-of-range
evaluation sets are generated with them. Entry point for a trained model:

```python
from src.surrogate import Surrogate
sur = Surrogate.from_checkpoint("checkpoints/<name>.pt")
traj = sur.predict(params, torque)          # (B, 151, 4); sur.predict(..., t=[0.5, 1.2]) for other times
```

## Method

| Component | Role |
|---|---|
| `src/dynamics.py` | Arm model: M(q), C, G, payload combination, state derivative, energy |
| `src/solvers.py` | Fixed-step RK4 (ground truth) and adaptive RK45 |
| `src/data.py` | Seeded dataset generation (sine and smooth torque families), train / val / test splits, OOD sets |
| `src/model.py` | DeepONet (branch: parameters + torque, trunk: query time) and MLP baseline |
| `src/train.py` | Normalisers, relative-L2 loss, training loop with best-validation checkpointing |
| `src/evaluate.py` | Relative L2 (per sample, channel, time), qdot-vs-dq/dt consistency, energy-balance residual |
| `src/validate.py`, `src/surrogate.py` | Input validation and a validated, raw-units prediction interface |
| `src/benchmark.py` | Timing, memory and FLOP helpers, vectorised RK4, `solve_ivp` timing, break-even |
| `src/viz.py` | Interactive pygame visualiser |

**Validation plan**
- Solver: energy conservation (no friction, no torque), energy balance `dE/dt = tau . qdot - b |qdot|^2`,
  pendulum limit, 4th-order convergence under timestep halving, agreement with `solve_ivp`.
- Surrogate: held-out in-range test set; four out-of-range sets (`ood`: heavier payload and friction;
  `ood_amp1`, `ood_amp2`: torque peak 6 to 9 and 9 to 12 N m; `ood_smooth`: rougher torque); error vs
  time and per channel; qdot-vs-dq/dt consistency; energy-balance residual.
- Validation split is used for hyperparameters; the test split is never used for model selection.
  The selection rule was written down before the final test/OOD evaluation (`docs/EXPERIMENTS.md`).
- Label accuracy: the stored RK4 labels are checked against DOP853 at rtol 1e-10 on every split
  (`scripts/truth_check.py`; median error about 1e-8).

## Repository layout

```
src/         source code (see Method)
tests/       pytest unit tests (some solver/model tests are TODO skeletons, reported as skipped)
scripts/     experiments, sweeps, diagnostics and figure scripts (python -m scripts.<name>)
notebooks/   exploration
docs/        design log, experiment log, study guide, hand derivations (docs/physics)
docs/html/   generated API documentation (pdoc), open docs/html/index.html
results/     figures, tables (final_eval.md, benchmark.md, runs.csv), test_report.xml
data/        generated datasets (not committed; regenerate with the recorded seed)
checkpoints/ trained models (not committed)
AI_LOG.md    running log of AI use
```

## Setup

Requires Python 3.11 and conda (or any virtualenv tool).

```bash
conda env create -f environment.yml
conda activate mma3001-deeponet
```

Without conda:

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch runs on CPU only; no GPU is needed.

## Reproduce

Datasets and checkpoints are not committed (large binaries). Everything is regenerated from seed 3001.

```bash
pytest --junitxml=results/test_report.xml   # tests + test report
python -m src.data --smooth                 # main dataset into data/ (about 10,000 RK4 trajectories)
python -m src.data                          # sine-family dataset (diagnostics)
bash scripts/run_sweep.sh                   # the fair-comparison sweep (docs/EXPERIMENTS.md)
python -m scripts.train_quick --model deeponet --epochs 200   # train (or reload) one model into checkpoints/
python -m scripts.summarise_runs            # every checkpoint on val -> results/runs.csv
python -m scripts.final_eval                # the 12 final checkpoints, once, on test and OOD -> results/final_eval.md
python -m scripts.final_eval --ff6 / --ff6cons   # the E4 follow-up variants on test and OOD (post hoc)
python -m scripts.run_benchmark             # speed / memory / FLOPs / break-even -> results/benchmark.md (idle machine!)
python -m scripts.final_analysis            # every remaining report figure and table -> results/final_analysis.md
python -m scripts.solver_figures            # RK4 convergence, energy, labels vs DOP853 -> results/solver_checks.md
python -m scripts.mutation_check            # do the tests catch deliberate bugs? -> results/mutation_check.md
python -m scripts.plot_training             # loss curves -> results/training_curves/loss_*.png
python -m scripts.plot_eval checkpoints/<a>.pt [<b>.pt ...]   # evaluation figures on val
python -m scripts.truth_check               # accuracy of the stored RK4 labels on every split
python -m scripts.data_stats                # dataset statistics (per-channel scales, wrapping)
python -m scripts.input_sensitivity         # sensitivity of trajectories to tiny input changes
pdoc src -o docs/html -d google --no-show-source   # HTML documentation
python -m src.viz                           # interactive visualiser
```

Random seed: 3001 (`SEED` in `src/data.py`). The experiments in this repository were run with Python
3.10 and PyTorch 2.5.1 (CPU used for the timing results).

## Results

Selected models, about 171,000 parameters each, 200 epochs, mean over training seeds 3001, 1, 2
(full tables with p95, max, per-channel and consistency: `results/final_eval.md`; protocol and
analysis: `docs/EXPERIMENTS.md`). Error is the median per-sample relative L2, channels scaled by the
train std.

| Model | Input | Test median | Test p95 | Test under 10 % | OOD (`m_p`, `b`) median | OOD torque 6 to 9 N m median |
|---|---|---|---|---|---|---|
| MLP | compact (24 coefficients) | **0.071** | **0.600** | **58.2 %** | **0.052** | 0.53 |
| MLP | 302 torque samples | 0.080 | 0.603 | 55.8 % | 0.052 | 0.54 |
| DeepONet | compact | 0.109 | 0.653 | 47.3 % | 0.060 | 0.62 |
| DeepONet | 302 torque samples | 0.130 | 0.630 | 39.3 % | 0.065 | 0.61 |

Summary: at equal parameters and tuning, the MLP is the more accurate surrogate in distribution; the
DeepONets have the smaller qdot-vs-dq/dt mismatch (about 0.11 to 0.15 vs 0.22 to 0.24); all models fail
on the hardest 5 to 10 % of trajectories (the spinning regime) and on torque amplitudes above the
training range. Label error (about 1e-8) is far below every surrogate error.

Follow-ups (E4, chosen on validation, evaluated on test afterwards, `docs/EXPERIMENTS.md`): Fourier time
features help the sample-input models (DeepONet test median 0.130 to 0.105) but do not close the gap; adding
a qdot = dq/dt penalty makes both families self-consistent (DeepONet 0.008, MLP 0.12) at no accuracy cost.
The best configuration found is the MLP with samples input, Fourier features and the penalty (test median
0.066 +- 0.002). Median error falls about as n^-0.5 with training-set size; the p95 much more slowly.

Speed (`results/benchmark.md`, 64 validation trajectories): vectorised NumPy RK4 gives the exact labels in
4.4 ms per trajectory at batch 64 (0.73 ms at batch 1500); `solve_ivp` at the surrogates' error level needs
about 7.5 to 8 ms and fails on some trajectories at rtol 0.08 to 0.1. The surrogates cost 0.14 to 0.28 ms at
batch 64; a DeepONet with its trunk cached on the fixed time grid costs about 0.02 ms, which the MLP cannot copy.
Including data generation (8 s vectorised) and one training run (560 to 1500 s), the surrogates pay off
after about 70,000 to 205,000 queries against `solve_ivp`, and 0.8 to 3.0 million against batched RK4
(EXPERIMENTS B2, quiet rerun 2026-10-21).

## Limitations

- Synthetic data from a model; no physical robot validation. The network matches the *model*.
- Planar 2-link arm, viscous friction only, fixed horizon (3 s) and fixed initial condition.
- The torque input is a low-dimensional random function (about 24 numbers) shown to the network as
  samples; no claim about arbitrary input functions.
- Neither family learns the spinning regime: p95 error is 0.6 to 0.65 and the worst trajectories are
  worse than predicting zero. Amplitude extrapolation fails for every model (median error 0.5 to 0.8).
- The `m_p`/`b` out-of-range set is mild (calmer trajectories); it tests a small smooth shift.
- One data split, three training seeds, one model size (about 171k parameters); the data-size and p
  sweeps are single-seed.
- Timings are from one laptop with some background desktop load; absolute milliseconds may be 20 to 30 %
  high, ratios are less affected.
- The test set has also been used to report two post-hoc variants (ff6, ff6 + consistency); the headline
  numbers are the rule-selected models.
- Out-of-range inputs are flagged with a warning, not corrected: the surrogate cannot extrapolate.

## AI use

AI tools were used in this project. Tool, task, contribution level, how the output was verified,
errors found and my own decisions are recorded in [`AI_LOG.md`](AI_LOG.md).

## Licence

MIT, see `LICENSE`.
