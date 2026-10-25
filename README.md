# DeepONet Surrogate for a 2-Link Planar Robot Arm

MMA3001 (Numerical Methods and Machine Learning), individual project, Monash University, 2026.
Author: Rain Lai. Licence: MIT (`LICENSE`).

A DeepONet is trained as a fast surrogate for the dynamics of a *family* of 2-link planar arms
(varying payload, joint friction and joint torque history), and compared honestly against a plain
MLP surrogate and against the numerical solvers it is meant to replace (RK4, `solve_ivp`).

**Headline result.** At equal parameter count and equal tuning effort, the MLP is the more accurate
surrogate (test median relative L2 0.071 vs 0.109 for the best DeepONet); the DeepONet is the less
self-inconsistent one (qdot vs dq/dt mismatch about 0.11 to 0.15 vs 0.22 to 0.24) and the cheapest per
query once its trunk is cached. Every model fails on the hardest 5 to 10 % of trajectories (the spinning
regime) and on torques above the training range. Details in [Results](#results).

## Contents

- [Where each brief requirement is](#where-each-brief-requirement-is)
- [Quick start for a reviewer](#quick-start-for-a-reviewer)
- [Engineering problem](#engineering-problem)
- [Inputs and outputs](#inputs-and-outputs)
- [Method](#method)
- [Alternatives compared](#alternatives-compared)
- [Validation](#validation)
- [Results](#results)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproduce](#reproduce)
- [Documentation map](#documentation-map)
- [AI use](#ai-use)
- [Licence](#licence)

## Where each brief requirement is

| Brief item | Where |
|---|---|
| ReadMe | this file |
| HTML documentation of the source code (pdoc) | [`docs/html/index.html`](docs/html/index.html) (open in a browser) |
| Licensing agreement | [`LICENSE`](LICENSE) (MIT) |
| Source code, inline comments and docstrings | [`src/`](src/) (functional `main` branch) |
| Test functions and test report (pytest) | [`tests/`](tests/), [`results/test_report.html`](results/test_report.html), [`results/test_report.xml`](results/test_report.xml) |
| Reproducible workflow, version control | [Setup](#setup), [Reproduce](#reproduce), fixed seed 3001, CI in [`.github/workflows/tests.yml`](.github/workflows/tests.yml) |
| 1. Engineering problem | [Engineering problem](#engineering-problem) |
| 2. Inputs and outputs, invalid input handling | [Inputs and outputs](#inputs-and-outputs); function-level inputs and outputs in the docstrings / `docs/html/` |
| 3. Computational solution | [Method](#method), [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) |
| 4. Alternative solutions | [Alternatives compared](#alternatives-compared), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) E0, E3 |
| 5. Validation | [Validation](#validation), [`results/solver_checks.md`](results/solver_checks.md), [`results/final_eval.md`](results/final_eval.md), [`results/mutation_check.md`](results/mutation_check.md) |
| 6. Optimisation and performance | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) E1 to E4 and B2, [`results/benchmark.md`](results/benchmark.md) (timing, memory, FLOPs, break-even) |
| 7. AI use and reflection | [`AI_LOG.md`](AI_LOG.md) |
| Written report (5 to 10 pages) | submitted separately on Moodle; not part of this repository |

## Quick start for a reviewer

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
pytest                              # about 1 minute on a laptop CPU, no data or checkpoints needed
```

On a fresh clone `pytest` gives 116 passed and 1 skipped: the skipped test
(`test_demo_registers_all_four_selected_models_and_they_track_rk4`) needs the trained checkpoints, which are
not committed. With the checkpoints present all 117 pass (`results/test_report.html`).

Then open `docs/html/index.html` (API documentation) and `results/INDEX.md` (every figure and table, and the
script that made it).

## Engineering problem

**Context.** A manipulator's dynamics change with its payload and joint friction. Design studies,
controller tuning and payload or friction sweeps need the arm's motion for thousands of combinations of
these parameters and of the commanded joint torques.

**Problem.** A numerical integrator gives accurate answers but needs one full ODE solve per query. The
question is whether a trained neural operator (DeepONet) can replace the solver for a whole family of
arms, how accurate it is, when it is actually faster once data generation and training are counted, and
whether it beats a simpler learned baseline.

**Why it matters.** A surrogate is only worth building if its error is acceptable *and* it pays back its
up-front cost. Both are often overstated (comparison against a slow solver, no baseline, error averaged
over easy cases). This project measures both against fair rivals.

**Intended use.** Fast parameter sweeps on a 2-link arm in the trained range, where a few percent error
is acceptable (screening, not final verification).

**Scope limits.**
- Planar rigid 2-link arm in a vertical plane, gravity on.
- Viscous joint friction only (no Coulomb friction).
- Fixed 3 s horizon and fixed initial condition (arm horizontal and at rest).
- Simulation only: the validation shows the network matches the *model*, not a physical robot.

## Inputs and outputs

### System model

State `x = [q1, q2, q1dot, q2dot]`. `q1` is measured from the horizontal axis, `q2` is relative
to link 1. Euler-Lagrange form:

    M(q) qddot + C(q, qdot) qdot + G(q) + b qdot = tau

Payload `m_p` is a point mass at the tip of link 2, folded into link 2's mass, centre of mass and
inertia (parallel-axis theorem). Diagram: `results/arm_schematic.png`.

Fixed constants: `l1 = 0.5 m`, `l2 = 0.4 m`, `m1 = 2.0 kg`, `m2 = 1.5 kg`, `lc = l/2`,
`I = m l^2 / 12`, `g = 9.81 m/s^2`.

### Top-level inputs and outputs

**Main experiment: smooth random-torque family (`data/dataset_smooth.npz`).** Settings in
`docs/DESIGN_LOG.md` A6. The torque is a random band-limited function (6 Fourier modes per joint,
correlation length 0.2 in normalised time, rescaled to a random peak). It is finite-dimensional in effect
(24 numbers) but is given to the network as 2 x 151 samples; no claim is made about arbitrary input
functions. The earlier sinusoidal family (`data/dataset.npz`) is kept for diagnostics only.

| | Definition | Units | Trained domain |
|---|---|---|---|
| Input | Payload mass `m_p` (point mass at the tip of link 2) | kg | [0, 1] |
| Input | Viscous joint friction `b` | N m s/rad | [0.05, 0.5] |
| Input | Joint torques `tau_1(t_i), tau_2(t_i)` sampled on the fixed 151-point grid | N m | peak of `abs(tau)` in [0, 6] N m |
| Input | Query time `t` | s | [0, 3] |
| Output | Joint angles `q1(t), q2(t)` (`q1` from the horizontal, `q2` relative to link 1) | rad | unbounded, no angle wrapping |
| Output | Joint velocities `q1dot(t), q2dot(t)` | rad/s | unbounded |

Array shapes: `params (B, 2)` = `[m_p, b]`; `torque (B, 2, 151)`; output `(B, n_t, 4)` with channels
`[q1, q2, q1dot, q2dot]`. Because the initial condition (`x0 = 0`) and the horizon (3 s) are fixed, the map
is parameters-to-one-trajectory, not a general dynamics operator. A "compact" model variant takes the 24
torque coefficients (zero-padded to 52 slots so the rougher `ood_smooth` set fits) instead of the 302
torque samples.

Meaning and use of the outputs: the predicted trajectory replaces an RK4 solve for screening studies
(peak joint angles, settling, how payload and friction change the motion). The error to expect is in
[Results](#results); outside the trained domain the prediction is an extrapolation and should not be used.

### Invalid, missing and unsupported inputs

Handled by `src/validate.py`, used by `src/surrogate.py` and by the simulators `data.simulate` /
`data.simulate_smooth`:

| Case | Examples | Behaviour |
|---|---|---|
| Invalid | wrong shape, non-numeric, `None`, NaN or infinite values, `m_p < 0`, `b < 0`, negative torque amplitude or frequency, params and torque with different batch sizes, query time `t < 0` | `ValueError` naming the argument |
| Unsupported (valid physics, outside the trained range) | `m_p` or `b` outside the trained range, torque peak above 6 N m, query time above 3 s | `OutOfRangeWarning` and the prediction is still returned (it is an extrapolation); `strict=True` raises `ValueError` instead |

The simulators only check physical validity (not the trained range), because RK4 is valid for any physical
input and the out-of-range evaluation sets are generated with them. Entry point for a trained model:

```python
from src.surrogate import Surrogate
sur = Surrogate.from_checkpoint("checkpoints/<name>.pt")
traj = sur.predict(params, torque)          # (B, 151, 4); sur.predict(..., t=[0.5, 1.2]) for other times
```

Function-level inputs and outputs (shapes, units, exceptions) are in each function's docstring and in
the generated documentation `docs/html/`.

## Method

| Component | Role |
|---|---|
| `src/dynamics.py` | Arm model: M(q), C, G, payload combination, state derivative, energy |
| `src/solvers.py` | Fixed-step RK4 (ground truth, h = 2.5 ms, 8 substeps per 20 ms output step) and adaptive RK45 |
| `src/data.py` | Seeded dataset generation (sine and smooth torque families), train / val / test splits, OOD sets |
| `src/model.py` | DeepONet (branch: parameters + torque, trunk: query time; unstacked, with bias) and MLP baseline |
| `src/train.py` | Normalisers (train statistics only), relative-L2 loss, training loop with best-validation checkpointing |
| `src/evaluate.py` | Relative L2 (per sample, channel, time), qdot-vs-dq/dt consistency, energy-balance residual |
| `src/validate.py`, `src/surrogate.py` | Input validation and a validated, raw-units prediction interface |
| `src/benchmark.py` | Timing, memory and FLOP helpers, vectorised RK4, `solve_ivp` timing, break-even |
| `src/viz.py`, `src/demo.py` | Interactive pygame visualiser; live RK4 vs the four selected surrogates |

Training protocol (every model): 7,000 train / 1,500 val / 1,500 test trajectories, per-channel z-scored
outputs, z-scored inputs, time scaled to `t / T`, Adam, batch 100, 200 epochs, learning rate and schedule
chosen per model on validation, about 171,000 parameters per model. Full protocol and the reasons for each
choice: `docs/EXPERIMENTS.md` ("Fixed protocol") and `docs/DESIGN_LOG.md`.

## Alternatives compared

| Alternative | Role | Where |
|---|---|---|
| MLP on the 302 torque samples + `t` | learned baseline, same input as the DeepONet | EXPERIMENTS E3 |
| MLP on the 24 compact coefficients + `t` | learned baseline without the redundant input | EXPERIMENTS E3 |
| DeepONet on the compact coefficients | isolates the effect of the input encoding | EXPERIMENTS E3 |
| Mean training trajectory, 1-nearest neighbour | naive yardsticks (what learning nothing scores) | EXPERIMENTS E0, `results/final_analysis.md` |
| Vectorised NumPy RK4 (exact) | the fair speed rival | EXPERIMENTS B2, `results/benchmark.md` |
| SciPy `solve_ivp` (RK45) at a tolerance matched to the surrogate error | the fair single-query speed rival | EXPERIMENTS B2 |

All four networks have about 171k parameters and the same tuning grid and selection rule, so a win
cannot be explained by size or tuning effort.

## Validation

- **Solver (ground truth):** energy conservation (no friction, no torque), energy balance
  `dE/dt = tau . qdot - b |qdot|^2`, pendulum limit, 4th-order convergence under step halving, agreement
  with `solve_ivp` / DOP853 (`tests/test_solvers.py`, `results/solver_checks.md`, `results/solver_*.png`).
- **Labels:** stored RK4 labels checked against DOP853 at rtol 1e-10 on every split (`scripts/truth_check.py`;
  median error about 1e-8, far below every surrogate error).
- **Code:** 117 unit tests; a mutation check shows 9 of 10 deliberate bugs are caught by a targeted test
  (`results/mutation_check.md`).
- **Surrogate:** held-out in-range test set; four out-of-range sets (`ood`: heavier payload and friction;
  `ood_amp1`, `ood_amp2`: torque peak 6 to 9 and 9 to 12 N m; `ood_smooth`: rougher torque); error per
  channel, per time and per regime; qdot-vs-dq/dt consistency; energy-balance residual.
- **No leakage:** the validation split is used for every choice; the selection rule was written down
  before the results (`docs/EXPERIMENTS.md`, "Selection rule"), and the test and OOD sets were evaluated
  once, after it.

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
- Planar 2-link arm, viscous friction only, fixed horizon (3 s) and fixed initial condition; the initial
  condition is not enforced by the networks (error at `t = 0` is 0.3 to 6 % of the channel rms).
- The torque input is a low-dimensional random function (about 24 numbers) shown to the network as
  samples; no claim about arbitrary input functions.
- Neither family learns the spinning regime: p95 error is 0.6 to 0.65 and the worst trajectories are
  worse than predicting zero. Amplitude extrapolation fails for every model (median error 0.5 to 0.8).
- The `m_p`/`b` out-of-range set is mild (calmer trajectories); it tests a small smooth shift.
- One data split, three training seeds, one model size (about 171k parameters); the data-size and p
  sweeps are single-seed; width and depth were not swept.
- Timings are from one laptop and one benchmark run (B2, desktop programs closed); repeated runs on this
  machine differ by about 10 to 20 %.
- The test set has also been used to report two post-hoc variants (ff6, ff6 + consistency); the headline
  numbers are the rule-selected models.
- Out-of-range inputs are flagged with a warning, not corrected: the surrogate cannot extrapolate.

## Repository layout

```
src/                 source code (see Method)
tests/               pytest unit tests (117; results/test_report.html)
scripts/             experiments, sweeps, diagnostics and figure scripts (python -m scripts.<name>)
docs/                DESIGN_LOG.md, EXPERIMENTS.md, REVIEW_NOTES.md (see Documentation map)
docs/html/           generated API documentation (pdoc); open docs/html/index.html
results/             figures and tables; every file and the script that writes it: results/INDEX.md
results/report/      the report figures, redrawn at print size (scripts/report_figures.py)
results/training_curves/  loss curves (scripts/plot_training.py)
data/                generated datasets (not committed; regenerate with seed 3001)
checkpoints/         trained models (not committed; regenerate with the sweep scripts)
.github/workflows/   CI: runs the test suite on every push
AI_LOG.md            log of all AI use (brief section 7)
CLAUDE.md            the instructions given to the AI assistant (kept as part of the AI-use record)
LICENSE              MIT licence
environment.yml, requirements.txt, pytest.ini   environment and test configuration
```

## Setup

Python 3.11 (as in CI and `environment.yml`); the experiments were run with Python 3.10. Conda or any
virtualenv tool.

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

Datasets and checkpoints are not committed (large binaries). Everything is regenerated from seed 3001
(`SEED` in `src/data.py`). Run the commands from the repository root, in this order:

```bash
# 0. Tests and test report
pytest --junitxml=results/test_report.xml                         # tests + XML report
pytest --html=results/test_report.html --self-contained-html      # HTML report (pytest-html)

# 1. Data
python -m src.data --smooth                 # main dataset into data/ (10,000 RK4 trajectories + OOD sets)
python -m src.data                          # sine-family dataset (diagnostics only)

# 2. Training and model selection (docs/EXPERIMENTS.md)
bash scripts/run_sweep.sh                   # the fair-comparison sweep (E3)
bash scripts/run_seeds.sh <model> <input> <lr> <schedule>   # selection-rule step 2, per selected setting
python -m scripts.summarise_runs            # every checkpoint on val -> results/runs.csv, runs_agg.csv
python -m scripts.train_quick --model deeponet --epochs 200   # train (or reload) a single model

# 3. Final evaluation (once, after the selection rule)
python -m scripts.final_eval                # 12 final checkpoints on test and OOD -> results/final_eval.md
python -m scripts.final_eval --ff6          # E4 follow-up variants (post hoc)
python -m scripts.final_eval --ff6cons

# 4. Analysis, benchmark and figures
python -m scripts.run_benchmark             # speed / memory / FLOPs / break-even -> results/benchmark.md (idle machine)
python -m scripts.final_analysis            # remaining tables and figures -> results/final_analysis.md
python -m scripts.solver_figures            # RK4 convergence, energy, labels vs DOP853 -> results/solver_checks.md
python -m scripts.report_figures            # report Figures 2 to 8 at print size -> results/report/
python -m scripts.mutation_check            # do the tests catch deliberate bugs? -> results/mutation_check.md
python -m scripts.plot_training             # loss curves -> results/training_curves/
python -m scripts.plot_eval checkpoints/<a>.pt [<b>.pt ...]   # evaluation figures on val
python -m scripts.truth_check               # accuracy of the stored RK4 labels on every split
python -m scripts.data_stats                # dataset statistics (per-channel scales, wrapping)
python -m scripts.input_sensitivity         # sensitivity of trajectories to tiny input changes
python -m scripts.arm_schematic             # arm schematic -> results/arm_schematic.png

# 5. Documentation and demo
pdoc src -o docs/html -d google --no-show-source   # HTML documentation
python -m src.viz                           # interactive visualiser
python -m src.demo                          # RK4 vs the four selected models (needs data and checkpoints)
```

The experiments were run with Python 3.10 and PyTorch 2.5.1 (CPU used for all timing results). The
documentation logs call the interpreter `py -3.10` (Windows launcher); `python` is equivalent.

## Documentation map

| File | What it is |
|---|---|
| [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) | design decisions, the reasoning behind them, and the status of each (section F) |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | every experiment: protocol, selection rule, sweeps, final evaluation, benchmark, verification |
| [`docs/REVIEW_NOTES.md`](docs/REVIEW_NOTES.md) | historical code and data audit (2026-10-12) that led to the smooth torque family; banner says what became of each finding |
| [`results/INDEX.md`](results/INDEX.md) | every results file, the report section it supports and the script that writes it |
| [`AI_LOG.md`](AI_LOG.md) | AI-use log and reflection notes |
| [`docs/html/index.html`](docs/html/index.html) | API documentation of `src/` |

The logs are dated working records: numbers in an older entry can be superseded by a later one, and each
superseded entry says so. The headline numbers are the ones in this README and in `results/final_eval.md`.

## AI use

AI tools (Claude, Anthropic) were used throughout this project. Tool, task, contribution level, how the
output was verified, errors found and my own decisions are recorded in [`AI_LOG.md`](AI_LOG.md); the
instructions given to the assistant are in [`CLAUDE.md`](CLAUDE.md).

## Licence

MIT, see [`LICENSE`](LICENSE).
