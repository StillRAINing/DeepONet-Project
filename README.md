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

Work in progress. See `AI_LOG.md` for the running AI-use log.

- [ ] Dynamics + RK4 solver, with unit tests
- [ ] Solver verification (energy, convergence order, pendulum limit, `solve_ivp`)
- [ ] Dataset generation (seeded)
- [ ] DeepONet + MLP baseline
- [ ] Evaluation (in-range and out-of-range)
- [ ] Benchmarking (runtime, memory, FLOPs)

## Problem definition

### System model

State `x = [q1, q2, q1dot, q2dot]`. `q1` is measured from the horizontal axis, `q2` is relative
to link 1. Euler-Lagrange form:

    M(q) qddot + C(q, qdot) qdot + G(q) + b qdot = tau

Payload `m_p` is a point mass at the tip of link 2, folded into link 2's mass, centre of mass and
inertia (parallel-axis theorem).

### Inputs and outputs

**Decisions locked (10 Oct 2026):** vertical plane with gravity on; sinusoidal torque family
(random amplitude, frequency and phase per joint). Parameter ranges, horizon and initial
condition are still provisional. Update this table when they are final.

| | Definition | Units | Domain (provisional) |
|---|---|---|---|
| Input | Payload mass `m_p` | kg | [0, 1] |
| Input | Viscous friction `b` | N m s/rad | [0.05, 0.5] |
| Input | Torque profile samples `tau_j(t_i) = A_j sin(2 pi f_j t_i + phi_j)`, j = 1, 2, fixed grid | N m | A in [0, 15] N m, f in [0.2, 1.0] Hz, phi in [0, 2 pi) (provisional) |
| Input | Query time `t` | s | [0, T], T = 2 to 4 s (TBD) |
| Output | Joint angles `q1(t), q2(t)` | rad | unbounded |
| Output | Joint velocities `q1dot(t), q2dot(t)` | rad/s | unbounded |

Fixed constants (placeholders): `l1 = 0.5 m`, `l2 = 0.4 m`, `m1 = 2.0 kg`, `m2 = 1.5 kg`,
`lc = l/2`, `I = m l^2 / 12`, `g = 9.81 m/s^2`.

**Invalid inputs:** parameters outside the trained range raise a clear error or warning;
NaN, negative mass and wrong array shapes are rejected. (To be implemented and documented.)

## Method

| Component | Role |
|---|---|
| `src/dynamics.py` | Arm model: M(q), C, G, payload combination, state derivative, energy |
| `src/solvers.py` | Fixed-step RK4 |
| `src/data.py` | Seeded dataset generation and train / validation / test splits |
| `src/model.py` | DeepONet (branch: parameters + torque samples, trunk: query time) and MLP baseline |
| `src/train.py` | Training loop and logging |
| `src/evaluate.py` | Relative L2 error, error vs time, energy drift |
| `src/benchmark.py` | Runtime, memory and FLOP estimates |

**Validation plan**
- Solver: energy conservation (no friction, no torque), energy balance `dE/dt = tau . qdot - b |qdot|^2`,
  pendulum limit, 4th-order convergence under timestep halving, agreement with `solve_ivp`.
- Surrogate: held-out in-range test set, out-of-range extrapolation set (heavier payload, higher
  friction), error vs time, physical-consistency (energy drift) of predicted trajectories.
- Validation split is used for hyperparameters; the test split is never used for model selection.

## Repository layout

```
src/         source code (see Method)
tests/       pytest unit tests
notebooks/   exploration and plots
docs/        generated HTML docs (pdoc)
results/     figures, tables, test report
data/        generated datasets (not committed; regenerate with the recorded seed)
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

The entry points below are the planned interface and will be finalised as each module is written.

```bash
pytest                                      # run all tests
pytest --junitxml=results/test_report.xml   # save the test report
python -m src.data                          # generate the seeded dataset into data/
python -m src.train                         # train DeepONet and MLP
python -m src.evaluate                      # metrics and figures into results/
python -m src.benchmark                     # timing, memory, FLOPs
pdoc src -o docs                            # build HTML docs
```

Random seed: TBD (will be recorded here and in `src/data.py`).

## Results

To be filled in. Planned table:

| Method | Rel. L2 error (in-range) | Rel. L2 error (out-of-range) | Params | Train time | Time / trajectory |
|---|---|---|---|---|---|
| RK4 (ground truth) | n/a | n/a | n/a | n/a | TBD |
| `solve_ivp` | TBD | TBD | n/a | n/a | TBD |
| MLP baseline | TBD | TBD | TBD | TBD | TBD |
| DeepONet | TBD | TBD | TBD | TBD | TBD |

## Limitations

- Synthetic data from a model; no physical robot validation.
- Planar 2-link arm, viscous friction only, fixed horizon and initial condition.
- More to be added from the results.

## AI use

AI tools were used in this project. Tool, task, contribution level, how the output was verified,
errors found and my own decisions are recorded in [`AI_LOG.md`](AI_LOG.md).

## Licence

MIT, see `LICENSE`.
