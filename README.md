# DeepONet Surrogate for a 2-Link Planar Robot Arm

MMA3001 (Numerical Methods and Machine Learning), individual project, Monash University, 2026.

## Purpose

Learn a fast surrogate for the dynamics of a 2-link planar robot arm using a DeepONet, and
compare it against a plain MLP baseline and a trusted numerical solver (RK4, cross-checked
against SciPy `solve_ivp`).

The surrogate learns the operator

    (payload mass m_p, joint friction b, torque profile tau(t)) -> joint trajectory q(t), qdot(t)

so one trained network answers queries for a whole family of arms instead of one ODE solve per query.

**Scope limits:** planar rigid 2-link arm, vertical plane, gravity on, viscous friction only,
simulation only. Validation shows the network matches the *model*, not a physical robot.

## Status

Work in progress. See `AI_LOG.md` for the running AI-use log.

- [ ] Dynamics + RK4 solver, with unit tests
- [ ] Solver verification (energy, convergence order, pendulum limit, `solve_ivp`)
- [ ] Dataset generation (seeded)
- [ ] DeepONet + MLP baseline
- [ ] Evaluation (in-range and out-of-range)
- [ ] Benchmarking (runtime, memory, FLOPs)

## Inputs and outputs

TODO: fill in with final units, domains and ranges once locked.

| | Definition |
|---|---|
| Inputs | `m_p` (kg), `b` (N m s/rad), torque samples `tau(t_i)` (N m), query time `t` (s) |
| Outputs | `q1, q2` (rad), `qdot1, qdot2` (rad/s) at the query times |

## Repository layout

```
src/         dynamics.py, solvers.py, data.py, model.py, train.py, evaluate.py, benchmark.py
tests/       pytest unit tests (saved report goes in results/)
notebooks/   exploration and plots
docs/        generated HTML docs (pdoc)
results/     figures, tables, test report
data/        generated datasets (not committed; regenerate with src/data.py and the recorded seed)
AI_LOG.md    running log of AI use
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

TODO: exact commands for data generation, training, evaluation and benchmarking.

```bash
pytest                         # run tests
pytest --junitxml=results/test_report.xml
pdoc src -o docs               # build HTML docs
```

## Results

TODO.

## Licence

MIT, see `LICENSE`.
