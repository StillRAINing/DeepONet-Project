# Start here (written 2026-10-12 while you napped)

Nothing was committed. No crucial code (`dynamics.py`, `solvers.py`, `model.py` bodies, train, evaluate) was changed.
The only edit to your source files: a stale comment in `src/data.py` (gravity torque ~15-24 N m, not ~12).

## The three things that matter most
1. **Dataset regime** (`REVIEW_NOTES.md` section 2). With `A` up to 15 N m: 72% of trajectories spin past 2 pi
   (q2 up to 131 rad), the RK4 truth is unconverged on the worst ones (up to 13% error), the OOD set is calmer
   than train, and peak `|q2|` is bimodal. A per-joint cap such as `(A1, A2) = (8, 2)` looks far saner. Decide, then
   regenerate (`python -m src.data`).
2. **Your tests miss real bugs** (section 1). Parallel-axis terms, RK4 order and the phase range can be broken
   without any test failing. Skeleton tests (skipped) are waiting in `tests/test_solvers.py`,
   `tests/test_dynamics.py`, `tests/test_model.py`: write them.
3. **The speed claim depends on batch size** (section 5): ~850x vs the scalar loop, but ~12-35x vs vectorised RK4.

## What I added (all logged in `AI_LOG.md` rows 8-12; fill in "Checked how" / "My decision")
| Path | What |
|---|---|
| `docs/REVIEW_NOTES.md` | audit findings with numbers |
| `docs/STUDY_GUIDE.md` | questions and hint ladders for every crucial module (no solutions) |
| `docs/FIGURE_PLAN.md` | report figure/table plan with status |
| `src/benchmark.py`, `tests/test_benchmark.py` | timing scaffolding, vectorised RK4, FLOPs, break-even |
| `scripts/` | `data_stats`, `data_figures`, `torque_range_sweep`, `input_sensitivity`, `torque_family_prototype` |
| `tests/` | `test_data`, `test_rk45`, `test_viz`, `test_torque_prototype`, plus skipped skeletons |
| `.github/workflows/tests.yml` | CI (never run on GitHub yet) |
| `results/*.png` | dataset figures and the prototype torque family |

Run tests: `pytest` (36 pass, 13 skipped skeletons when I ran them under Python 3.10 with pytest installed in a scratch
venv; your conda env should be Python 3.11, which I could not test here).

## Suggested first 30 minutes
1. Read `REVIEW_NOTES.md` sections 2.1, 2.2, 2.5 and look at `results/data_peak_vs_inputs.png`.
2. Write your answer to this, unaided, then show me: *"Why does halving the RK4 step cut the global error by 16x, and why
   can't I see that from a single run?"* Then write `test_rk4_global_error_order_is_four`.
3. Decide the `A` caps and `substeps`; update the README table.

## Things I am not sure about
- The mechanism behind the bimodal `q2` (my explanation is a hypothesis; check it).
- Timings are from one noisy run on a 10-thread desktop.
- The `(8, 2)` recommendation rests on small screens (40-400 samples).
- I could not run the project's own conda env (no conda on PATH here); I used Python 3.10 with torch 2.5.1.
