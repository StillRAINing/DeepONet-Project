# AI Use Log — MMA3001 Project

Running record of AI use for the report's AI reflection (brief section 7).
Keep it updated **as you work**. Fill in the blanks in the "Checked how" and "My decision" columns yourself; those are the parts the rubric marks.

**Tool(s) used:** Claude (Anthropic), Sonnet 5.5, via Claude Code / claude.ai Projects.

## How to use this log

For each AI interaction that shapes the project, add a row below:

| Field | What to write |
|---|---|
| Date | When |
| Task | What you asked AI to do |
| AI contribution | None / low / medium / high, and what exactly |
| Why AI | Why you used it for this task |
| Checked how | What you did to verify the output (derivation, test, comparison, reading docs). **Be specific.** |
| Errors / unhelpful output | Anything wrong, misleading or unusable. Write these down even if small. |
| My decision | What you decided yourself, and where you departed from the AI's suggestion |

Disclosure doesn't replace verification. You are responsible for all code, results and claims.

---

## Log

### 2026-10-10 (Sat)

| # | Task | AI contribution | Why AI | Checked how | Errors / unhelpful output | My decision |
|---|---|---|---|---|---|---|
| 1 | Read the project brief PDF and summarise requirements, deliverables and rubric | **High** for the summary. Source is the brief itself. | Quick orientation on a 15-page document | [ ] Re-read brief pages 1-9 myself and confirm the summary matches | [fill in] | [fill in] |
| 2 | Discuss project scope: whether a physical robot is needed, whether a dataset is needed, whether a 2-link arm is too simple | **Medium.** AI proposed options and trade-offs. Project idea (DeepONet on a robot arm) was already mine. | Sanity-check scope against the brief's "not about complexity" emphasis | [ ] Check against the brief (simulation allowed, validation required) and, ideally, the Unit Coordinator's answer | [fill in] | [fill in: e.g. vertical plane vs horizontal, torque input family] |
| 3 | Placeholder simulation + figure of a 2-link arm (`arm_demo.py`, SciPy `solve_ivp`) | **High.** Script and figure written entirely by AI. Throwaway illustration, **not part of the repo, not validated.** Parameters are invented. | See what the system looks like | [ ] Re-derive M(q), C, G and the payload parallel-axis formula by hand. Not yet done. Write unit tests (energy conservation, pendulum limit). | [fill in] | Will write my own `dynamics.py` / RK4 for the repo |
| 4 | Draft the 14-day project plan (`MMA3001_project_plan.md`) | **High** for structure and wording. Dates and parameter ranges are placeholders. | Plan structure and rubric mapping | [ ] Check dates against the Unit Overview Spreadsheet. Adjust timeline to my own schedule. | Dates are assumed, not from the brief. | [fill in] |
| 5 | Summarise the brief's AI-use policy and start this log | **Medium.** Policy is from the brief (section 7). Log format is AI-suggested. | Clarify what disclosure requires | [ ] Compare with brief section 7 | [fill in] | [fill in] |
| 6 | Scaffold the GitHub repo (README, LICENSE, .gitignore, requirements, pytest config, empty `src/` modules, smoke test) | **High.** Boilerplate only, no project logic. | Save time on setup | [ ] Check README layout matches the brief's repo requirements; fill in LICENSE name | [fill in] | [fill in] |
| 7 | Interactive pygame visualiser (`src/viz.py`): arm animation, parameter sliders, RK4/RK45 overlay, surrogate hook | **High.** UI code only; physics is imported from `dynamics.py`, `solvers.py`, `data.py`. | Explore the model interactively and make report figures | [ ] Run it; compare RK4 vs RK45 overlay; check energy plot against `energy()` | [fill in] | [fill in] |

### 2026-10-12 (Mon), while away from keyboard

| # | Task | AI contribution | Why AI | Checked how | Errors / unhelpful output | My decision |
|---|---|---|---|---|---|---|
| 8 | Independent audit of `dynamics.py` (geometric T/V reference, finite-difference M, G, Coriolis power identity) and mutation test of the existing test suite | **Medium.** AI wrote throwaway scripts and the findings in `docs/REVIEW_NOTES.md` sections 1. No repo code changed. | A second opinion on correctness and on what the tests miss | [ ] Re-run the audit idea yourself; re-derive M, C, G by hand; reproduce the mutation survivors | [fill in] | [fill in] |
| 9 | Dataset diagnostics: `scripts/data_stats.py`, `scripts/torque_range_sweep.py`, `scripts/data_figures.py`, `scripts/input_sensitivity.py`; findings in `REVIEW_NOTES.md` section 2 (spinning, input sensitivity, reference floors, unconverged RK4 tail, calmer OOD set, bimodal peak q2) | **High** for scripts and write-up. Numbers are from the project's own dataset. | Quickly see whether the data regime is sensible before training | [ ] Re-run each script; check at least one number by hand; decide the A caps and `substeps` yourself | Note: my first caveat ('low A makes torque irrelevant') was contradicted by the sweep | [fill in: A range, substeps, extra OOD set] |
| 10 | `.github/workflows/tests.yml` (CI, not yet run on GitHub), `src/benchmark.py` (vectorised RK4, solve_ivp timing, FLOP/break-even helpers) + `tests/test_benchmark.py`, `tests/test_data.py`, `tests/test_torque_prototype.py`, `tests/test_viz.py`, `tests/test_rk45.py`, plus TODO-skeleton tests in `tests/test_solvers.py`, `test_dynamics.py`, `test_model.py` (names and docstrings only) | **High.** Scaffolding and boilerplate tests. `batched_rhs` duplicates the dynamics algebra for speed; it is tested against `dynamics.state_derivative`. | Save time on timing scaffolding; weak point 3 in CLAUDE.md | [ ] Read `batched_rhs` and confirm it matches your own derivation; run `python -m src.benchmark` on your machine | [fill in] | [fill in: matched tolerance for solve_ivp, what you headline] |
| 11 | `scripts/torque_family_prototype.py`: optional smooth random Fourier torque family (24 coefficients) as an answer to CLAUDE.md weak point 1 | **High**, prototype only, not wired into `data.py`. | Explore whether the operator framing can be earned | [ ] Decide whether to adopt, then regenerate data and rerun the checks | [fill in] | [fill in: adopt or state 8-dim input] |
| 12 | `docs/STUDY_GUIDE.md`: question lists, hint ladders, self-checks for dynamics, solver, model, train, evaluate | **High** for the guide. It contains no solutions to your crucial code. | Prepare for the no-AI interview | [ ] Answer each question in your own words, then have the answers challenged | [fill in] | [fill in] |

### 2026-10-12 (Mon), interactive design session

| # | Task | AI contribution | Why AI | Checked how | Errors / unhelpful output | My decision |
|---|---|---|---|---|---|---|
| 13 | Walkthrough of the existing setup and `model.py`; design discussion (parameter counts, `p`, width/depth sweeps, activation, time normalisation, float32 vs float64, torque family options, baselines, wording of the operator-learning claim); written up in `docs/DESIGN_LOG.md` | **High** for the reasoning and the wording of the log. The choices themselves were put to me and I decided. No crucial-part code (`model.py`, `train.py`, `dynamics.py`, `solvers.py`) was written or changed. | Understand my own model and weigh design options before sweeping | [ ] Re-derive the parameter counts with `n_params`; explain `einsum("bkp,btkp->btk")` unaided; re-derive K = 6 (done, numbers agreed) | AI errors caught and corrected in `DESIGN_LOG.md` A5: said ReLU lacks a useful *second* derivative (the consistency check needs the *first*); first draft of the spectral std had `/2` instead of `/4` in the exponent; float32 check described as RK4 precision (it is rounding error) | [fill in] |
| 14 | Comparison with Lu et al. (2021), arXiv 1910.03193, written up as `docs/DESIGN_LOG.md` A4 | **Medium.** AI fetched and read extracted text of the paper's data, architecture and experiment-setup sections. | Ground my design in the standard reference | [ ] Read Sections 2 to 4 of the paper myself and confirm each point in A4 | First fetch returned only the abstract and then a garbled summary, so the PDF text was extracted with `pypdf`; figure captions were not read and no error figures are quoted. "Other papers use `l` 0.1 to 0.5" came from AI memory and is unverified | [fill in] |
| 15 | Smooth random torque generator in `src/data.py` (`generate_smooth` and helpers, `python -m src.data --smooth`), `scripts/smooth_family_check.py`, `tests/test_data_smooth.py`, and generating `data/dataset_smooth.npz` | **High.** Code written by AI at my request after I chose the settings (`l = 0.2` in normalised time, K = 6, 6 N m peak cap, peak-scaling scheme, 10,000 samples, four OOD sets). The sine path in `data.py` was left untouched. | Save time on generation code; data generation is not a crucial-part module in CLAUDE.md | [ ] Read the new section of `data.py`; re-derive `spectral_std`; run `python -m scripts.smooth_family_check --plot`; run `pytest tests/test_data_smooth.py` | Initial diagnostic figure had an unreadable variance axis and overlapping labels; fixed. Observation, not an error: the amplitude OOD shells are mostly spinning trajectories (see `DESIGN_LOG.md` A7) | [fill in] |

---

## Reflection notes (draft material for the report)

Fill in as the project goes. Cover each point from the brief:

- **Which tools and what for:**
- **Approximate level of AI contribution (overall, and by area: planning / code / writing / analysis):**
- **Why AI was used for those tasks:**
- **How AI-generated material was checked:**
- **Decisions that stayed mine:**
- **Errors, limitations or unhelpful suggestions (specific examples):**
- **Effect on my understanding and workflow:**

## Reminder for the presentation

No AI is allowed during the presentation and interview. Before booking your slot, check you can explain every function in the repo and every figure in the report without help.
