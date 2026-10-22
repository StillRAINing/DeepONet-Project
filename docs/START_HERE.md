# Start here (updated 2026-10-22, the day before the report is due)

The earlier version of this page (2026-10-12) described the sine-family dataset and work that has
since been done. It is in git history (commit `1aee834`) if you need it.

## Where the project is

Every experiment, figure, benchmark and test is finished. What is left is yours:

1. **The written report** (due Fri 23 Oct 2026). Draft scaffold: `report/MMA3001_Report_DRAFT.docx`
   (facts, numbers, figure placements filled in; every justification, decision, lesson and the AI
   reflection left as "YOU WRITE" prompts).
2. **The open "[fill in]" items** that the report depends on (list below).
3. **Interview readiness**: explain every crucial function and every figure without help.

## Reading order

| # | Read | Why |
|---|---|---|
| 1 | `docs/PROJECT_OVERVIEW.md` | the whole project, every result and number, in one place |
| 2 | `docs/RESULTS_SUMMARY.md` | claim -> number -> file -> figure, per report section; decisions still yours |
| 3 | `docs/STUDY_GUIDE.md` | questions you must answer unaided (no answers given) |
| 4 | `docs/FIGURE_PLAN.md` | every report figure and what you must be able to say about it |
| 5 | `docs/EXPERIMENTS.md` | the full record of what was run (detail behind the numbers) |
| 6 | `docs/DESIGN_LOG.md` | why each design choice was made; your "My own words" cells |
| 7 | `AI_LOG.md` | the AI-use record for the reflection section |

## Status of every doc

| Doc | Status |
|---|---|
| `PROJECT_OVERVIEW.md` | current (new, 2026-10-22) |
| `RESULTS_SUMMARY.md`, `FIGURE_PLAN.md`, `EXPERIMENTS.md`, `STUDY_GUIDE.md`, `README.md` | current (checked against the results files 2026-10-22) |
| `DESIGN_LOG.md` | current for decisions; statuses of W1-W10 updated 2026-10-22; your "My own words" cells still partly blank |
| `REVIEW_NOTES.md` | **historical**: the 2026-10-12 audit of the old sine-family dataset; a banner at the top says what was resolved |
| `AI_LOG_PROMPT.md` | in use: another AI session ran its AI-side fixes and the interview pass on 2026-10-22 (AI_LOG rows 46 to 53); delete the file once your "Checked how" / "My decision" cells are filled |
| `DeepONet_Project_Explained.pdf` | **out of date** (2026-10-12: 2,200 sine-family trajectories, no models or results). Superseded by `PROJECT_OVERVIEW.md`. Do not quote it; delete it or keep it as history |
| `physics/*.pdf` | still correct for `dynamics.py` (worked examples match the code); reading material, not your own derivation; not logged as AI-made unless you add a row |
| `html/` | pdoc API docs, regenerated 2026-10-22 with `pdoc src -o docs/html -d google --no-show-source` |

## Open items the report depends on (all yours)

- [ ] Headline: rule-selected four (recommended) vs also quoting the ff6 + cons MLP as "best found" (EXPERIMENTS E4 "Decision").
- [ ] Framing: MLP accuracy vs DeepONet consistency and cached per-query cost (EXPERIMENTS rule step 3 "Decision").
- [ ] The "Decision: [fill in]" lines under E1, E2, E3 and the E4 prediction line in `EXPERIMENTS.md`.
- [ ] Phase-coverage test for the sine family, or state that the sine family is diagnostics only (the one missed mutation).
- [ ] O5: enforce `x(0) = x0` or report it as a limitation (it is currently reported).
- [ ] O8: keep the amplitude shells as decided and say they are a regime change (they are currently reported that way).
- [ ] DESIGN_LOG: "My own words" cells for D10 to D14 and A8; section E checklist.
- [ ] AI_LOG: "Checked how" and "My decision" columns (rows 1 to 53) and the reflection notes (another AI session was running the `AI_LOG_PROMPT.md` interview pass on 2026-10-22, AI_LOG row 53).
- [ ] Re-derive `M`, `C`, `G` and the payload terms by hand (AI_LOG item 3). This is the oldest open item.
- [ ] Fix two figure defects before they go in the report (FIGURE_PLAN, "Known figure defects").

## Run things

Tests: `pytest` (117 pass, 0 skipped; `results/test_report.xml`). On this machine only `py -3.10` has torch;
pytest lives in a scratch venv. Live demo: `python -m src.demo`. Full reproduce list: `README.md`.
