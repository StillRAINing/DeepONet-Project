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
