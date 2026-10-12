# Design log: my decisions and changes

This records the design decisions I have made and the changes I have developed.
It complements `AI_LOG.md`, which records AI *use*. This file records *my choices*.

**How to read the "Where the reasoning came from" column.** Several decisions below were
reached after discussion with an AI assistant (Claude) on 2026-10-12. I considered its
recommendation and chose to adopt it. The reasoning is therefore AI-supplied. Each entry has a
"My own words" line that I must fill in myself, after re-deriving or testing the point, so that I
can defend it unaided in the interview. An entry with that line still blank is not yet defended.

**Status key:** DECIDED = I have chosen, nothing built yet. DONE = built and checked.
OPEN = not decided. TO TEST = decided provisionally, a sweep or experiment will confirm.

---

## A. Decisions made on 2026-10-12

| # | Decision | Status | Where the reasoning came from | My own words (fill in) |
|---|---|---|---|---|
| D1 | **Adopt a smooth random torque family** (random Fourier / GP-like, evaluable at any time) as the main experiment, instead of only `A sin(2 pi f t + phi)`. | DECIDED | AI recommendation (CLAUDE.md weak point 1): the sine family is 6 numbers in disguise, so the 304-value branch input is redundant and the operator framing is not earned. | A smooth random torque family is more general and fits the project's intent better. A sinusoidal family reduces the problem to fitting six parameters, so it would not demonstrate operator learning. |
| D2 | **Keep the sine family as an easy sanity-check dataset** (debugging, overfit-5 test) and report both. | DECIDED | AI recommendation. | The sine family is kept as a simple dataset for debugging and sanity checks, since it already exists in `data.py` and needs no new code. |
| D3 | **Torque must be a continuous function of t**, not only grid samples. | DECIDED | RK4 evaluates the torque at `t`, `t + h/2`, `t + h` on the fine grid (8 substeps per output step), so a grid-only GP sample cannot drive the integrator. | The torque must be a continuous function of time because real actuator torques are continuous. [Incomplete: add the RK4 mid-step evaluation reason.] |
| D4 | **Normalise time as `t / T`** (trunk sees [0, 1]), `T = 3 s`. | DECIDED | AI recommendation: tanh behaves best on O(1) inputs. `T` is a problem constant, not a data statistic, so there is no leakage. The identical scaling must be applied in train, evaluate and benchmark. | Scaling time to [0, 1] matches the input range tanh handles best and makes training easier. |
| D5 | **Train in float32, compute error metrics in float64** (cast predictions up). | DECIDED, TO TEST | AI recommendation: float32 precision (~1e-7) is far below expected surrogate error (~1e-3 to 1e-2). Check: cast a float64 trajectory to float32 and measure the relative L2 difference. | Float32 is precise enough for the accuracy this surrogate is expected to reach. [Check still to run: float32 rounding error, not RK4 error.] |
| D6 | **Activation: default tanh, compare against SiLU once, skip sine activations.** ReLU excluded. | DECIDED, TO TEST | AI recommendation: smooth activations are needed for the `dq/dt` consistency check (weak point 6). If tanh and SiLU are within seed noise, keep tanh. | ReLU is excluded because it is piecewise linear and so provides no useful derivative information. [Pending: restate in own words why a derivative of the output w.r.t. t matters here.] |
| D7 | **Sweep architecture one axis at a time, not a full grid**: p in {4, 8, 16, 32, 64, 128}, width in {32, 64, 128, 256}, branch depth and trunk depth each in {1, 2, 3, 4}, varied separately. At least 3 seeds per point. Select on validation error only. | DECIDED | AI recommendation: a full grid is ~650 runs. Seed spread is needed to tell a real difference from noise. | Varying one hyperparameter at a time isolates the effect of each. Running at least three seeds per setting lets me compare the seed-to-seed variation with the difference between settings, which helps identify when a smaller or cheaper model performs equivalently. |
| D8 | **Sweep order**: (1) fix torque family and regenerate data, (2) normalisation and `train.py`, (3) overfit-5 test, (4) sweeps. | DECIDED | AI recommendation: the best architecture depends on the data and the scaling, so sweeping first means redoing it. | Fixing the data and normalisation first means the architecture sweeps are run on the final problem and do not need repeating. |
| D9 | **Record every architecture decision in one table** (decision, options tried, validation metric, choice, reason), to go into the report. | DECIDED | AI recommendation. | A single decision table gives every architecture choice a recorded, evidence-based justification, lets me answer "why this value?" from validation results rather than from a guess, and separates tested choices from defaults. |

## A2. Further decisions and reasoning (2026-10-12, later in the same session)

Status for these is stated precisely. "PENDING" means I have not yet approved it. Reasoning is
AI-supplied unless I have written otherwise, so each "My own words" cell is for me to fill.

| # | Decision | Status | Reasoning (AI-supplied, to be checked by me) | My own words (fill in) |
|---|---|---|---|---|
| D10 | **Torque family recipe**: fixed-frequency Fourier series with a decaying spectrum, mode-k coefficient standard deviation `exp(-(omega_k * ell)^2 / 4)`, so that the variance follows the RBF spectrum `exp(-omega^2 ell^2 / 2)` (corrected from an earlier AI draft that had `/ 2` in the exponent of the standard deviation), period `L = 2T` to avoid forcing `tau(0) = tau(T)`. | DECIDED (accepted when I approved the generator and K, 2026-10-12) | See the comparison in A3. It meets four criteria at once: evaluable at any t, compact fixed-meaning coefficients (so the compact baseline is well defined), a direct smoothness knob, and smooth enough for the physics. Approximates an RBF-kernel GP with length `ell`. | [fill in] |
| D11 | **Wording of the operator-learning claim**: the torque input is a random smooth function from a band-limited family with correlation length `ell`. It is finite-dimensional in effect (about `2K` coefficients per joint) but is presented to the network as a function sampled at 151 points, as in the standard DeepONet setup. I do not claim generalisation to arbitrary input functions. | DECIDED (wording) | Honest: the sine family is 6 numbers, the Fourier family about 24, and standard GP benchmarks are also effectively low-dimensional (about `T/ell` or a few times that). The difference is one of degree. This pre-empts the examiner's obvious question. | [fill in] |
| D12 | **Baselines.** Accuracy: (a) MLP on the full 304-dim input, already built; (b) MLP on the compact coefficients plus `t`, not yet built. Speed: your Python-loop RK4, batched NumPy RK4, `solve_ivp` at a tolerance matched to the surrogate's error, with a break-even query count including data generation and training. RK4 is the ground truth, not an accuracy baseline. | DECIDED | A baseline gives the DeepONet's error a reference. (a) alone is handicapped by 302 redundant inputs, so a win would be overstated. (b) removes that handicap and may win, which is a legitimate reportable result. Matched parameter counts and equal tuning effort keep the comparison fair (W1). | [fill in] |
| D13 | **Extra evaluation experiments, taken from Lu et al.** (1) sensor-count sweep; (2) error vs correlation length `ell` and vs horizon `T`; (3) resolution robustness (re-sample the torque on 76 or 301 points; the MLP cannot do this); (4) generalisation to a different function family (e.g. random splines, or a different `ell`) after training on the Fourier family. | NOTED, scope to be confirmed (not all may fit the timeline) | These turn the operator-learning claim into something tested rather than asserted, and give a defensible answer to "why this `ell`?". | [fill in] |
| D14 | **Data regeneration plan.** Keep the sine path in `data.py` untouched. Add the smooth generator as a separate function. Generate a few hundred samples first and inspect them (torque curves, spectrum, peak distribution, trajectory statistics, worst cases using the existing `scripts/`) before approving the settings. Then regenerate the full dataset with seed 3001. | DECIDED (plan) | Approving settings from plots rather than from abstract numbers avoids committing to an amplitude or `ell` that makes the arm spin chaotically or lets gravity dominate. | [fill in] |

## A3. Why a Fourier series, and how it compares with the alternatives

Criteria used: (1) evaluable at any `t` (RK4 needs the torque between grid points), (2) compact,
fixed-meaning coefficients (for the compact baseline), (3) a direct, explainable smoothness
control, (4) smooth enough for the physics and the `dq/dt` check. The recommendation is a
trade-off, not a claim that Fourier is best on every axis.

| Alternative | Where it is better | Where fixed-frequency Fourier is better |
|---|---|---|
| Random Fourier features (GP-like) | Closer to an exact RBF GP; no periodicity artefact without a workaround | Random frequencies differ per sample, so the coefficient vector has no fixed meaning and the compact baseline learns little. Periodicity is avoidable with `L = 2T`. |
| GP on a fine grid + cubic spline | The exact GP and the literature standard (Lu et al.). The strongest rival. | No Cholesky cost, no interpolation step, and a compact representation. A truncated spectral Fourier series is close to a stationary GP anyway. Pick the grid GP if "literature-standard" outranks "compact baseline". |
| Random cubic spline through knots | Simplicity; knot values are a natural compact form | Finer smoothness control (a continuous `ell` rather than an integer knot count); independent knot values can overshoot |
| Piecewise constant or linear | Trivial | Not smooth: kinks and jumps degrade RK4's order in the affected steps, contradict D3, and give a poor `dq/dt` |
| Chebyshev polynomials | Compact for smooth functions | Oscillation and growth near the interval ends make torque statistics non-uniform in time |

Downsides of the Fourier choice to remember: built-in periodicity (hence `L > T`); band-limited at
`K` modes; not literally a GP, so do not call it one. Say "a random Fourier series with a
Gaussian-shaped spectral decay, matching an RBF-kernel GP with correlation length `ell`".

## A4. Alignment with Lu et al. (2021)

Source: Lu, Jin, Pang, Zhang, Karniadakis, "Learning nonlinear operators via DeepONet based on
the universal approximation theorem of operators", arXiv 1910.03193 (Nature Machine
Intelligence, 2021). I read the extracted text of the architecture, data-generation and
experiment-setup sections; I did not read every figure caption, so no error figures are quoted.

What they do: unstacked DeepONet with bias performed best (smaller test error and fewer
parameters than stacked); input functions from a GRF with RBF kernel (main setting `l = 0.2` on
[0, 1], 100 sensors) and from Chebyshev polynomials; 10,000 training samples; Adam, learning rate
0.001; reference solutions by Runge-Kutta (4, 5); small networks (trunk depth 3 and branch depth
2, width 40 for the ODE cases); a grid-searched fully connected baseline; a theoretical bound on
the number of sensors needed that depends on `l`; a gravity pendulum with an external force as
Section 4.2, studied across `k`, `T` and `l`.

| Point | Lu et al. | This project | Consequence |
|---|---|---|---|
| Architecture | Unstacked, with bias | Unstacked, with bias, one set of `p` basis functions per output channel | Matches the best-performing variant |
| Input | Only the forcing function | Forcing plus `m_p` and `b` | Harder, mixed input |
| Smoothness | `l = 0.2` on a unit domain | Planned `ell` about 0.5 to 1 s on a 3 s window, so `ell/T` about 0.17 to 0.33 | Comparable range |
| Data | 10,000 samples | 1,400 training samples | Much less data: the biggest practical difference |
| Size | Width 40, depth 2 to 3 | Width 128, depth 3 | Supports sweeping down from my defaults |
| Horizon | `T = 1` | `T = 3` | Errors grow with horizon; do not expect to match their error levels |
| Output points | Random `y` per sample | The same fixed 151-point grid | Simpler, but only learns on that grid |

Suggested report wording: "Following Lu et al. (2021), the input is a fixed-sensor representation
of a random smooth function and the network is an unstacked DeepONet with bias. Unlike their
pendulum example, the input also includes two physical parameters, the system is a two-link arm,
the dataset is smaller, and the torque family is a band-limited random Fourier series that
approximates an RBF-kernel GP."

## A5. Corrections to earlier AI statements (so the record is accurate)

- **D6 / ReLU.** I first said ReLU has "no useful second derivative". The relevant quantity for my
  planned consistency check is the *first* derivative: `dq/dt` from autodiff of the `q` outputs
  compared with the `qdot` outputs. A ReLU network is piecewise linear in `t`, so `dq/dt` is
  piecewise constant and a poor match for a smooth `qdot`. The second derivative would matter only
  for a physics-informed loss using the dynamics, which is not planned.
- **D5 / float32 check.** The check is float32 *rounding* error (cast a float64 trajectory to
  float32 and back, compute the relative L2 difference, expect about 1e-7), not the RK4 truncation
  error.
- **D3 / continuity.** Real torques being continuous is true, but the reason the generator must be
  evaluable at any `t` is that RK4 evaluates the torque at `t`, `t + h/2` and `t + h` on the fine
  grid (8 substeps per output interval).
- **Fourier periodicity.** A Fourier series with period equal to the 3 s window forces
  `tau(0) = tau(T)`; use `L = 2T`.
- **Seeds (D7).** The data seed (3001) stays fixed; only the training seed varies.
- **D10 spectral formula.** Coefficient standard deviation is `exp(-(omega_k * ell)^2 / 4)` (variance
  `exp(-omega_k^2 ell^2 / 2)`), not `/ 2` inside the standard deviation as first written. The K = 6
  derivation used the variance spectrum and is unaffected.

## A6. Data-generation settings (O1), 2026-10-12

Status of each: DECIDED = I chose it; PROPOSED = AI-proposed, awaiting my check. Measurements
were made by the AI on the *old sine family* (150 samples, seed 1, so only indicative of the new
family). Machine: Intel i7-13620H (10 cores, 16 threads), 31.7 GB RAM, RTX 4050 Laptop (6 GB);
PyTorch 2.5.1+cu121 on Python 3.10 reports CUDA available.

| Setting | Value | Status | Reasoning | My own words (fill in) |
|---|---|---|---|---|
| Correlation length `ell` | `l = 0.2` in **normalised time** `t/T` (= 0.6 s for `T = 3 s`), RBF-style spectrum | DECIDED | Matches the setting of Lu et al. (2021), who use `l = 0.2` on a domain of length 1 (read in the paper). Quoting `l = 0.2` in seconds would be `ell/T = 0.067`, about 3 times rougher than theirs, so the unit must be stated. The claim that other papers use 0.1 to 0.5 is from AI memory and is **not verified**; cite a source before using it. | I use `l = 0.2` in normalised time, the value Lu et al. (2021) chose for their Gaussian-random-field input functions, so the smoothness matches the standard benchmark. [Unverified: other work reportedly uses 0.1 to 0.5; source needed before quoting.] |
| Fourier modes `K` | 6 per joint (12 coefficients each, 24 in total), period `L = 2` in normalised time (`2T`) | DECIDED (derived from `l`) | Spectrum standard deviation `1/(2 pi l)` = 0.80 cycles per unit time; mode `k` has frequency `k/2`; `k = 6` is 3.0, about 3.8 standard deviations, leaving about 0.016% of variance beyond it. To verify numerically. | I re-derived K independently from the spectrum width `1/(2 pi l)` and the mode frequency `k/2`, and obtained 6 modes per joint, matching the AI's derivation. [Numerical check of the variance beyond K: see section A7.] |
| Amplitude cap | 6 N m peak per joint | DECIDED (provisional; value to be verified by spin statistics on the new family) | Joint-1 gravity torque at the start pose is about 15 to 24 N m, so 6 N m cannot hold the arm up and perturbs the swing. Old sine data at 15 N m: 73% of samples exceeded 2 pi in abs q, 95th percentile 46 rad (max 65 rad); at 6 N m: 25% and 13 rad. Criterion proposed by the AI: 95th percentile of max abs q about 15 rad or less. Re-run on the new family and with `scripts/torque_range_sweep.py` before approving. | A 6 N m peak is well below the 15 to 24 N m gravity torque at the start pose, so it cannot hold the arm up and instead perturbs its swing without making most trajectories spin. I will verify this with the spin statistics. |
| Amplitude scheme | Draw the Fourier shape, rescale to peak `A`, with `A ~ U(0, cap)`; store the scaled coefficients | DECIDED | Covers weak to strong torques, hard-caps the peak (controls spinning), keeps the old meaning of `A`, and keeps the stored coefficients an exact linear representation. Alternative: fixed-spread Gaussian coefficients (unbounded peak, roughly equal strength per sample). | Rescaling each sampled shape to a peak drawn uniformly from [0, cap] covers a wider range of torque strengths than a fixed spread and imposes a hard cap on the peak. |
| Dataset size | 10,000 total (7,000 train, 1,500 val, 1,500 test) | DECIDED | Lu et al. use 10,000 samples for their ODE cases. Generation measured at 0.09 s per trajectory, about 15 minutes in total; memory is negligible (about 36 MB of arrays). Allows a learning curve (error vs training size). | Lu et al. use 10,000 samples for their ODE cases. Generating that many is feasible on my hardware (about 15 minutes of CPU time and negligible memory), so I follow their dataset size. |
| OOD sets | Keep the `m_p` / `b` shift; add (2) amplitude shell 1: peak `A` in [6, 9] N m; (3) amplitude shell 2: `A` in [9, 12] N m; (4) smoothness shift: `l = 0.1` in normalised time with K = 13 modes. 200 samples per set. Shells rather than caps, so that no OOD sample lies inside the training amplitude range. Never used for selection. The "different family" set is dropped for now. | DECIDED | Addresses weak point 4: the current OOD set only changes the smooth parameters `m_p` and `b`. Framing is "how does error degrade", not pass or fail. Keep shifts separate to attribute any error increase. | The original OOD set changed only the smooth parameters `m_p` and `b`. Amplitude and smoothness shifts, reported as error against distance from the training range, test whether the network learned the dynamics or merely interpolated the training range. Each shift is varied separately so any error increase can be attributed. Smoothness shift note: the compact-coefficient baseline cannot represent K = 13 inputs without truncation, whereas the DeepONet reads sensor samples. |
| `m_p`, `b` ranges | Unchanged: `m_p` in [0, 1], `b` in [0.05, 0.5] | DECIDED | Change one thing at a time: the torque family is the main change, so keeping these makes results attributable and comparable with the sine data and diagnostics. The OOD ranges start where these end. | Changing one factor at a time keeps results attributable to the torque family and keeps them comparable with the sine data and the existing diagnostics. |

Open follow-ups: run the spin statistics and variance-beyond-`K` check on the new family; the GPU
is available for training (PyTorch CUDA), while data generation is CPU NumPy.

## A7. Smooth-family generator: checks and observations (2026-10-12)

Code: `src/data.py` (`generate_smooth` and helpers, `python -m src.data --smooth`), checks in
`scripts/smooth_family_check.py` (300 samples, seed 0, 100 per OOD set), tests in
`tests/test_data_smooth.py`. The sine path in `data.py` is unchanged. Figure:
`results/smooth_family_check.png`. AI-written code, logged as AI_LOG item 15.

**Recipe as implemented.** `tau_j(t) = sum_{k=1..6} c_jk cos(2 pi k t / P) + s_jk sin(2 pi k t / P)`,
`P = 2T = 6 s`, coefficient std `exp(-(omega_k ell)^2 / 4)` with `ell = 0.2` in normalised time. Each
joint's shape is rescaled so its peak over [0, T] equals `A_j ~ U(0, 6)` N m, and the scaled
coefficients are stored (shape `(n, 2, 2, 6)`: joint, cos/sin, mode). No constant (k = 0) mode, so the
count is 12 coefficients per joint as decided. Smoothness-shift set: `ell = 0.1`, K = 13.

**Checks**
| Check | Result |
|---|---|
| Variance beyond K = 6 at `ell = 0.2` | 4.4e-5 (0.0044%). My hand estimate of 0.016% was a continuous-Gaussian-tail approximation; the discrete sum is smaller. Both are negligible, so K = 6 stands. |
| Variance beyond K = 13 at `ell = 0.1` | 2.4e-5 |
| Torque peak on the 151-point grid | 0.01 to 5.99 (cap 6), as designed |
| Effective dimension of the sampled torques (both joints) | 12 for 99% of the variance, 14 for 99.9%, against 24 coefficients. So the input is richer than the sine family (6) but still low-dimensional; this supports the honest wording in D11. |
| Tests | 8 new tests pass, plus the existing 7 data tests; full suite 45 passed, 13 skipped (the skips are my own `TODO (student)` test stubs in `test_dynamics.py`, `test_model.py`, `test_solvers.py`) |

**Motion statistics** (max over time and joints of `|q|`, in rad)
| Set | n | Median | Fraction > pi | Fraction > 2 pi | 95th percentile | Max `|q|` | Max `|qdot|` (rad/s) |
|---|---|---|---|---|---|---|---|
| In-range (training settings) | 300 | 2.9 | 0.31 | 0.19 | 13.6 | 43 | 63 |
| `ood` (`m_p`, `b` shift) | 100 | 2.8 | 0.05 | 0.00 | 3.1 | 3.4 | 15 |
| `ood_amp1` (A in [6, 9]) | 100 | 8.8 | 0.88 | 0.81 | 45 | 98 | 120 |
| `ood_amp2` (A in [9, 12]) | 100 | 15.9 | 0.99 | 0.94 | 52 | 126 | 141 |
| `ood_smooth` (`ell = 0.1`) | 100 | 2.9 | 0.25 | 0.14 | 9.2 | 22 | 40 |

**Observations**
1. In-range meets the proposed spin criterion (95th percentile about 15 rad or less): 13.6 rad. The
   6 N m cap is therefore provisionally confirmed. The median of 2.9 rad is about what a free swing from
   the horizontal pose under gravity gives.
2. The `m_p`/`b` OOD set is *calmer* than the training set (none beyond 2 pi): heavier payload and more
   friction damp the motion. This is weak point 4 in practice; that set alone is a mild test.
3. The amplitude shells are mostly spinning trajectories (81% and 94% beyond 2 pi, up to 126 rad).
   Going from A <= 6 to A in [6, 9] is as much a change of regime (not spinning to spinning) as a
   change of amplitude. Expect large errors there, and report them as degradation, not failure.
   See open item O8.
4. The smoothness shift is moderate (14% beyond 2 pi).
5. Caveat: OOD statistics use 100 samples per set, so they are indicative only.

**Full dataset** (`python -m src.data --smooth --workers 8`, seed 3001, 5.5 minutes, written to
`data/dataset_smooth.npz`, 78 MB, git-ignored by `*.npz`; the sine `data/dataset.npz` is untouched).
All arrays finite. Metadata stored: seed 3001, `ell` 0.2, K 6, period 6 s.

| Set | n | Coefficients shape | Fraction > 2 pi | 95th percentile max `|q|` | Max `|q|` |
|---|---|---|---|---|---|
| train | 7000 | (2, 2, 6) | 0.20 | 14.2 | 129 |
| val | 1500 | (2, 2, 6) | 0.19 | 14.0 | 74 |
| test | 1500 | (2, 2, 6) | 0.18 | 14.1 | 65 |
| ood (`m_p`, `b` shift) | 200 | (2, 2, 6) | 0.00 | 3.2 | 3 |
| ood_amp1 (A in [6, 9]) | 200 | (2, 2, 6) | 0.73 | 35.1 | 157 |
| ood_amp2 (A in [9, 12]) | 200 | (2, 2, 6) | 0.94 | 65.7 | 122 |
| ood_smooth (`ell = 0.1`) | 200 | (2, 2, 13) | 0.14 | 8.8 | 29 |

The train, val and test splits agree closely (about 19% beyond 2 pi, 95th percentile about 14 rad),
so the random split is balanced. A few training trajectories reach tens to over a hundred radians
(max 129 rad): these extreme samples will dominate unnormalised error and should be looked at
individually when deciding the loss and output normalisation (weak point 5, open item O2).

## B. Weaknesses I have identified and committed to work on

These come from the review of `model.py` and the existing setup. Each needs an experiment, not
an assertion.

| # | Weakness | Planned action | Status |
|---|---|---|---|
| W1 | Parameter counts are unmatched. By hand: DeepONet 171,396 vs MLPBaseline 89,220 (verify with `n_params`). A win could be explained by size alone. | Match parameter counts, or plot error against parameter count for both. | OPEN |
| W2 | `p = 64`, width 128, depth 3 have no stated justification. | Sweeps in D7; choose at the knee of validation error vs p. | OPEN |
| W3 | Branch and trunk share the same depth. The trunk sees a 1-D input and may need far less capacity. | Vary trunk depth and width separately from the branch. | OPEN |
| W4 | Unstacked design (separate basis per channel) is untested against a shared trunk. | Optional comparison. Relevant to `qdot = dq/dt` consistency. | OPEN |
| W5 | Nothing enforces `x(0) = x0`. | Measure the error at `t = 0`. If not about zero, consider a hard constraint and justify it. | OPEN |
| W6 | MLP baseline takes the raw 304-dim input, so it is handicapped. | Add a baseline on the compact coefficients plus `t`. It may win; report it either way. | OPEN |
| W7 | Speed claim could be inflated by comparing against a Python-loop RK4. | Compare against batched NumPy RK4 and `solve_ivp` at matched tolerance; report break-even including data generation and training time. | OPEN |
| W8 | OOD set only changes `m_p` and `b`. | Frame as "how error degrades"; add OOD in torque amplitude, frequency or smoothness. | OPEN |
| W9 | Relative L2 over all four channels lets large channels dominate. | Per-channel normalisation; report per-channel and per-time error and worst cases. | OPEN |
| W10 | `README.md` status boxes are unticked and the Method table lists only fixed-step RK4 (RK45 now exists). | Update after I have verified each piece myself. | OPEN |

## C. Open decisions (mine to make)

| # | Question | Notes |
|---|---|---|
| O1 | Settings of the smooth torque family, needed before regenerating the data: (1) correlation length `ell` (0.5 to 1 s would match the old sine family's roughness); (2) amplitude cap (old family: 15 N m, chosen to compete with gravity of about 15 to 24 N m); (3) number of Fourier modes `K` per joint (2K coefficients each; the prototype `scripts/torque_family_prototype.py` uses 24 in total, which I assume is 12 per joint but have not checked); (4) dataset size (2000 / 1400 train is small for a harder family; Lu et al. used 10,000); (5) keep `m_p` in [0, 1] and `b` in [0.05, 0.5]? | Answers still owed by me, or "AI proposes, I approve after inspecting samples" (D14). Starting point for (1) and (3): a torque whose fastest wiggle has period 1 s makes about 3 bumps in a 3 s window; work out how many modes that needs. |
| O6 | Approve or change D10 (Fourier recipe) versus the grid-GP alternative. | Depends on whether "literature-standard" or "compact baseline" ranks higher for me. |
| O8 | Keep the amplitude shells [6, 9] and [9, 12] N m as decided, or use narrower shells (for example [6, 8] and [8, 10])? | The decided shells are mostly spinning (A7, observation 3), so they are a regime change. This is my call; if I keep them, the report must say so. |
| O7 | Which of the D13 experiments are in scope. | Sensor-count sweep and resolution robustness are cheap; cross-family generalisation is the most informative for the operator claim. |
| O2 | Output normalisation scheme (per-channel mean/std from the training split only). | Statistics from train only; test and OOD untouched (weak point 8). |
| O3 | Loss function and any channel weighting. | |
| O4 | Learning rate, optimiser, epochs, batch size, early stopping. | |
| O5 | Whether to enforce the initial condition architecturally (W5). | |

## D. Changes I have developed (code and data)

Nothing has been changed in the codebase as a result of the 2026-10-12 discussion. The decisions
above are recorded but not yet implemented. As each is built, add a row here.

| Date | Change | Files | How I checked it | AI involvement |
|---|---|---|---|---|
| | | | | |

## E. Verification I still owe

- [ ] Re-derive M(q), C, G and the payload parallel-axis formula by hand (AI_LOG item 3).
- [ ] Confirm the 8-substep RK4 ground truth against RK45 or `solve_ivp` at a tight tolerance.
- [ ] Run `n_params` on both models and confirm the counts in W1.
- [ ] Run the float32/float64 cast check (D5).
- [ ] Explain the `einsum("bkp,btkp->btk")` in `DeepONet.forward` and its shapes, unaided
      (for B = 32, n_t = 151, p = 64: shapes of `b`, `k` and the result).
- [ ] Explain in my own words why the first derivative of the output w.r.t. `t` matters (A5, D6).
- [ ] Read Lu et al. (arXiv 1910.03193) Sections 2 to 4 myself, since the comparison in A4 was
      prepared from extracted text by the AI and I should confirm it against the paper.
- [ ] Fill in every "My own words" cell in A and A2.
- [ ] Add an `AI_LOG.md` row for the 2026-10-12 review and design discussion, and fill in the
      "Checked how" and "My decision" columns myself.
