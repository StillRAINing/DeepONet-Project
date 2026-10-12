# Study guide: things you must be able to defend unaided

Purpose: no AI in the presentation. Each section lists (1) questions to answer *before* you code or
review, (2) a hint ladder if you get stuck (go one rung at a time), (3) checks you should write
yourself, (4) "examiner" questions. Answers are deliberately not here. Write yours in your own words
in a notebook or `docs/my_answers.md`, then ask me to challenge them.

Status key: [ ] not yet, [x] done. Tick honestly.

---

## 1. Dynamics (`dynamics.py`)

### Questions first
1. Why does the equation of motion have the form `M qddot + C qdot + G + b qdot = tau`? Where does each term come from in the Lagrangian `L = T - V`?
2. `M(q)` depends on `q2` only. Why not `q1`? What would break if the arm were in 3D or had a third joint?
3. Why must `M` be symmetric positive definite? What does a negative eigenvalue mean physically?
4. Why is `C(q, qdot) qdot` quadratic in velocities? Which two physical effects hide in `q1d*q2d` and `q2d^2`?
5. Payload: why is the *new* COM at `(m2 lc2 + m_p l2)/(m2 + m_p)`? Why does the inertia about the new COM need the parallel-axis theorem, and why two correction terms?
6. Energy: why is `dE/dt = tau . qdot - b |qdot|^2` and not 0? What happens to this identity if friction is Coulomb instead?

### Hint ladder (payload inertia, the item the tests miss)
- Rung 1 (concept): inertia of a composite body about its own COM = sum of each part's own inertia + each part's mass x (distance from part-COM to composite COM)^2.
- Rung 2 (what are the parts): link 2 (mass `m2`, own inertia `I2`, COM at `lc2`) and a point mass (own inertia 0, at `l2`).
- Rung 3: write the two distances symbolically in terms of `lc2`, `l2`, `m2`, `m_p`. Check they sum to `l2 - lc2`.
- Rung 4: only if stuck, ask me to review your formula; I will not hand over the line.

### Checks to write yourself
- [ ] **Payload test against brute force.** Build the same arm as three bodies (link 1, link 2, point mass), compute T at random states directly from geometry, compare to `0.5 qdot^T M qdot`. (This is the test that kills the surviving mutations in `REVIEW_NOTES.md` section 1.)
- [ ] Coriolis identity: `qdot^T (Mdot - 2C) qdot = 0` using a finite-difference `Mdot`.
- [ ] Power balance: with torque and friction on, `dE/dt` from a finite difference of `energy` along a trajectory equals `tau . qdot - b |qdot|^2` to O(h^4)-ish.
- [ ] Pendulum limit: find a setting where the arm is one rigid pendulum (for example joint 2 held at `q2 = 0` by a very stiff PD torque, or simply a hanging-equilibrium small-oscillation check). Work out the effective inertia, the gravity term and the small-angle period yourself, then compare with the simulation. Which of the two routes is valid, and why is zero torque on joint 2 *not* a lock?
- [ ] Re-derive `M`, `C`, `G` by hand (AI_LOG item 3). The PDFs in `docs/physics/` are reading material (they look AI-generated; see `REVIEW_NOTES.md` section 1), not a substitute: can you do it with every page closed?

### Examiner questions
- "Show me where gravity enters and why it is `cos q1`, not `sin q1`." (Convention: `q1` from horizontal.)
- "What changes if the payload were at the end of link 1 instead?"
- "Is the system conservative? Prove it with your own test."
- "What are the units of every term in `M`, `C`, `G`?"

---

## 2. Solver (`solvers.py`)

### Questions first
1. What does "RK4 is fourth order" mean as a statement about error? Global vs local error: which is O(h^5) and which O(h^4), and why do they differ?
2. If you halve `h`, by what factor should the global error fall? Why can't you read the order off a *single* run?
3. Why four stages, and why those weights `1, 2, 2, 1`? What is the order condition RK4 satisfies that Euler fails?
4. What does "stiff" mean, and is this system stiff? (Look at `M` for tiny `I2` and big `b`.)
5. Why is energy not exactly conserved by RK4 even when the true system conserves it? What would a symplectic integrator do differently? Why not use one here (friction and torque make the system non-conservative)?

### Hint ladder (convergence-order test)
- Rung 1: pick an ODE with a known solution (`y' = -y`, `y(0) = 1`), integrate to `T = 1` with `h, h/2, h/4, h/8`.
- Rung 2: `p ~ log2(err(h) / err(h/2))`; expect about 4.
- Rung 3: repeat on the arm, using a very fine RK4 (or `solve_ivp` with `rtol=1e-12`) as the reference. Why must the reference be much better than the finest run?
- Rung 4: why does the measured order break down at tiny `h` (round-off) or huge `h` (outside the asymptotic regime)? Plot error vs `h` on log-log axes.

### Checks to write yourself
- [ ] `tests/test_solvers.py` does not exist. Create it. At minimum: exact solution on `y' = -y`, order estimate in `[3.8, 4.2]`, exactness on a cubic polynomial RHS that RK4 integrates exactly (`y' = t^3`).
- [ ] Re-run the mutation "perturb `k2` by 1%" and confirm your test now fails.
- [ ] Compare `rk4_integrate` with `solve_ivp(rtol=1e-10)` on the arm for a mild sample, then on a *spinning* sample, and explain the difference in agreement (`REVIEW_NOTES.md` 2.2).

### Examiner questions
- "Why is a fixed-step solver acceptable for ground truth? When would it not be?"
- "How did you choose `substeps = 8`? Show me the evidence." (You currently cannot: see 2.2.)
- "Where does the 1/6 come from?"

---

## 3. Model (`model.py`)

### Questions first
1. What is the *universal approximation theorem for operators* claiming, in plain words? What does it not promise (rate, sample count)?
2. Shapes: for `B = 5`, `n_t = 151`, `p = 64`, `N_OUT = 4`, write the shape of the branch output, the trunk output, and the result of the einsum. Which index is summed?
3. Why does the trunk see only `t`? Why can the trunk output be shared across samples, and what does that buy you computationally?
4. What would you lose by using a single set of `p` basis functions for all four channels?
5. Why is the DeepONet output linear in the branch outputs? Does that matter for what it can represent?
6. The ground-truth map has `q(0) = 0` exactly. Does your model enforce that? Should it (hard constraint such as `x(t) = t * net(t)`, or a soft loss term)? Which would an examiner prefer you justify?
7. Weak point 6: nothing ties the `qdot` channels to `d q/dt`. How would you measure that mismatch using autograd?

### Hint ladder (einsum test)
- Rung 1: write the equation `out[b,t,k] = sum_p ...` on paper.
- Rung 2: implement it with three nested Python loops on tiny sizes (`B=2, n_t=3, p=4`).
- Rung 3: assert `torch.allclose` against the model's einsum with the same weights (grab branch/trunk outputs by calling the sub-networks directly).
- Rung 4: add a "causality" check: change `u[0]` only; outputs for `b >= 1` must be unchanged.

### Checks to write yourself
- [ ] Einsum equals loop version.
- [ ] Trunk is independent of `u`: changing `u` leaves trunk output unchanged.
- [ ] Output at time `t_i` depends on `t_i` only through the trunk (permuting the time grid permutes the outputs).
- [ ] Parameter counts: derive `n_params` by hand for one config and compare with `n_params(model)`.
- [ ] Overfit 5 samples (see section 4): if the DeepONet cannot do that, there is a bug, not a hyperparameter problem.

### Examiner questions
- "Draw the architecture and label every tensor shape."
- "Why not just feed `(u, t)` to one MLP? Show me a case where DeepONet should win, and whether your data is that case."
- "How many parameters, and how many training trajectories? Is that over-parameterised, and does it matter?"

---

## 4. Training (`train.py`)

### Questions first
1. What is the exact loss: MSE on normalised channels? Per-channel weights? Which channel dominates the raw MSE (see `scripts/data_stats.py` output: `q2dot` carries ~57% of `sum(x^2)`)?
2. Which statistics are used for normalisation, from which split, and where in the code could test data leak into them?
3. Normalise the branch input and the time input too. Why? What goes wrong if `t` is in [0, 3] and torque samples are in [-15, 15] while `m_p` is in [0, 1]?
4. Do you train on all `(sample, time)` pairs per step, or subsample times? What does each cost and bias?
5. How do you choose the epoch to keep, and on which split? Why not the test split?
6. How will you report variance across seeds? One training run is an anecdote.

### Hint ladder (leakage)
- Rung 1: list every number that is *computed from data* (means, stds, min/max, scalers). 
- Rung 2: each must be computed from `train_*` only, saved with the checkpoint, and applied unchanged to val/test/ood.
- Rung 3: write a test that fits the scaler on train, then perturbs the test set wildly and confirms the scaler (and any selected checkpoint) is unchanged.

### Sanity protocol (do all of these before real training)
1. [ ] Overfit 5 samples to near-zero loss for both models. Time it; it should take seconds.
2. [ ] Zero-torque, zero-payload slice: does the model at least learn the easy sub-problem?
3. [ ] Shuffle labels: loss must stay at the "predict the mean" level. If it drops, there is leakage.
4. [ ] Loss curves for train and val on one plot, log y-axis; note where they separate.
5. [ ] Learning-rate sweep over a few decades, 1 epoch each, to find a range (you decide the range; I will question it).
6. [ ] Same tuning budget for every model in the comparison (`REVIEW_NOTES.md` section 4).

### Examiner questions
- "Why that learning rate and batch size? What did you try that failed?"
- "What is your naive baseline error (predict the training-mean trajectory)? Is your model beating it by enough?"
- "Is validation error a good proxy for the test error here? How do you know?"

---

## 5. Evaluation (`evaluate.py`)

### Questions first
1. Define relative L2 error precisely: per trajectory then averaged, or pooled? Which one does the report use, and why? (They differ when a few large trajectories dominate.)
2. Which channels are in the norm? Mixed units: rad vs rad/s. Should `q` and `qdot` be combined at all?
3. What does error vs time look like, and why should it grow? What does a flat-then-jump profile suggest (e.g. spinning, phase slip)?
4. Worst cases: report the 95th/99th percentile and the maximum, not just the mean. What do the worst trajectories have in common (large `A`? small `b`?)
5. Physical consistency: (a) `qdot_pred` vs autograd `d q_pred / dt`; (b) energy *balance* `E(t) - E(0) - integral(tau . qdot - b|qdot|^2) dt`, evaluated on the predicted trajectory. Why is raw energy drift the wrong metric?
6. What is the floor? The RK4 reference itself has error (`REVIEW_NOTES.md` 2.2). A surrogate cannot be judged below that.

### Checks to write yourself
- [ ] Metric sanity: error of the truth against itself = 0; error of zeros = 1 (for relative L2).
- [ ] Error of the *ground truth at substeps 4* against substeps 8: use it to calibrate what "good" looks like.
- [ ] Balance residual of the *true* trajectories is ~0 (up to quadrature error); only then is it meaningful to measure on predictions.

### Examiner questions
- "Your table says 2% error. 2% of what, averaged how, on which split?"
- "What does the OOD number actually show?" (`REVIEW_NOTES.md` 2.3: the OOD set is calmer than train.)
- "What is the worst trajectory and why does the model fail on it?"

---

## 6. Design decisions you must be able to justify (and one-line honest answers to draft)

| Decision | Draft your justification here | Evidence you have |
|---|---|---|
| Vertical plane + gravity | | |
| Torque family (sinusoids) and 8-dim input | | |
| `A`, `f`, `m_p`, `b` ranges | | `scripts/data_stats.py`, `scripts/torque_range_sweep.py` |
| Horizon `T` and initial condition | | |
| `substeps` for ground truth | | convergence test |
| Normalisation scheme | | train-only stats |
| Loss and channel weighting | | |
| DeepONet size (`p`, width, depth) | | validation sweep |
| OOD definition | | |
| Baselines and tuning budget | | |
| Speed comparison and tolerance matched to surrogate error | | `src/benchmark.py` |

## 7. Report/presentation risks (from CLAUDE.md, restated as questions to rehearse)
1. "Is this operator learning or a regression on eight numbers?"
2. "Did the baseline get the same tuning effort?"
3. "What exactly is faster than what, at what accuracy, and how many queries until it pays off?"
4. "Does your OOD test test anything?"
5. "What happens at late times and large angles?"
6. "Is the output physically consistent?"
7. "What does this not show?" (fixed initial condition, simulation only, one arm.)
8. "Which parts did AI write and how do you know they are right?"
