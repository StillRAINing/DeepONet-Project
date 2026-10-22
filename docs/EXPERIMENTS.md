# Experiment log

One entry per experiment: what was run, why, and what came out. All numbers are on the
**validation** split (n = 1500). The test and OOD splits are untouched until the final evaluation.

- Raw numbers for every run: `results/runs.csv` (regenerate with `py -3.10 -m scripts.summarise_runs`).
- Exact commands: `scripts/run_sweep.sh`. Per-run training logs: `logs/` (not committed).
- Checkpoints: `checkpoints/` (not committed; reproducible from the seed).

## Fixed protocol (applies to every run unless stated)

| Item | Setting |
|---|---|
| Data | `data/dataset_smooth.npz`, 7000 train / 1500 val, seed 3001 |
| Output scaling | per-channel z-score, train statistics only (`ChannelNormaliser`) |
| Input scaling | per-feature z-score, train statistics only (`InputNormaliser`); features constant in training get std 1 |
| Time input | `t / T`, in [0, 1] |
| Loss | per-sample relative L2 on channel-scaled data, batch mean (`relative_l2_loss`) |
| Optimiser | Adam (no weight decay; AdamW considered and not used, student's decision 2026-10-13), batch 100, 200 epochs, seed 3001 |
| Selection | weights with the lowest validation loss over the epochs |
| Headline metric | per-sample relative L2, channels scaled by train std; report median and p95, plus the fraction under 1 / 5 / 10 % |

## Metric columns

`val_loss` is the mean of the per-sample error (what training selects on). `median` and `p95` are
of the same per-sample error. `under_5pct` is the fraction of trajectories below 5 % error. `q2` is
the median own-scale relative L2 of the q2 channel alone. `consistency` is the median of
rms(dq/dt - qdot) / rms(true qdot), the physical-consistency check.

## E0. Naive reference errors (2026-10-13, AI-run, `py -3.10 -m scripts.naive_baselines`)

What a "model" that learns nothing scores on the same metric, so the trained numbers have a yardstick.

| Predictor | median | p95 | max | under 5 % | under 10 % | q2 (own scale) |
|---|---|---|---|---|---|---|
| mean training trajectory (ignores the input) | 0.426 | 0.855 | 1.114 | 0.0 % | 0.0 % | 1.058 |
| 1-nearest neighbour in compact input space | 0.409 | 0.986 | 3.330 | 0.3 % | 4.3 % | 0.828 |
| *for comparison: DeepONet, lr 1e-3 (E1)* | *0.134* | *0.645* | *1.276* | *9.7 %* | *38.5 %* | *0.350* |

Observation: the DeepONet beats both baselines by about 3x at the median, but its p95 (0.645) is only
about 25 % below the mean-trajectory baseline (0.855): on the hardest 5 % of trajectories it has learnt
little more than the average motion. 1-NN being no better than the mean (with a far worse max)
says the map is sensitive: nearby inputs in 26-D coefficient space do not give nearby trajectories.

## E1. Learning rate, reference DeepONet (2026-10-13)

DeepONet, 304 torque samples, p 64, width 128 (171,396 parameters), constant lr.

| lr | best epoch | train at best | val loss | median | p95 | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|---|
| 3e-4 | 198 | 0.220 | 0.2495 | 0.180 | 0.668 | 3.5 % | 23.5 % | 0.297 | 0.271 |
| **1e-3** | 173 | 0.173 | **0.2178** | **0.134** | 0.645 | 9.7 % | 38.5 % | 0.350 | 0.182 |
| 3e-3 | 126 | 0.207 | 0.2390 | 0.157 | 0.703 | 2.9 % | 29.0 % | 0.352 | 0.209 |

Observations (single seed):
- 1e-3 is best by 0.021 to 0.032 in val loss, against epoch-to-epoch noise of about 0.003.
- 3e-4 was still improving at epoch 200 (val slope about -0.0004 per epoch); 1e-3 and 3e-3 had flattened.
- At 1e-3 the train/val gap is 0.169 vs 0.221 at the last epoch.
- Error grows with how violent the trajectory is (val, 1e-3 run): peak |q| 2 to 4 rad median 0.104
  (n = 1205); 4 to 8 rad 0.375 (n = 110); above 8 rad 0.536 (n = 185). Spearman correlation of
  error with peak |qdot| +0.83, with `b` -0.57, with `m_p` -0.04.

Prediction made before the results: "the smaller rate will win, maybe with more epochs; the
fraction under 5 % will rise". Outcome: 1e-3 won at 200 epochs; 3e-4 was undertrained.

Decision: [fill in]

## E2. Cosine learning-rate schedule (queued 2026-10-13)

Question: does decaying the lr (cosine, from the start value to 1/100 of it over 200 epochs) beat
the constant lr of E1? Compare `deeponet / samples / 1e-3 / cosine` against the E1 row at 1e-3, then
every model at 1e-3 and 3e-3 with and without cosine (E3).

Results for the reference DeepONet (val, single seed, unseeded init; filled in by AI from `runs.csv`, 2026-10-13 15:35):

| lr 1e-3 | best epoch | train at best | val loss | median | p95 | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|---|
| constant (E1) | 173 | 0.173 | 0.2178 | 0.134 | **0.645** | 9.7 % | 38.5 % | 0.350 | 0.182 |
| cosine | 198 | 0.171 | **0.2126** | **0.130** | 0.679 | **17.5 %** | **41.1 %** | **0.209** | **0.150** |

Observations (not conclusions):
- The val-loss gap (0.005) is smaller than the gaps between learning rates in E1, and the seed-to-seed
  spread is not measured yet (see the reproducibility note below), so on val loss alone this is not
  yet a demonstrated improvement.
- The gain is concentrated on the easy samples and on q2: the fraction under 5 % almost doubles and
  the q2 median falls from 0.35 to 0.21, while p95 gets slightly worse (0.645 to 0.679). In
  `results/evalfig_val_E1_E2_deeponet_cdf.png` the two CDFs separate below 10 % error and merge above 30 %.
- In `results/evalfig_val_E1_E2_deeponet_regime.png` all four runs have nearly the same median error
  on trajectories with peak |q| > 2 pi (about 0.45 to 0.55): no lr or schedule fixes the spinning regime.

Decision: [fill in]

## Reproducibility note (found 2026-10-13, ~15:30)

Until 15:30 on 2026-10-13, `scripts/train_quick.py` built the model **before** any seed was set
(`fit` seeds, but only after the weights already exist). So every run above has a random
initialisation: "seed 3001" fixed only the mini-batch order and these runs cannot be reproduced bit
for bit. Fixed in `train_quick.py` (`torch.manual_seed(args.seed)` just before `build_model`);
checkpoints trained after the fix store `init_seeded: True` in their history, shown as the
`init_seeded` column in `runs.csv`. The fix does not bias any comparison (both kinds of run start from
an equally random draw), but:
- the E1 and E2 runs, and any E3 runs that started before 15:30, are not exactly reproducible; say so
  in the report or retrain the finalists (the 3-seed step of the selection rule retrains them anyway);
- differences between runs smaller than the seed-to-seed spread mean nothing. The 3-seed retrain
  measures that spread; until then treat val-loss gaps under about 0.01 as unresolved.

## E3. Fair baselines (queued 2026-10-13)

Four models, each with about 171k parameters (`--match-params 171396`, DESIGN_LOG W1), the same lr
grid (3e-4, 1e-3, 3e-3), the same protocol and the same selection rule. Each model is reported at
its own best lr.

| Model | Input | Width | Parameters |
|---|---|---|---|
| DeepONet | 304 torque samples | 128 (p 64) | 171,396 |
| DeepONet | 54 compact (m_p, b, 2 x 2 x 13 Fourier coefficients) | 147 (p 64) | 171,183 |
| MLP | 304 torque samples + t | 192 | 170,692 |
| MLP | 54 compact + t | 229 | 171,754 |

The compact input zero-pads the 6-mode training coefficients to 13 modes, so the `ood_smooth`
set (13 modes, shorter correlation length) can be fed to the compact models. Modes 7 to 13 are
always zero in training, so for those models `ood_smooth` is a genuine extrapolation: the weights
on those inputs were never trained. The sample-input models see the same torque as 151 values and
have no such blind spot. This difference is itself a result to report (the case for function input).

### E3 sweep record (AI-written from the checkpoints, 2026-10-13 23:15)

The full grid of `scripts/run_sweep.sh`, run sequentially 15:12 to ~23:30 on the student's laptop
(CPU). Best-epoch validation loss (mean per-sample relative L2), seed 3001, 7000 training samples,
200 epochs, Adam, about 171k parameters each. **Bold** = lowest val loss for that model = the setting the
selection rule picks. Full metrics per run: `results/runs.csv`.

| Model | Input | lr | constant: val loss (best epoch) | cosine: val loss (best epoch) |
|---|---|---|---|---|
| DeepONet | 304 samples | 3e-4 | 0.2495 (198) | not in grid |
| DeepONet | 304 samples | 1e-3 | 0.2178 (173) | **0.2126** (198) |
| DeepONet | 304 samples | 3e-3 | 0.2390 (126) | 0.2235 (194) |
| DeepONet | compact | 3e-4 | 0.2245 (200) | not in grid |
| DeepONet | compact | 1e-3 | 0.2134 (176) | 0.2140 (198) |
| DeepONet | compact | 3e-3 | 0.2146 (113) | **0.2055** (198) |
| MLP | 304 samples | 3e-4 | 0.1962 (183) | not in grid |
| MLP | 304 samples | 1e-3 | 0.1952 (169) | **0.1732** (200) |
| MLP | 304 samples | 3e-3 | 0.2217 (162) | 0.1822 (199) |
| MLP | compact | 3e-4 | 0.1869 (198) | not in grid |
| MLP | compact | 1e-3 | 0.1846 (189) | 0.1778 (199) |
| MLP | compact | 3e-3 | 0.1907 (138) | **0.1694** (197) |

Train loss at the selected epoch, for the four winners: DeepONet samples 0.171, DeepONet compact
0.146, MLP samples 0.144, MLP compact 0.096 (3e-3 cosine, finished 23:20; the largest train/val gap of the four, 0.074).

How the lead changed during the sweep (the order runs finished in), for the report's narrative:

| Time | Run finished | Best DeepONet so far | Best MLP so far |
|---|---|---|---|
| 14:26 | DeepONet samples 1e-3 (E1, student-run) | 0.2178 | n/a |
| 15:21 | DeepONet samples 1e-3 cosine | 0.2126 | n/a |
| 15:34 | DeepONet compact 1e-3 | 0.2126 | n/a |
| 15:50 | MLP samples 1e-3 | 0.2126 | 0.1952 |
| 16:13 | MLP compact 1e-3 | 0.2126 | 0.1846 |
| 21:41 | MLP samples 1e-3 cosine | 0.2126 | 0.1732 |
| 22:46 | DeepONet compact 3e-3 cosine | 0.2055 | 0.1732 |
| 23:20 | MLP compact 3e-3 cosine (last run) | 0.2055 | 0.1694 |

From the first MLP run onwards an MLP led, and the lead never dropped below about 0.03 in val loss.

Observations (single seed each; not conclusions until the 3-seed retrain):
- **MLP beats DeepONet at matched parameters.** Best MLP 0.169 vs best DeepONet 0.206, a gap of about
  0.035, larger than every other effect in the grid. Anticipated by CLAUDE.md weak point 2.
- **Cosine helps or is neutral in all 8 pairs**: changes in val loss of -0.005, -0.016
  (DeepONet samples, 1e-3 / 3e-3), +0.001, -0.009 (DeepONet compact), -0.022, -0.040 (MLP samples),
  -0.007, -0.021 (MLP compact). The +0.001 is within noise. Cosine also moves the best epoch to the end
  of training (194 to 200 in every cosine run), so with cosine "best epoch" and "last epoch" coincide.
- **lr matters less with cosine.** Constant lr: 3e-3 is clearly worse than 1e-3 for three of four models.
  With cosine the 3e-3 runs recover most of the gap (and win for the compact DeepONet).
- **Compact vs samples input:** for the DeepONet compact is a little better (0.206 vs 0.213); for the
  MLP compact is also a little better (0.169 vs 0.173). Both gaps are under 0.01, below the seed noise
  threshold in the reproducibility note, so unresolved.
- **3e-4 never wins** but was still improving at epoch 200 in every model (best epoch 183 to 200).
- **Overfitting is moderate everywhere**: train/val gaps of about 0.03 to 0.06 at the selected epoch.

Run events to state in the report (they affect timings, not the losses):
- **Throttling.** `mlp_samples_lr1e-3_cosine` took 9348 s (about 75 s per epoch for most of the run)
  against 954 to 1553 s for the other MLP-samples runs. The laptop was throttled from about 19:10 to
  21:40 (cause not established: power saving, sleep or other load). Its loss is unaffected; its
  time must not be used in any speed comparison. All `train_s` values were measured on a shared,
  non-idle laptop and are indicative only.
- **Concurrent edits.** Two other AI sessions edited `src/train.py`, `src/model.py`,
  `scripts/train_quick.py` and `scripts/summarise_runs.py` while the sweep ran (22:59 and 23:01). The
  new options they added (`--fourier`, `--consistency`) default to off; the last run's name and parameter
  count are unchanged, so it ran the same configuration. Checked by the AI at about 23:15.
- **Seeded initialisation changed mid-sweep** (reproducibility note above). Runs before ~15:30 have a
  random init (`init_seeded` False): DeepONet samples at all three constant lrs and at 1e-3 cosine, and
  DeepONet compact 1e-3 constant. All later runs are seeded.
- **Aborted batch.** A first baseline batch with unmatched widths (128 for every model) was started at
  about 14:55 and stopped part-way at ~15:05; no checkpoint from it was kept.
- **Timings, seconds per run (indicative):** DeepONet samples 560 to 980, DeepONet compact 710 to 1250,
  MLP samples 950 to 1550 (excluding the throttled run), MLP compact 1400 to 1880. The MLPs are slower
  to train because they evaluate the whole network at every one of the 151 times; the DeepONet runs the
  branch once per trajectory.

**Selected settings (rule step 1, applied by the AI at 23:20 from `runs_agg.csv`):** MLP compact 3e-3
cosine (0.1694); MLP samples 1e-3 cosine (0.1732); DeepONet compact 3e-3 cosine (0.2055); DeepONet
samples 1e-3 cosine (0.2126). Seeds 1 and 2 for each started 23:21 (`logs/seeds.log`).

Interim (AI note, 15:35, one run only, unseeded init, NOT a result): DeepONet with compact input,
lr 1e-3 constant: val loss 0.2134 at epoch 176, median 0.124, p95 0.685 (reference DeepONet with
304 samples: 0.2178, 0.134, 0.645). Its
train loss is the lowest of all runs so far (~0.155 at epoch 200), so its train/val gap is the largest
(`results/training_curves/loss_all_runs.png`). MLP with 304-sample input, lr 1e-3 (seeded init, finished 15:50): val loss
0.1952 at epoch 169, median 0.107, p95 0.640, under 10 % 47.7 %: better than every DeepONet run so
far on all of these (`runs.csv`, `results/training_curves/loss_by_model.png`). But its qdot vs dq/dt consistency is
the worst (0.241 vs 0.110 to 0.27 for the DeepONets), so 'more accurate' and 'more physical' may disagree.
CLAUDE.md weak point 2 anticipated this; if it holds over the grid and the seeds, it is a legitimate
headline result. Wait for the full grid and the seeds before reading anything into it.

Decision: [fill in]

## Selection rule, fixed before any E2/E3 result was seen (2026-10-13, ~15:30)

Decided by the student ("fixed for now"), written down before the sweep results existed, so that
the test and OOD numbers cannot influence any choice:

1. For each of the four models, the final setting is the (lr, schedule) pair with the lowest
   **validation loss** (`val_loss`, seed 3001) over the grid in `scripts/run_sweep.sh`.
2. That setting is re-trained with seeds 1 and 2. (`bash scripts/run_seeds.sh <model> <input> <lr> <schedule>`) Each model is reported as mean ± std over the
   three seeds (3001, 1, 2).
3. Only then are those 12 checkpoints evaluated, once, on the test split and on every OOD set
   (`ood`, `ood_amp1`, `ood_amp2`, `ood_smooth`). No setting is changed after that evaluation; if
   one is changed later, the report must say so and treat the test numbers as no longer clean.

## Rule step 2. Seeds (2026-10-13 23:20 to 2026-10-14 01:00, AI-run)

The four selected settings re-trained with seeds 1 and 2 (`logs/seeds.log`). Validation, mean ± sample
std over seeds 3001, 1, 2 (full table and gap/spread ratios: `results/analysis_seeds.md`).

| Model | Input | Setting | val loss | median | p95 | under 10 % |
|---|---|---|---|---|---|---|
| MLP | compact | 3e-3 cosine | **0.1703 ± 0.0026** | 0.073 ± 0.001 | 0.641 ± 0.010 | 58.0 ± 0.8 % |
| MLP | samples | 1e-3 cosine | 0.1756 ± 0.0024 | 0.079 ± 0.002 | 0.645 ± 0.014 | 56.2 ± 0.6 % |
| DeepONet | compact | 3e-3 cosine | 0.2017 ± 0.0034 | 0.109 ± 0.002 | 0.688 ± 0.010 | 47.1 ± 0.7 % |
| DeepONet | samples | 1e-3 cosine | 0.2168 ± 0.0039 | 0.133 ± 0.004 | 0.680 ± 0.011 | 39.1 ± 2.5 % |

- Every MLP-vs-DeepONet val-loss gap is 7.7 to 12 times the larger seed std, and the seed ranges do
  not overlap: the MLP win is beyond seed noise.
- Within a family, compact vs samples: MLP 0.0053 apart (about 2 std), DeepONet 0.015 apart (about 4
  std). Compact input is better for both, more clearly for the DeepONet.
- Run events: the two compact-DeepONet seed checkpoints were trained by another AI session at 00:10 and
  00:27 (it ran the same seed batch in parallel); this batch found them and reloaded them instead of
  retraining. Checked: same configuration (171,183 parameters, width 147, p 64, 3e-3 cosine, no
  Fourier or consistency terms, seeded init). The parallel runs shared the CPU, so their `train_s` are
  not comparable.

## Rule step 3. Final evaluation on test and OOD (2026-10-14 ~01:10, AI-run, once)

`py -3.10 -m scripts.final_eval` on the 12 checkpoints above. Settings were fixed before this ran and
must not be changed because of it. Full tables (with max, q2 and consistency for every split):
`results/final_eval.md`; per checkpoint: `results/final_eval.csv`. Mean ± std over the 3 seeds.

**Test (n = 1500)**

| Model | Input | median | p95 | max | under 5 % | under 10 % | consistency | energy excess |
|---|---|---|---|---|---|---|---|---|
| MLP | compact | **0.071 ± 0.000** | **0.600 ± 0.004** | 1.52 ± 0.17 | **39.7 %** | **58.2 %** | 0.244 | **0.162** |
| MLP | samples | 0.080 ± 0.001 | 0.603 ± 0.005 | 1.57 ± 0.04 | 35.1 % | 55.8 % | 0.223 | 0.177 |
| DeepONet | compact | 0.109 ± 0.001 | 0.653 ± 0.013 | 1.54 ± 0.18 | 24.5 % | 47.3 % | **0.109** | 0.196 |
| DeepONet | samples | 0.130 ± 0.007 | 0.630 ± 0.018 | 2.06 ± 0.12 | 15.5 % | 39.3 % | 0.147 | 0.257 |

**OOD, median headline error (p95 in brackets)**

| Model | Input | ood (m_p, b beyond range) | ood_amp1 (A 6 to 9 N m) | ood_amp2 (A 9 to 12 N m) | ood_smooth (rougher torque, 13 modes) |
|---|---|---|---|---|---|
| MLP | compact | 0.052 (0.13) | 0.53 (0.97) | 0.70 (1.11) | 0.220 (0.82) |
| MLP | samples | 0.052 (0.13) | 0.54 (1.00) | 0.78 (1.06) | 0.167 (0.70) |
| DeepONet | compact | 0.060 (0.16) | 0.62 (1.01) | 0.83 (1.11) | 0.212 (0.85) |
| DeepONet | samples | 0.065 (0.19) | 0.61 (1.03) | 0.81 (1.09) | **0.155** (0.72) |

Analysis (AI-written; the student should check each point against the tables before using it):

1. **The validation ranking survives on test.** Same order (MLP compact, MLP samples, DeepONet compact,
   DeepONet samples), test medians within 0.003 of the val medians. The selection did not overfit val.
2. **Accuracy: the MLP wins at matched parameters and tuning.** Test median 0.071 vs 0.109 for the best
   DeepONet; 58 % vs 47 % of trajectories under 10 %. CLAUDE.md weak point 2 predicted this: with an
   input that is really about 26 numbers, a plain MLP is a strong surrogate.
3. **Nobody fixes the tail.** p95 is 0.60 to 0.65 for every model, and max error is 1.5 to 2.1 (worse
   than predicting zero). The hardest ~5 to 10 % of trajectories (the spinning regime, see E1) are not
   learnt by either family. Report this as the main limitation, not just the median.
4. **Physical consistency: the DeepONet wins.** Median qdot-vs-dq/dt mismatch 0.11 to 0.15 for the
   DeepONets vs 0.22 to 0.24 for the MLPs, on test and on every OOD set (the MLP's mismatch roughly
   doubles on the amplitude shells; the DeepONet's does not move). So "more accurate" and "more
   physically consistent" disagree. The energy-balance excess follows accuracy instead (MLP compact
   best), because it is dominated by how wrong the trajectory is.
5. **"OOD" in m_p and b is not hard.** Errors on `ood` (median 0.05 to 0.07) are lower than on test.
   Higher payload and friction make the motion calmer, and calm trajectories are the easy ones. This
   set tests a smooth, small change and should be described that way (CLAUDE.md weak point 4).
6. **Torque amplitude beyond the training range breaks every model.** Median error 0.53 to 0.62 in the
   first shell and 0.70 to 0.83 in the second; 0 to 0.8 % of trajectories under 10 %. Error grows with
   distance beyond the range: a degradation curve, not a pass/fail. These shells are mostly spinning
   trajectories (DESIGN_LOG A7), so this is also a regime change.
7. **The function-input encoding helps on a new torque family.** On `ood_smooth` both sample-input
   models beat their compact twins (MLP 0.167 vs 0.220, DeepONet 0.155 vs 0.212), the reverse of the
   in-distribution result. The compact models have never trained the weights on Fourier modes 7 to 13,
   while the sample-input models see the torque directly. The best model here is the DeepONet with
   samples (0.155 ± 0.009 vs MLP samples 0.167 ± 0.013): a gap of about 1 std, so **unresolved**; do
   not claim the DeepONet wins it. What is supported: sampled-function input generalises better to a
   rougher torque than the compact coefficients, for both families.
8. **Honest one-line summary for the report:** at equal parameters and tuning, an MLP is the more
   accurate surrogate in distribution; the DeepONet is more physically consistent; both fail on the
   spinning regime and on larger torques; feeding the torque as samples rather than coefficients pays
   off only when the torque family changes.

Caveats: one data split; 3 seeds; ~171k parameters only (no size sweep); 200 epochs; validation and
test come from the same distribution; `ood_*` sets have 200 samples each, so their percentages move in
steps of 0.5 %. Timings are not part of this evaluation (see B0, still preliminary).

Decision (student): [fill in: which model is the headline, and how the report frames points 2, 4 and 7]

## E4. Fair-fight improvements (code added 2026-10-13, AI-written; run 2026-10-14 to 15, results below)

Question: can the DeepONet beat the MLP under an equal-effort protocol? Every change below is
available to both families (same flag, same budget, same selection on validation relative L2).

| Change | Flag / script | What it targets |
|---|---|---|
| Fourier time features `[t, sin k pi t, cos k pi t]`, k = 1..K | `--fourier K` | spectral bias of a tanh net on a scalar `t` (trunk input / MLP input) |
| qdot = dq/dt self-consistency term in the training loss | `--consistency w` | the consistency gap (the MLP is the worst); costs ~2-3x time per epoch |
| Seeds, mean +- std per setting | `scripts/run_seeds.sh`, then `summarise_runs` -> `results/runs_agg.csv` | whether any gap exceeds seed noise |
| Data-size learning curve | `scripts/run_datasize.sh <lr> <sched>` | does the ranking change with n = 500..7000? |
| Off-grid query times | `scripts/offgrid_eval.py <ckpts>` | interpolation in t. Both models take t continuously, so this is NOT a DeepONet-specific test |
| Variants for both families | `scripts/run_improve.sh <lr> <sched>` | ff6, cons 0.1, both; each differs from the base by one thing |

Prediction before running (student to fill in): [fill in] (AI_LOG row 29 correction records it: "Fourier
features will improve the trunk and might let the DeepONet beat the MLP".)

### E4 results (AI-written 2026-10-15 from `runs_agg.csv`, `final_eval_ff6*.csv`; tables in `results/final_analysis.md`)

Runs: ff6 = `--fourier 6` for all four models at each model's selected lr, 3 seeds
(`scripts/run_ff6_psweep.sh`); ff6 + cons = Fourier features plus the qdot = dq/dt penalty at weight 0.1, samples
input, both families, 3 seeds (`scripts/run_cons_seeds.sh`); cons only, 1 seed per family; data size n = 500 to
7000, lr 1e-3 cosine, 1 seed (`scripts/run_datasize.sh`); DeepONet p sweep 8 to 256 at equal budget and at fixed
width 128, compact input, 1 seed (`run_ff6_psweep.sh`, `run_psweep_fixedwidth.sh`). Selection and comparison on
**validation** only. The ff6 and ff6 + cons checkpoints were also evaluated on test/OOD afterwards
(`results/final_eval_ff6.md`, `final_eval_ff6cons.md`), as reported variants, not a re-selection.

Validation, mean +- std over 3 seeds (`results/e4_variants.png`):

| model | input | base val loss | ff6 | ff6 + cons 0.1 | base consistency | ff6 + cons consistency |
|---|---|---|---|---|---|---|
| MLP | coef | 0.1703 +- 0.0026 | 0.1756 +- 0.0002 | not run | 0.244 | (ff6: 0.421) |
| MLP | samples | 0.1756 +- 0.0024 | 0.1696 +- 0.0006 | **0.1655 +- 0.0008** | 0.222 | 0.115 |
| DeepONet | coef | 0.2017 +- 0.0034 | 0.2006 +- 0.0009 | not run | 0.109 | (ff6: 0.058) |
| DeepONet | samples | 0.2168 +- 0.0039 | 0.1958 +- 0.0018 | 0.1974 +- 0.0002 | 0.149 | **0.008** |

Observations:
1. **Fourier time features help the samples-input models clearly** (DeepONet -0.021 in val loss, about 5 seed
   std; MLP -0.006, about 2.5 std) and the compact ones not at all (MLP coef is slightly worse, +0.005).
   Half of the student's prediction held: the trunk improved, but **the MLP still wins**. Best DeepONet
   (0.196) vs best MLP (0.166): the gap shrank from about 0.031 to 0.030 in val loss. No E4 change closes it.
2. **The consistency penalty works and costs nothing in accuracy.** DeepONet: consistency 0.071 -> 0.008
   (9x), val loss unchanged within noise. MLP: 0.303 -> 0.115, val loss slightly better. With the penalty
   both families are self-consistent; without it the DeepONet is the more consistent one (it was 0.11 to
   0.15 vs the MLP's 0.22 to 0.24 at base). Fourier features make the MLP *less* consistent (0.22 -> 0.30
   samples, 0.24 -> 0.42 coef): the extra high-frequency inputs let q wiggle where qdot does not.
3. **Test agrees with validation.** ff6 + cons on test: MLP samples median 0.066 +- 0.002 (best of every
   configuration evaluated), DeepONet samples 0.109 +- 0.002. Rule-selected best: MLP coef 0.071. No variant
   changes the p95 (0.58 to 0.66 everywhere) or the amplitude extrapolation (0.53 to 0.62 on `ood_amp1`).
4. **Data size** (`results/e4_datasize.png`): median error falls about as n^-0.5 (slopes -0.43 to -0.55)
   with no saturation at 7000; p95 falls only 0.77-0.82 -> 0.64-0.68. MLPs lead at every n >= 1000.
   The DeepONet does **not** overtake at small n (at n = 500 the four are within 0.06 of each other, MLP coef still best).
5. **p sweep** (`results/e4_psweep.png`): p = 8 is clearly worse (val loss 0.25, best epoch 49 to 66);
   p = 16 to 128 is a plateau within about 2 seed std; at equal budget p = 256 (width 71) is worse. p = 32
   is marginally lowest (0.199 vs 0.206 at p = 64, one seed). p = 64 is a defensible choice (W2).

Decision: [fill in: (a) does the headline stay the rule-selected four (recommended: yes; report E4 as
follow-up), (b) whether to quote the ff6 + cons MLP as "best configuration found" with the caveat that test was
seen for it]

## B1. Speed benchmark, final (2026-10-15 21:28, AI-run, `py -3.10 -m scripts.run_benchmark`)

Supersedes B0 and the 2026-10-14 contended run. Full tables: `results/benchmark.md`; figures
`results/bench_pareto.png`, `bench_batch.png`, `bench_breakeven.png`; break-even vs batched RK4 in
`results/final_analysis.md`. No training or other benchmark ran alongside; background desktop load was 30 to 35 %
(see the banner in `benchmark.md`), so absolute ms may be up to ~20 to 30 % high.

| Method (64 val trajectories) | ms / traj | median err | p95 err |
|---|---|---|---|
| scalar Python RK4 (made the labels), one at a time | 266 | (truth) | |
| vectorised NumPy RK4, batch 64 / batch 1500 | 8.7 / 1.48 | exact (1e-15) | |
| `solve_ivp` RK45 rtol 0.03 / 0.05 (matched to surrogate median) | 14.0 / 14.5 | 0.075 / 0.116 | 0.71 / 1.05 |
| `solve_ivp` RK45 rtol 1e-3 | 20.6 | 0.002 | 0.067 |
| MLP coef / samples, batch 64 | 0.37 / 0.32 | 0.074 / 0.071 | 0.60 / 0.55 |
| DeepONet coef / samples, batch 64 | 0.23 / 0.21 | 0.097 / 0.115 | 0.64 / 0.66 |
| DeepONet, trunk cached, batch 64 | 0.03 | same | same |

Observations:
1. **The fair rival is batched RK4, not `solve_ivp`.** Vectorised RK4 is exact and already faster than
   `solve_ivp` at the surrogates' (poor) error level; `solve_ivp` at rtol 0.08 to 0.1 even fails on 1 to 2 of
   64 trajectories. At matched *median* error `solve_ivp`'s p95 is similar to or worse than the surrogates'.
2. **Speed-up shrinks with batch size**: about 170 to 310x at batch 1, 27 to 52x at batch 64, 5 to 11x at
   batch 1500 (un-cached surrogates). The DeepONet with its trunk cached stays about 290 to 400x faster than RK4 at every batch,
   because a query then costs one branch pass; the MLP cannot do this. This is the one clear DeepONet win.
3. **Training dominates the up-front cost**: 800 to 1700 s per run (idle), data generation 17 s vectorised
   (2,660 s with the scalar solver actually used). Break-even: 57k to 128k queries vs matched `solve_ivp`;
   0.56M to 1.4M vs batched RK4 at batch 1500; about 5x more if the 5-run lr grid is counted.
4. FLOPs per trajectory: MLP 5.2e7, DeepONet 2.0 to 2.5e7, DeepONet cached 2.6e5 to 2.9e5. Parameter memory
   0.69 MB for every model.

Run events: first rerun crashed in the CSV writer after measuring (fixed, rerun); the earlier
2026-10-14 run was contended at 100 % CPU and is superseded.

## B2. Speed benchmark, quiet rerun (2026-10-21 16:29 to 16:31, AI-run, `py -3.10 -m scripts.run_benchmark`)

Supersedes B1 (student's decision 2026-10-21: rerun before quoting milliseconds). Conditions: IDE open, every
other desktop program closed; system CPU outside the process 20 % before, 14 % after (B1: 30 to 35 %). Same code,
same checkpoints, same 64 validation trajectories, so errors are identical to B1; only times changed. Snapshot of
the B1 files is in commit a7f41b7. `scripts/final_analysis.py` rerun afterwards to redraw `bench_*.png` and the
break-even table (its "Reading" sentence was hardcoded to the B1 ranges; now computed from the data).

| Method (64 val trajectories) | ms / traj (B2) | B1 for comparison |
|---|---|---|
| scalar Python RK4 (made the labels), one at a time | 121 | 266 |
| vectorised NumPy RK4, batch 64 / batch 1500 | 4.4 / 0.73 | 8.7 / 1.48 |
| `solve_ivp` RK45 rtol 0.03 / 0.05 (matched to surrogate median) | 7.6 / 8.1 | 14.0 / 14.5 |
| MLP coef / samples, batch 64 | 0.27 / 0.28 | 0.37 / 0.32 |
| DeepONet coef / samples, batch 64 | 0.18 / 0.14 | 0.23 / 0.21 |
| DeepONet, trunk cached, batch 64 (default threads / 1 thread) | 0.02 / 0.007 | 0.03 / - |

Observations:
1. Everything got faster, but the **solvers about 2x and the surrogates only 1.2 to 1.5x**, so the background load
   in B1 had flattered the surrogates. The speed-ups and break-even numbers move against the surrogates.
2. Speed-up vs vectorised RK4 (un-cached surrogates): 137 to 161x at batch 1, 17 to 35x at batch 64, 3 to 7x at
   batch 1500. DeepONet with cached trunk: 201 to 359x at every batch.
3. Training (idle, scaled from 3 epochs): 556 to 1490 s per run; data generation 8 s vectorised (1,206 s scalar).
   Break-even: **70k to 205k** queries vs matched `solve_ivp`; **0.8M to 3.0M** vs batched RK4 at batch 1500
   (3.8M to 14.9M with the 5-run lr grid). B1 had 57k to 128k and 0.56M to 1.4M.
4. FLOPs and memory unchanged (they are not timings).

Caveat that still applies: one laptop, one run; repeated runs on this machine differ by ~10 to 20 %. B1 vs B2
differ by ~2x, which is the background load, not noise.

## E5. Final analysis of the selected models (2026-10-15, AI-run, `py -3.10 -m scripts.final_analysis`)

After the selection rule; nothing chosen here. Tables: `results/final_analysis.md`. Figures:
`results/final_overview.png`, `final_consistency.png` (#14), `final_ood_degradation.png` (#15),
`final_seeds.png` (#18), `final_sensitivity.png`, `evalfig_{test,ood_amp1,ood_smooth}_final4_*.png`
(#10 to #13, #17 on test and two OOD sets).

Findings (seed 3001 per-sample, unless "3 seeds"):
1. **Where the error lives.** By regime (peak |q|): < pi (67 % of test) median 0.045 to 0.089; pi to 2 pi
   0.12 to 0.18; > 2 pi 0.37 to 0.54 for every model. By time: the median error grows about 4x from
   t = 1 s to t = 3 s; the p90 grows 6 to 9x. By channel (3 seeds): q1 0.03, q2 0.16 to 0.21, q1dot 0.08 to
   0.15, q2dot 0.19 to 0.35: the elbow and the velocities are the hard part.
2. **Naive yardsticks on test**: mean training trajectory median 0.43 (p95 0.87); 1-NN 0.42 (p95 1.01).
   Every model beats them 3 to 6x at the median, but only by ~25 % at p95.
3. **The tail is a property of the map, not only of the networks.** Perturbing every input by ~1 % moves the
   true trajectory by 0.6 % (median) for calm samples but 4 to 8 % (p90 32 to 56 %) for spinning ones;
   Spearman correlation between that sensitivity and the MLP's error is +0.81
   (`results/final_sensitivity.png`). Correlation, not proof.
4. **Out of range.** Amplitude: error grows ~20x across the training range itself, roughly doubles across
   the 6 N m edge, then saturates at 0.6 to 0.8 (no cliff). m_p, b: once matched on trajectory size, OOD
   errors equal in-range ones within +-25 % (a mild shift). Rougher torque: samples input beats coef input
   for both families (3 seeds, beyond seed noise), the one place where function-as-samples input pays off.
5. **Physics.** Consistency (rms(dq/dt - qdot) / rms(qdot)): DeepONet 0.10 to 0.15, MLP 0.23 to 0.26; with the
   E4 penalty 0.009 and 0.12. Energy-balance excess follows accuracy (MLP 0.15 to 0.17, DeepONet 0.21 to 0.25
   of the energy swing), and its p95 exceeds 1 for every model: on hard trajectories the predictions do not
   obey the energy balance at all.
6. **Initial condition (W5).** Median |x_pred(0)| is 0.3 to 1.3 % of the channel rms for positions and 1.3 to
   6.3 % for velocities; worst for the MLP with samples input. Not enforced by any model.
7. **Seeds** (`final_seeds.png`): the seed std of the test median is 0.000 to 0.007, far below the
   MLP-vs-DeepONet gap (0.03 to 0.06); p95 spread 0.004 to 0.018.

## V1. Verification of the ground truth and the tests (2026-10-15, AI-run)

- `scripts/solver_figures.py` -> `results/solver_checks.md`, FIGURE_PLAN #2 to #4. RK4 order 4.00 to 4.12 on
  y' = -y; on a calm arm sample orders approach 4 (3.83, 3.92, 3.96 at the smallest h); on the most violent
  test sample the coarse steps are not yet asymptotic and mid-range orders are about 5. Energy with b = 0,
  tau = 0: drift 4.3e-5 J at the dataset step (relative 2e-6); 16x-or-more per halving. Energy balance with
  torque + friction: residual 1.2e-3 J on the 2.5 ms grid vs 7.4e-2 J on the 20 ms output grid (the floor the
  evaluation subtracts). Labels vs DOP853: 3e-8 (calm), 5e-5 (most violent test), 2.4e-2 (most violent
  ood_amp2).
- `scripts/mutation_check.py` -> `results/mutation_check.md`: 10 deliberate bugs, one at a time, in a temp copy.
  9 of 10 are caught by a targeted test (not only by the cross-implementation checks in `test_benchmark.py`).
  The three dynamics/solver survivors of the 2026-10-12 review are now caught: both parallel-axis bugs by
  `test_payload_inertia_matches_brute_force_at_mid_range_payload`, the perturbed RK4 stage by the exact-solution,
  order and `solve_ivp` agreement tests. Still missed: the phase range halved in the **sine** family sampler
  (diagnostics only; the smooth family has no phase input).

## Not yet run

Nothing required. Optional: rerun `scripts/run_benchmark.py` with desktop programs closed (3 minutes) to
confirm the ms figures; a seed-1/2 replicate of the data-size and p sweeps.

## B0. Speed vs accuracy, first look (2026-10-13 15:37, AI-run, PRELIMINARY; superseded by B1)

`python -m src.benchmark --checkpoint <ckpt>` (added today; smooth family, **val** split, the
surrogate time includes building, normalising and decoding the input/output). Run on 8 to 16 val
trajectories **while the training sweep was using the CPU**, so the times below are only rough
orders of magnitude. Rerun on an idle machine before quoting anything. The error column is valid (it
does not depend on load): median headline relative L2 against the stored RK4 truth.

| Method (E1 DeepONet, lr 1e-3) | ms / trajectory (contended) | median rel. L2 |
|---|---|---|
| DeepONet, batch of 16 | ~0.4 | 0.087 (these 16 samples) |
| DeepONet, single query | ~2 | same |
| vectorised RK4, 8 substeps (= the truth) | ~38 | 0 |
| `solve_ivp` RK45, rtol 1e-1 | not timed | 0.58 |
| `solve_ivp` RK45, rtol 1e-2 | ~18 | 0.014 |
| `solve_ivp` RK45, rtol 1e-3 | ~24 | 0.0013 |
| `solve_ivp` RK45, rtol 1e-6 | ~80 | 3e-7 |

Observations, not conclusions:
- `solve_ivp` at rtol 1e-2 is already ~6x **more accurate** than the surrogate, and at rtol 1e-1 it is
  far worse. The matched-accuracy solver lies between the two (rtol 3e-2 is now in the default grid).
  The fair speed-up is against that one, not against rtol 1e-6 or the 8-substep truth.
- Break-even (weak point 3): at a gain of ~15 to 20 ms per query, the ~560 s training run alone needs
  roughly 30,000 queries to pay off, before counting data generation (10,000 RK4 trajectories) or
  the lr / model sweep. Use `break_even_queries` with idle-machine numbers.

## W5 check. Initial-condition error at t = 0 (2026-10-13 ~15:40, AI-run, val, throwaway snippet)

Every true trajectory starts at x0 = 0, and nothing in the model or the loss forces the prediction to.
|prediction at t = 0| in raw units, per channel [q1, q2, q1dot, q2dot] (rad, rad, rad/s, rad/s):

| Run | median | p95 | median as a fraction of the channel rms |
|---|---|---|---|
| DeepONet samples, lr 1e-3 | 0.019, 0.069, 0.112, 0.293 | 0.089, 0.278, 0.512, 1.498 | 1.0 %, 2.1 %, 3.4 %, 5.7 % |
| DeepONet samples, lr 1e-3, cosine | 0.015, 0.037, 0.097, 0.237 | 0.059, 0.131, 0.544, 1.056 | 0.8 %, 1.1 %, 3.0 %, 4.6 % |
| DeepONet coef, lr 1e-3 | 0.015, 0.050, 0.094, 0.185 | 0.062, 0.205, 0.280, 0.710 | 0.8 %, 1.5 %, 2.9 %, 3.6 % |

(channel rms over val: 1.94, 3.31, 3.26, 5.14.) The start is wrong by a few percent of the
channel size, worst on q2dot. It is visible in `results/evalfig_val_E1_E2_deeponet_cases.png` (the
q2dot predictions do not start at 0). Whether to fix it architecturally (O5) is your decision. If you
look into it, ask: what simple change to the output makes x(0) = x0 hold exactly for every input,
and what does it do to the `qdot = dq/dt` consistency near t = 0?

## Truth check. Label accuracy on every split (2026-10-13 ~15:41, AI-run, `py -3.10 -m scripts.truth_check`)

Stored RK4 (8 substeps) vs DOP853 at rtol 1e-10, 20 random samples per split (seed 3001), headline
metric. No model involved, so this does not touch model selection. Extends DESIGN_LOG A8 (train only).

| split | train | val | test | ood | ood_amp1 | ood_amp2 | ood_smooth |
|---|---|---|---|---|---|---|---|
| median | 6.7e-9 | 9.1e-9 | 1.4e-8 | 3.3e-9 | 3.7e-8 | 1.6e-6 | 6.3e-9 |
| max | 7.0e-6 | 8.1e-7 | 4.1e-6 | 9.4e-9 | 6.8e-4 | 2.0e-3 | 6.3e-7 |

Reading: the label floor is 2 to 8 orders of magnitude below the surrogate errors (~0.1), on every
split including the spinning amplitude shells. Label error cannot explain any surrogate error here.
Only 20 samples per split: the max is not the true worst case.
