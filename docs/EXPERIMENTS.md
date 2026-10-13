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
(`results/loss_all_runs.png`). MLP with 304-sample input, lr 1e-3 (seeded init, finished 15:50): val loss
0.1952 at epoch 169, median 0.107, p95 0.640, under 10 % 47.7 %: better than every DeepONet run so
far on all of these (`runs.csv`, `results/loss_by_model.png`). But its qdot vs dq/dt consistency is
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

## E4. Fair-fight improvements (code added 2026-10-13, AI-written, NOT YET RUN)

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

Prediction before running (student to fill in): [fill in]

Results: [not run yet]

Decision: [fill in]

## Not yet run

- E4 (architecture / Fourier features / consistency loss) under an equal-effort rule. A single
  `deeponet_samples_lr1e-3_cosine_ff6_seed3001` log exists from another AI session; it is not part of
  the selection and was not used here.
- Data-size learning curve (`scripts/run_datasize.sh`).
- Speed benchmark on an idle machine, with the final checkpoints (B0 is preliminary).

## B0. Speed vs accuracy, first look (2026-10-13 15:37, AI-run, PRELIMINARY)

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
