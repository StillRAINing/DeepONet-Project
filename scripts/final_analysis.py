"""Final analysis: every remaining report figure and table, from the fixed checkpoints and result files.

    py -3.10 -m scripts.final_analysis            # about 5 to 10 minutes on a laptop CPU
    py -3.10 -m scripts.final_analysis --no-cache # recompute the per-sample arrays

Runs AFTER the selection rule (docs/EXPERIMENTS.md): nothing here chooses a model or a setting. It reads
  * the rule-selected checkpoints (``final_eval.FINAL``) and the E4 follow-up ``FINAL_FF6CONS``,
    seed 3001 for per-sample figures, all three seeds via the CSVs;
  * results/final_eval*.csv (test / OOD, per seed), results/runs.csv and runs_agg.csv (validation),
    results/benchmark.json (speed).
Writes results/final_*.png, results/e4_*.png, results/bench_*.png and results/final_analysis.md.
Per-sample arrays are cached in results/final_persample.npz (git-ignored).

Figures (FIGURE_PLAN numbers in brackets):
  final_overview        median and p95 per split, four models, seed mean +- std        (#17 summary)
  final_consistency     qdot vs dq/dt and energy-balance excess on test                  (#14)
  final_ood_degradation error vs torque amplitude across the range edge; OOD m_p/b vs a
                        magnitude-matched test subset; rougher-torque set by input type  (#15)
  final_seeds           every seed's test median and p95, per model and variant          (#18)
  final_sensitivity     how far a 1 % input change moves the TRUE trajectory, vs surrogate error
  e4_variants           Fourier features / consistency penalty, validation, 3 seeds      (E4)
  e4_datasize           validation error vs number of training trajectories              (E4)
  e4_psweep             DeepONet basis size p, equal parameter budget and fixed width    (W2)
  bench_pareto, bench_batch, bench_breakeven                                             (#16)

Plotting / analysis glue written by the AI (AI_LOG row to add). Every metric comes from src/evaluate.py.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from src import benchmark as bm
from src.data import DATA_PATH_SMOOTH, T_FINAL, T_GRID, X0, load
from src.evaluate import (energy_balance_summary, error_vs_time, predict, qdot_consistency,
                          rel_l2_per_sample)
from src.train import load_checkpoint, make_split

from .final_eval import FINAL, FINAL_FF6CONS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = RES / "final_persample.npz"
SEED = 3001
SPLITS = ("test", "ood", "ood_amp1", "ood_amp2", "ood_smooth")
SPLIT_NAMES = {"test": "test\n(in range)", "ood": "ood\n(m_p, b beyond)", "ood_amp1": "ood_amp1\n(peak 6-9 N m)",
               "ood_amp2": "ood_amp2\n(peak 9-12 N m)", "ood_smooth": "ood_smooth\n(rougher torque)"}
MODELS = {  # tag -> (label, checkpoint stem, colour, marker)
    "mlp/coef": ("MLP, coef", FINAL[("mlp", "coef")], "#0072B2", "o"),
    "mlp/samples": ("MLP, samples", FINAL[("mlp", "samples")], "#56B4E9", "s"),
    "deeponet/coef": ("DeepONet, coef", FINAL[("deeponet", "coef")], "#D55E00", "D"),
    "deeponet/samples": ("DeepONet, samples", FINAL[("deeponet", "samples")], "#E69F00", "^"),
}
CONS = {
    "mlp/samples/ff6cons": ("MLP, samples, ff6 + cons", FINAL_FF6CONS[("mlp", "samples")], "#56B4E9", "s"),
    "deeponet/samples/ff6cons": ("DeepONet, samples, ff6 + cons", FINAL_FF6CONS[("deeponet", "samples")], "#E69F00", "^"),
}
ALL = {**MODELS, **CONS}
CH = ("q1", "q2", "q1dot", "q2dot")


def ckpt(stem, seed=SEED):
    return ROOT / "checkpoints" / f"{stem.format(s=seed)}.pt"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style(ax, log=None):
    ax.grid(alpha=0.25, which="both")
    if log:
        getattr(ax, f"set_{log}scale")("log")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RES / name, dpi=150)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------------------------- per-sample arrays
def compute_persample(d):
    """Per-sample arrays for every model in ALL (seed 3001) on every split; physics on test only."""
    out = {}
    std = d["train_traj"].reshape(-1, 4).std(0)
    for tag, (_, stem, _, _) in ALL.items():
        model, in_norm, out_norm, h = load_checkpoint(ckpt(stem))
        mode = h.get("input_mode", "samples")
        for split in SPLITS:
            u, t, _ = make_split(d, split, in_norm, out_norm, T_FINAL, mode)
            true = d[f"{split}_traj"].astype(np.float64)
            pred = predict(model, u, t, out_norm)
            out[f"{tag}|{split}|err"] = rel_l2_per_sample(pred, true, out_norm.std.numpy())
            if split == "test":
                out[f"{tag}|test|err_time"] = error_vs_time(pred, true, out_norm.std.numpy())
                res = qdot_consistency(model, u, t, out_norm)
                out[f"{tag}|test|cons"] = (np.sqrt((res ** 2).mean(axis=(1, 2)))
                                           / np.sqrt((true[..., 2:] ** 2).mean(axis=(1, 2))))
                eb = energy_balance_summary(pred, true, d["test_params"], d["test_torque"], d["t_grid"])
                out[f"{tag}|test|energy"] = eb["excess_rel"]
                out[f"{tag}|test|ic"] = np.abs(pred[:, 0, :])           # |prediction at t = 0|, raw units
        print("per-sample arrays:", tag)
    out["channel_rms"] = np.sqrt((d["test_traj"].astype(np.float64) ** 2).mean(axis=(0, 1)))
    out["std"] = std
    return out


def naive_baselines(d, L):
    """Mean-trajectory and 1-NN reference errors on TEST (no training, train statistics only)."""
    from .naive_baselines import compact
    train, test = d["train_traj"].astype(np.float64), d["test_traj"].astype(np.float64)
    std = train.std(axis=(0, 1))
    e_mean = rel_l2_per_sample(np.broadcast_to(train.mean(axis=0), test.shape), test, std)
    xt, xs = compact(d, "train"), compact(d, "test")
    mu, sd = xt.mean(0), xt.std(0)
    xt, xs = (xt - mu) / sd, (xs - mu) / sd
    nn = np.array([np.argmin(((xt - x) ** 2).sum(1)) for x in xs])
    e_nn = rel_l2_per_sample(train[nn], test, std)
    L += ["## Naive reference errors on test (no model)", "",
          "| predictor | median | p95 | under 10 % |", "|---|---|---|---|"]
    for name, e in (("mean training trajectory", e_mean), ("1-nearest neighbour, compact input", e_nn)):
        L.append(f"| {name} | {np.median(e):.3f} | {np.percentile(e, 95):.3f} | {(e < 0.1).mean():.1%} |")
    L.append("")
    return {"mean": e_mean, "nn": e_nn}


# ------------------------------------------------------------------------------------- figures from CSVs
def agg(rows, key, model, inp, split):
    v = np.array([float(r[key]) for r in rows if r["model"] == model and r["input"] == inp and r["split"] == split])
    return v.mean(), v.std(ddof=1), v


def fig_overview(fe, L):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(SPLITS))
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for k, (tag, (label, _, c, m)) in enumerate(MODELS.items()):
        model, inp = tag.split("/")
        for a, key in zip(ax, ("median", "p95")):
            mu = [agg(fe, key, model, inp, s)[0] for s in SPLITS]
            sd = [agg(fe, key, model, inp, s)[1] for s in SPLITS]
            a.errorbar(x + off[k], mu, yerr=sd, fmt=m, color=c, ms=7, capsize=3, lw=1.5, label=label)
    for a, key in zip(ax, ("median", "p95")):
        a.set_xticks(x, [SPLIT_NAMES[s] for s in SPLITS], fontsize=8)
        a.set_ylabel(f"{key} per-sample relative L2")
        a.set_title(f"{key} error (mean +- std over seeds 3001, 1, 2)")
        style(a, "y")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rule-selected models on test and the four out-of-range sets (171k parameters each)")
    save(fig, "final_overview.png")


def fig_seeds(variants, L):
    """#18: every seed as a dot. variants: {name: rows of final_eval*.csv}."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xt, xl, x = [], [], 0
    for tag, (label, _, c, m) in MODELS.items():
        model, inp = tag.split("/")
        for vk, (vname, rows) in enumerate(variants.items()):
            sel = [r for r in rows if r["model"] == model and r["input"] == inp and r["split"] == "test"]
            if not sel:
                continue
            for a, key in zip(ax, ("median", "p95")):
                v = [float(r[key]) for r in sel]
                a.scatter([x] * len(v), v, color=c, marker=m, s=40, facecolors=c if vk == 0 else "none", lw=1.5)
                a.hlines(np.mean(v), x - 0.3, x + 0.3, color=c, lw=1)
            xt.append(x)
            xl.append(f"{label}\n{vname}")
            x += 1
        x += 0.6
    for a, key in zip(ax, ("median", "p95")):
        a.set_xticks(xt, xl, fontsize=6.5, rotation=60, ha="right")
        a.set_ylabel(f"test {key} relative L2")
        a.set_title(f"test {key}: one dot per training seed, line = mean")
        style(a)
    fig.suptitle("Seed-to-seed spread (filled = rule-selected setting, hollow = E4 variants)")
    save(fig, "final_seeds.png")


def fig_e4(agg_rows, runs, L):
    """E4 variants on validation, from runs_agg.csv (3 seeds) plus the single-seed consistency-only runs."""
    def pick(model, inp, nf, cw, n=7000, seeds=None):
        cand = [r for r in agg_rows if r["model"] == model and r["input"] == inp and int(r["n_fourier"]) == nf
                and float(r["cons_weight"]) == cw and int(r["n_train"]) == n and abs(int(r["params"]) - 171396) < 1500
                and (seeds is None or int(r["n_seeds"]) == seeds)]
        if not cand:
            return None
        if seeds is None and nf == 0 and cw == 0:   # base: the rule-selected 3-seed setting
            cand = [r for r in cand if int(r["n_seeds"]) == 3]
        return min(cand, key=lambda r: float(r["val_loss_mean"]))
    variants = [("base", 0, 0.0), ("cons 0.1 (1 seed)", 0, 0.1), ("ff6", 6, 0.0), ("ff6 + cons 0.1", 6, 0.1)]
    keys = [("val_loss", "val loss"), ("median", "median"), ("p95", "p95"), ("consistency", "qdot vs dq/dt")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    L += ["## E4 variants on validation (mean +- std over seeds where 3 exist)", "",
          "| model | input | variant | lr | seeds | val loss | median | p95 | under 10 % | q2 | consistency |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    xt, xl, x = [], [], 0
    for tag, (label, _, c, m) in MODELS.items():
        model, inp = tag.split("/")
        for vk, (vname, nf, cw) in enumerate(variants):
            r = pick(model, inp, nf, cw)
            if r is None:
                continue
            for a, (key, title) in zip(ax, keys):
                mu, sd = float(r[f"{key}_mean"]), float(r[f"{key}_std"] or 0)
                a.bar(x, mu, yerr=sd if r[f"{key}_std"] else None, color=c, alpha=[1, 0.35, 0.6, 0.8][vk],
                      hatch=["", "..", "//", "xx"][vk], edgecolor="#333333", lw=0.5, capsize=3)
            xt.append(x)
            xl.append(f"{label}\n{vname}")
            x += 1
            def f(k, pct=False):
                mu, sd = float(r[f"{k}_mean"]), r[f"{k}_std"]
                s = (f"{mu:.1%}" if pct else f"{mu:.4f}")
                return s + ("" if not sd else (f" +- {float(sd):.1%}" if pct else f" +- {float(sd):.4f}"))
            L.append(f"| {model} | {inp} | {vname} | {r['lr']} | {r['n_seeds']} | {f('val_loss')} | {f('median')} | "
                     f"{f('p95')} | {f('under_10pct', True)} | {f('q2')} | {f('consistency')} |")
        x += 0.8
    for a, (key, title) in zip(ax, keys):
        a.set_xticks(xt, xl, fontsize=6, rotation=70, ha="right")
        a.set_title(f"validation {title}")
        style(a)
    ax[3].set_yscale("log")
    fig.suptitle("Equal-effort improvements, available to both families (validation, n = 1500)")
    save(fig, "e4_variants.png")
    L.append("")


def fig_datasize(runs, L):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    L += ["## Data-size learning curve (validation, seed 3001, lr 1e-3 cosine, 200 epochs)", "",
          "| model | input | " + " | ".join(f"n = {n}" for n in (500, 1000, 2000, 3500, 7000)) + " | slope of log median vs log n (n >= 1000) |",
          "|---|---|---|---|---|---|---|---|"]
    p95s = []
    for tag, (label, _, c, m) in MODELS.items():
        model, inp = tag.split("/")
        sel = sorted([r for r in runs if r["model"] == model and r["input"] == inp and r["lr"] == "0.001"
                      and r["schedule"] == "cosine" and int(r["n_fourier"]) == 0 and float(r["cons_weight"]) == 0
                      and r["seed"] == "3001" and abs(int(r["params"]) - 171396) < 1500], key=lambda r: int(r["n_train"]))
        n = np.array([int(r["n_train"]) for r in sel])
        med, p95 = np.array([float(r["median"]) for r in sel]), np.array([float(r["p95"]) for r in sel])
        ax[0].loglog(n, med, marker=m, color=c, lw=2, label=label)
        ax[1].loglog(n, p95, marker=m, color=c, lw=2, label=label)
        k = n >= 1000
        slope = np.polyfit(np.log(n[k]), np.log(med[k]), 1)[0]
        L.append(f"| {model} | {inp} | " + " | ".join(f"{v:.3f}" for v in med) + f" | {slope:.2f} |")
        p95s.append(f"{label}: " + ", ".join(f"{v:.2f}" for v in p95))
    for a, key in zip(ax, ("median", "p95")):
        a.set_xlabel("number of training trajectories")
        a.set_ylabel(f"validation {key} relative L2")
        style(a)
        a.legend(frameon=False, fontsize=8)
    ax[0].set_title("median: falls ~4x from 500 to 7000, still falling")
    ax[1].set_title("p95: falls far more slowly (the spinning tail)")
    fig.suptitle("Learning curve: each model at its own lr 1e-3 cosine setting, one seed")
    save(fig, "e4_datasize.png")
    L += ["", "p95 at the same n: " + "; ".join(p95s) + ".", "",
          "The median falls roughly as n^-0.5 with no sign of saturation at 7000, so more data would still help the typical "
          "trajectory. The p95 falls only from about 0.77-0.82 to 0.64-0.68 over a 14x increase in data: the spinning tail "
          "improves much more slowly. The ranking (MLPs ahead of DeepONets) holds at every n >= 1000; at n = 500 all four "
          "are close. One seed per point.", ""]


def fig_psweep(runs, L):
    sel = [r for r in runs if r["model"] == "deeponet" and r["input"] == "coef" and r["lr"] == "0.003"
           and r["schedule"] == "cosine" and int(r["n_fourier"]) == 0 and r["seed"] == "3001" and int(r["n_train"]) == 7000]
    p_of = lambda r: int(r["run"].split("_p")[1].split("_")[0])
    eq = sorted([r for r in sel if abs(int(r["params"]) - 171396) < 1500], key=p_of)
    fw = sorted([r for r in sel if int(r["width"]) == 128], key=p_of)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    L += ["## DeepONet basis size p (compact input, lr 3e-3 cosine, seed 3001, validation)", "",
          "| p | equal budget: width / params / val loss / median / best epoch | fixed width 128: params / val loss / median / best epoch |",
          "|---|---|---|"]
    for rows, name, c, m in ((eq, "equal budget (~171k params, width adjusted)", "#D55E00", "o"),
                             (fw, "fixed width 128 (params grow with p)", "#0072B2", "s")):
        p = [p_of(r) for r in rows]
        ax[0].semilogx(p, [float(r["val_loss"]) for r in rows], marker=m, color=c, lw=2, label=name, base=2)
        ax[1].semilogx(p, [float(r["median"]) for r in rows], marker=m, color=c, lw=2, label=name, base=2)
    for p in sorted({p_of(r) for r in eq + fw}):
        a = next((r for r in eq if p_of(r) == p), None)
        b = next((r for r in fw if p_of(r) == p), None)
        fa = f"{a['width']} / {int(a['params']):,} / {float(a['val_loss']):.4f} / {float(a['median']):.4f} / {a['best_epoch']}" if a else ""
        fb = f"{int(b['params']):,} / {float(b['val_loss']):.4f} / {float(b['median']):.4f} / {b['best_epoch']}" if b else ""
        L.append(f"| {p} | {fa} | {fb} |")
    for a, t in zip(ax, ("validation loss (selection metric)", "validation median")):
        a.set_xlabel("p (basis functions per output channel)")
        a.set_title(t)
        style(a)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Is p = 64 a sensible choice?  One seed per point: differences under ~0.005 are within seed noise")
    save(fig, "e4_psweep.png")
    L += ["", "Seed noise on val loss for this model is about 0.003 (3 seeds at p = 64), so only p = 8 (and p = 256 at equal "
          "budget, where width falls to 71) are clearly worse; p from 16 to 128 is a plateau. p = 32 has the lowest val loss "
          "(0.199 vs 0.206 at p = 64, about 2 seed std, one seed): suggestive, not established. p = 8 peaks early (best "
          "epoch 49 to 66): too few basis functions to represent four channels' time behaviour.", ""]


# ------------------------------------------------------------------------------------- figures from per-sample arrays
def fig_consistency(ps, d, L):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    L += ["## Physical consistency on test (seed 3001)", "",
          "`cons` = rms(dq/dt - qdot) / rms(true qdot), dq/dt by autodiff of the q outputs. `energy` = excess "
          "energy-balance residual over the RK4/quadrature floor, as a fraction of the trajectory's energy swing.", "",
          "| model | cons median | cons p95 | energy median | energy p95 | headline median |", "|---|---|---|---|---|---|"]
    for tag, (label, _, c, m) in ALL.items():
        ls = "--" if tag in CONS else "-"
        for a, key in ((ax[0], "cons"), (ax[2], "energy")):
            e = np.sort(ps[f"{tag}|test|{key}"])
            a.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=c, ls=ls, lw=2,
                   label=f"{label} (median {np.median(e):.3f})")
        cons, en, err = ps[f"{tag}|test|cons"], ps[f"{tag}|test|energy"], ps[f"{tag}|test|err"]
        L.append(f"| {label} | {np.median(cons):.3f} | {np.percentile(cons, 95):.3f} | {np.median(en):.3f} | "
                 f"{np.percentile(en, 95):.3f} | {np.median(err):.4f} |")
    for a, t in ((ax[0], "rms(dq/dt - qdot) / rms(true qdot)"), (ax[2], "excess energy-balance residual / energy swing")):
        a.set_xlabel(t)
        a.set_ylabel("fraction of test samples below x")
        style(a, "x")
        a.legend(frameon=False, fontsize=7, loc="lower right")
    ax[0].set_title("(a) are the q and qdot outputs consistent?")
    ax[2].set_title("(c) does the prediction obey dE/dt = tau.qdot - b|qdot|^2?")

    # (b) one sample: the median-consistency sample of the base MLP (samples), base vs ff6 + cons
    i = int(np.argsort(ps["mlp/samples|test|cons"])[len(d["test_traj"]) // 2])
    true = d["test_traj"][i]
    for tag, ls in (("mlp/samples", "-"), ("mlp/samples/ff6cons", "--")):
        model, in_norm, out_norm, h = load_checkpoint(ckpt(ALL[tag][1]))
        u, t, _ = make_split({k: v[i:i + 1] if k.startswith("test_") else v for k, v in d.items()}, "test",
                             in_norm, out_norm, T_FINAL, h.get("input_mode", "samples"))
        pred = predict(model, u, t, out_norm)[0]
        res = qdot_consistency(model, u, t, out_norm)[0]
        ax[1].plot(T_GRID, pred[:, 3], color=ALL[tag][2] if ls == "-" else "#0072B2", ls=ls, lw=1.8,
                   label=f"{ALL[tag][0]}: q2dot output")
        ax[1].plot(T_GRID, pred[:, 3] + res[:, 1], color="#CC79A7" if ls == "-" else "#009E73", ls=ls, lw=1.2,
                   label=f"{ALL[tag][0]}: d q2/dt (autodiff)")
    ax[1].plot(T_GRID, true[:, 3], color="#222222", lw=2.5, alpha=0.5, label="truth q2dot")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("rad/s")
    ax[1].set_title(f"(b) test #{i}: the two velocity estimates of one network")
    style(ax[1])
    ax[1].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Physical consistency is not in the data loss: DeepONets are more self-consistent; a penalty term "
                 "(dashed) closes the gap for both")
    save(fig, "final_consistency.png")
    L.append("")


def fig_ood(ps, d, fe, L):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    # (a) error vs peak torque, across the edge of the training range
    peak = np.concatenate([np.abs(d[f"{s}_torque"]).max(axis=(1, 2)) for s in ("test", "ood_amp1", "ood_amp2")])
    edges = np.arange(0, 13, 1.0)
    cen = 0.5 * (edges[1:] + edges[:-1])
    L += ["## Out-of-range degradation (seed 3001)", "", "(a) Median error by peak |torque| (max over joints and time); "
          "training range is 0 to 6 N m.", "",
          "| model | " + " | ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])) + " |",
          "|---|" + "---|" * len(cen)]
    for tag, (label, _, c, m) in MODELS.items():
        e = np.concatenate([ps[f"{tag}|{s}|err"] for s in ("test", "ood_amp1", "ood_amp2")])
        idx = np.digitize(peak, edges) - 1
        med = np.array([np.median(e[idx == k]) if (idx == k).sum() >= 5 else np.nan for k in range(len(cen))])
        p95 = np.array([np.percentile(e[idx == k], 95) if (idx == k).sum() >= 5 else np.nan for k in range(len(cen))])
        ax[0].plot(cen, med, marker=m, color=c, lw=2, label=f"{label}: median")
        ax[0].plot(cen, p95, marker=m, color=c, lw=1, ls=":", alpha=0.7)
        L.append(f"| {label} | " + " | ".join("" if np.isnan(v) else f"{v:.3f}" for v in med) + " |")
    counts = np.histogram(peak, edges)[0]
    L += ["| samples per bin | " + " | ".join(str(n) for n in counts) + " |", "",
          "Reading: there is no cliff at the 6 N m edge. Inside the training range the median error already grows about "
          "20x from the weakest to the strongest torques (stronger torque = more violent motion, see the regime table); "
          "it roughly doubles across the edge (5-6 to 6-7 N m) and then saturates at 0.6 to 0.8, where the prediction "
          "carries little information about the true trajectory (the mean-trajectory baseline scores 0.43 on test). "
          "The bins just past the edge hold only 30 and 56 samples.", ""]
    ax[0].axvspan(0, 6, color="#009E73", alpha=0.08)
    ax[0].text(3, 0.012, "training range", ha="center", fontsize=8, color="#006b4f")
    ax[0].set_xlabel("peak |tau| over both joints [N m]  (test, then ood_amp1, ood_amp2)")
    ax[0].set_ylabel("per-sample relative L2  (dotted: p95)")
    ax[0].set_title("(a) amplitude: error rises inside the range,\n~doubles past the edge, then saturates", fontsize=10)
    style(ax[0], "y")
    ax[0].legend(frameon=False, fontsize=7)

    # (b) m_p / b out of range vs a magnitude-matched test subset
    pk_test = np.abs(d["test_traj"][..., :2]).max(axis=(1, 2))
    pk_ood = np.abs(d["ood_traj"][..., :2]).max(axis=(1, 2))
    match = np.array([np.argmin(np.abs(np.log(pk_test) - np.log(x))) for x in pk_ood])
    tag = "mlp/coef"
    ax[1].scatter(pk_test, ps[f"{tag}|test|err"], s=6, color="#999999", alpha=0.4, lw=0, label="test (in range)")
    ax[1].scatter(pk_ood, ps[f"{tag}|ood|err"], s=12, color="#D55E00", alpha=0.8, lw=0, label="ood (m_p 1-1.5, b 0.5-0.8)")
    ax[1].set_xlabel("peak |q| of the true trajectory [rad]")
    ax[1].set_ylabel("per-sample relative L2")
    ax[1].set_title(f"(b) {MODELS[tag][0]}: OOD m_p, b errors match\nin-range errors of the same size", fontsize=10)
    style(ax[1], "x")
    ax[1].set_yscale("log")
    ax[1].legend(frameon=False, fontsize=8)
    L += ["(b) The `ood` set (m_p, b beyond range) against a test subset matched on trajectory size (each OOD sample "
          "paired with the test sample of nearest peak |q|):", "",
          f"Peak |q| median: test {np.median(pk_test):.2f} rad, ood {np.median(pk_ood):.2f} rad, matched test subset "
          f"{np.median(pk_test[match]):.2f} rad.", "",
          "| model | test median (all) | matched test subset median | ood median | ood / matched |", "|---|---|---|---|---|"]
    for t2, (label, _, _, _) in MODELS.items():
        et, eo = ps[f"{t2}|test|err"], ps[f"{t2}|ood|err"]
        L.append(f"| {label} | {np.median(et):.3f} | {np.median(et[match]):.3f} | {np.median(eo):.3f} | "
                 f"{np.median(eo) / np.median(et[match]):.2f} |")
    L += ["", "Reading: the `ood` set has no tail of violent trajectories (its p95 is about 0.13 against 0.60 on test), "
          "and once compared with in-range trajectories of the same size its error is within about +-25 % of theirs. "
          "Heavier payload and more friction are a small, smooth shift that the models absorb; this set does not test "
          "extrapolation in any demanding sense.", ""]

    # (c) rougher torque: coef vs samples input, seed mean +- std
    x = np.arange(2)
    for k, fam in enumerate(("mlp", "deeponet")):
        for j, inp in enumerate(("coef", "samples")):
            tagk = f"{fam}/{inp}"
            mu_s, sd_s, _ = agg(fe, "median", fam, inp, "ood_smooth")
            mu_t, _, _ = agg(fe, "median", fam, inp, "test")
            pos = k + (j - 0.5) * 0.36
            ax[2].bar(pos, mu_s, 0.34, yerr=sd_s, color=MODELS[tagk][2], capsize=4, edgecolor="#333333", lw=0.5,
                      label=f"{inp} input" if k == 0 else None, hatch="" if inp == "samples" else "//")
            ax[2].plot([pos - 0.17, pos + 0.17], [mu_t, mu_t], color="#222222", lw=2, ls=":")
    ax[2].plot([], [], color="#222222", ls=":", label="same model on test")
    ax[2].set_xticks(x, ["MLP", "DeepONet"])
    ax[2].set_ylabel("ood_smooth median relative L2 (3 seeds)")
    ax[2].set_title("(c) rougher torque (13 modes):\nsampled input generalises better", fontsize=10)
    style(ax[2])
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Out of range is a degradation curve, not a pass / fail")
    save(fig, "final_ood_degradation.png")


def fig_sensitivity(ps, d, L, n=500, eps=(1e-3, 1e-2)):
    """How much does the TRUE trajectory move when every input moves by a small relative amount?"""
    rng = np.random.default_rng(SEED)
    params, coef, true = d["test_params"][:n], d["test_tau_coef"][:n], d["test_traj"][:n].astype(np.float64)
    std = ps["std"]
    peak = np.abs(true[..., :2]).max(axis=(1, 2))
    sens = {}
    for e in eps:
        pp = params * (1 + e * rng.standard_normal(params.shape))
        cc = coef * (1 + e * rng.standard_normal(coef.shape))
        sens[e] = rel_l2_per_sample(bm.rk4_batched(pp, cc, T_GRID, X0), true, std)
    err = ps["mlp/coef|test|err"][:n]
    edges = [0, np.pi, 2 * np.pi, 4 * np.pi, np.inf]
    names = ["< pi", "pi to 2 pi", "2 pi to 4 pi", "> 4 pi"]
    L += [f"## Input sensitivity of the TRUE map (first {n} test samples)", "",
          "Every input (m_p, b and each Fourier coefficient) multiplied by (1 + eps z), z ~ N(0, 1), re-simulated with "
          "vectorised RK4; the change is measured with the headline metric. No model involved. Compared with the error "
          "of the best surrogate (MLP, coef, seed 3001) on the same samples, split by how violent the motion is.", "",
          "| peak \\|q\\| | n | median change, eps = 1e-3 | median change, eps = 1e-2 | p90 change, eps = 1e-2 | "
          "surrogate median error | surrogate p95 |", "|---|---|---|---|---|---|---|"]
    for nm, lo, hi in zip(names, edges[:-1], edges[1:]):
        k = (peak >= lo) & (peak < hi)
        if k.sum() < 5:
            continue
        L.append(f"| {nm} | {k.sum()} | {np.median(sens[1e-3][k]):.1e} | {np.median(sens[1e-2][k]):.1e} | "
                 f"{np.percentile(sens[1e-2][k], 90):.1e} | {np.median(err[k]):.3f} | {np.percentile(err[k], 95):.3f} |")
    rho = spearmanr(sens[1e-2], err)[0]
    L += ["", f"Spearman correlation between the 1 % sensitivity and the surrogate's error: {rho:+.2f}. The "
          "trajectories a 1 % input change moves most are the ones this surrogate gets most wrong (and the regime table "
          "shows the same ordering for all four models): their error reflects "
          "how hard the map is there, not only how the network was built. (Correlation, not proof.)", ""]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sens[1e-2], err, c=np.log10(peak), s=10, cmap="viridis", lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("change of the TRUE trajectory when all inputs move by ~1 % (relative L2)")
    ax.set_ylabel("MLP (coef) surrogate error (relative L2)")
    ax.set_title(f"Sensitive samples are the hard ones\n(Spearman {rho:+.2f}, first {n} test samples)")
    fig.colorbar(sc, label="log10 peak |q| [rad]")
    style(ax)
    save(fig, "final_sensitivity.png")


def tables_test(ps, fe, d, L):
    L += ["## Test: per-channel median relative L2 (each channel on its own scale; mean over 3 seeds)", "",
          "| model | " + " | ".join(CH) + " |", "|---|---|---|---|---|"]
    for tag, (label, _, _, _) in MODELS.items():
        model, inp = tag.split("/")
        L.append(f"| {label} | " + " | ".join(f"{agg(fe, c, model, inp, 'test')[0]:.3f}" for c in CH) + " |")
    L += ["", "q2 and the velocities are the hard channels; q1 is learnt well by every model.", "",
          "## Test: error by regime (seed 3001; bins by peak |q| of the true trajectory)", ""]
    peak = np.abs(d["test_traj"][..., :2]).max(axis=(1, 2))
    edges = [0, np.pi, 2 * np.pi, 4 * np.pi, np.inf]
    names = ["< pi", "pi to 2 pi", "2 pi to 4 pi", "> 4 pi"]
    bins = [(peak >= lo) & (peak < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    L += ["| model | " + " | ".join(f"{nm} (n = {b.sum()}): median / p95" for nm, b in zip(names, bins)) + " |",
          "|---|" + "---|" * len(bins)]
    for tag, (label, _, _, _) in ALL.items():
        e = ps[f"{tag}|test|err"]
        L.append(f"| {label} | " + " | ".join(f"{np.median(e[b]):.3f} / {np.percentile(e[b], 95):.3f}" for b in bins) + " |")
    L += ["", "## Test: error vs time (seed 3001; |error(t)| / rms of the whole true trajectory)", "",
          "| model | " + " | ".join(f"t = {t:g} s: median / p90" for t in (0.5, 1, 2, 3)) + " |", "|---|---|---|---|---|"]
    it = [int(np.argmin(np.abs(T_GRID - t))) for t in (0.5, 1, 2, 3)]
    for tag, (label, _, _, _) in ALL.items():
        e = ps[f"{tag}|test|err_time"]
        L.append(f"| {label} | " + " | ".join(f"{np.median(e[:, i]):.3f} / {np.percentile(e[:, i], 90):.3f}" for i in it) + " |")
    L += ["", "## Test: initial condition (W5). Every true trajectory starts at x0 = 0; nothing forces the prediction to.", "",
          "| model | median \\|x_pred(0)\\| q1, q2, q1dot, q2dot (rad, rad/s) | as % of channel rms |", "|---|---|---|"]
    for tag, (label, _, _, _) in ALL.items():
        ic = np.median(ps[f"{tag}|test|ic"], axis=0)
        L.append(f"| {label} | " + ", ".join(f"{v:.3f}" for v in ic) + " | "
                 + ", ".join(f"{100 * v / r:.1f}" for v, r in zip(ic, ps["channel_rms"])) + " |")
    L.append("")


def tables_e4_test(L):
    """Test / OOD of the E4 variants next to the base models (from the CSVs), clearly labelled post hoc."""
    files = [("rule-selected", RES / "final_eval.csv"), ("ff6", RES / "final_eval_ff6.csv"),
             ("ff6 + cons 0.1", RES / "final_eval_ff6cons.csv")]
    rows = {n: read_csv(p) for n, p in files if p.exists()}
    L += ["## E4 variants on test and OOD (evaluated AFTER the selection rule fixed the headline models)", "",
          "These are follow-up variants, not a re-selection: the headline remains the rule-selected row. Mean +- std over 3 seeds.", "",
          "| model | input | variant | test median | test p95 | test consistency | ood_amp1 median | ood_smooth median |",
          "|---|---|---|---|---|---|---|---|"]
    for tag in MODELS:
        model, inp = tag.split("/")
        for name, rr in rows.items():
            if not any(r["model"] == model and r["input"] == inp for r in rr):
                continue
            def ms(k, s):
                mu, sd, _ = agg(rr, k, model, inp, s)
                return f"{mu:.3f} +- {sd:.3f}"
            L.append(f"| {model} | {inp} | {name} | {ms('median', 'test')} | {ms('p95', 'test')} | "
                     f"{ms('consistency', 'test')} | {ms('median', 'ood_amp1')} | {ms('median', 'ood_smooth')} |")
    L.append("")
    return rows


# ------------------------------------------------------------------------------------- benchmark figures
def fig_bench(L):
    path = RES / "benchmark.json"
    if not path.exists():
        print("no benchmark.json; skipping speed figures")
        return
    B = json.loads(path.read_text(encoding="utf-8"))
    meta, rows, ivp = B["meta"], B["rows"], B["ivp"]
    colours = {f"{r['model']}/{r['input']}": MODELS[f"{r['model']}/{r['input']}"] for r in rows}

    # pareto: accuracy vs cost
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ok = {k: v for k, v in ivp.items() if np.isfinite(v["median_err"])}
    ms = [v["ms"] for v in ok.values()]
    ax.plot(ms, [v["median_err"] for v in ok.values()], "o-", color="#555555", lw=1.5, label="solve_ivp RK45: median")
    ax.plot(ms, [v["p95_err"] if np.isfinite(v["p95_err"]) else np.nan for v in ok.values()], "o:", color="#555555",
            mfc="none", lw=1, label="solve_ivp RK45: p95")
    for k, v in ok.items():   # label every other tolerance: the loose ones sit within ~1 ms of each other
        if float(k) in (0.1, 0.06, 0.03, 0.01, 1e-3, 1e-4, 1e-6):
            ax.annotate(f"rtol {k}", (v["ms"], v["median_err"]), fontsize=6.5, xytext=(7, -3), textcoords="offset points")
    for r in rows:
        label, _, c, m = colours[f"{r['model']}/{r['input']}"]
        ax.plot(r["ms_b64_best"], r["median_err"], marker=m, color=c, ms=9, ls="none", label=f"{label}: median")
        ax.plot(r["ms_b64_best"], r["p95_err"], marker=m, color=c, ms=9, mfc="none", ls="none")
        ax.vlines(r["ms_b64_best"], r["median_err"], r["p95_err"], color=c, lw=1, alpha=0.6)
    ax.axvline(meta["rk4_ms"], color="#009E73", lw=1.5, ls="--", label=f"vectorised RK4 = the truth, batch 64 ({meta['rk4_ms']:.1f} ms)")
    big = max(int(b) for b in B["rk4_sweep"])
    ax.axvline(B["rk4_sweep"][str(big)] * 1e3, color="#009E73", lw=1.5, ls=":",
               label=f"vectorised RK4, batch {big} ({B['rk4_sweep'][str(big)] * 1e3:.2f} ms)")
    for key, sw in B["sweeps"].items():
        if "cached" in sw:
            label, _, c, m = MODELS[key]
            r = next(r for r in rows if f"{r['model']}/{r['input']}" == key)
            ax.plot(sw["cached"][str(big)] * 1e3, r["median_err"], marker=m, color=c, ms=7, ls="none", alpha=0.5)
    ax.text(0.02, 0.97, f"faded markers: DeepONet, trunk cached, batch {big}", transform=ax.transAxes, fontsize=7, va="top")
    style(ax, "x")
    ax.set_yscale("log")
    ax.set_xlabel("milliseconds per trajectory (surrogates: batch 64, fastest variant; solve_ivp: one at a time)")
    ax.set_ylabel("relative L2 vs stored truth (filled: median, hollow: p95)")
    ax.set_title("Accuracy vs cost on 64 validation trajectories")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "bench_pareto.png")

    # batch sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    Bs = sorted(int(b) for b in B["rk4_sweep"])
    ax.loglog(Bs, [B["rk4_sweep"][str(b)] * 1e3 for b in Bs], "o-", color="#009E73", lw=2.5, label="vectorised NumPy RK4")
    for key, sw in B["sweeps"].items():
        label, _, c, m = MODELS[key]
        for name, s in sw.items():
            ax.loglog(Bs, [s[str(b)] * 1e3 for b in Bs], marker=m, color=c, lw=1.8, ls="--" if name == "cached" else "-",
                      label=label + (" (trunk cached)" if name == "cached" else ""))
    ax.set_xlabel("batch size B (trajectories per call)")
    ax.set_ylabel("ms per trajectory")
    ax.set_title("Per-trajectory cost vs batch size: the speed-up shrinks as B grows")
    style(ax)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "bench_batch.png")

    # break-even
    fig, ax = plt.subplots(figsize=(8, 5))
    N = np.logspace(2, 8, 200)
    best_ivp = None
    for r in rows:
        label, _, c, m = colours[f"{r['model']}/{r['input']}"]
        fixed = meta["t_gen_vec_s"] + r["train_s_idle"]
        ax.loglog(N, fixed + N * r["ms_b64_best"] / 1e3, color=c, lw=2, label=f"{label}: data + 1 training run + queries")
        ax.loglog(N, meta["t_gen_vec_s"] + 5 * r["train_s_idle"] + N * r["ms_b64_best"] / 1e3, color=c, lw=1, ls=":")
        ax.plot(r["N_ivp_vec_one"], fixed + r["N_ivp_vec_one"] * r["ms_b64_best"] / 1e3, marker=m, color=c, ms=8)
        best_ivp = r["matched_ms"] if best_ivp is None else best_ivp
    rts = sorted({r["matched_rtol"] for r in rows})
    msi = min(ivp[rt]["ms"] for rt in rts)
    ax.loglog(N, N * msi / 1e3, color="#555555", lw=2, ls="--",
              label=f"solve_ivp at matched error (rtol {', '.join(rts)}, {msi:.1f} ms)")
    ax.loglog(N, N * meta["rk4_ms"] / 1e3, color="#009E73", lw=1.5, ls="-.",
              label=f"vectorised RK4, batch 64 ({meta['rk4_ms']:.1f} ms, exact labels)")
    big = max(int(b) for b in B["rk4_sweep"])
    rk4_big = B["rk4_sweep"][str(big)] * 1e3
    ax.loglog(N, N * rk4_big / 1e3, color="#009E73", lw=1.5, ls=":",
              label=f"vectorised RK4, batch {big} ({rk4_big:.2f} ms, exact labels)")
    # the strongest classical rival: exact RK4, vectorised over the same large batch as the surrogate
    L += ["## Speed: the fair rival and the break-even", "",
          f"Vectorised RK4 (exact labels) costs {meta['rk4_ms']:.1f} ms per trajectory at batch 64 and {rk4_big:.2f} ms at "
          f"batch {big}, while `solve_ivp` at the surrogates' error level costs {msi:.1f} ms (one trajectory at a time) and "
          "fails outright on 1 to 2 of 64 trajectories at rtol 0.08 to 0.1. So for batched queries the strongest classical "
          "rival is batched RK4, which is also exact. Break-even N* = (data + training) / (rival - surrogate), training measured "
          "idle, data generated with vectorised RK4:", "",
          f"| model | surrogate ms / traj at batch {big} (variant) | N* vs matched solve_ivp | N* vs RK4 at batch {big}, 1 training run | "
          f"same, incl. the 5-run lr grid |", "|---|---|---|---|---|"]
    n1s, n5s, sur_big = [], [], {}
    for r in rows:
        key = f"{r['model']}/{r['input']}"
        variant = min(B["sweeps"][key], key=lambda v: B["sweeps"][key][v][str(big)])
        sur = B["sweeps"][key][variant][str(big)] * 1e3
        n1 = bm.break_even_queries(meta["t_gen_vec_s"], r["train_s_idle"], rk4_big / 1e3, sur / 1e3)
        n5 = bm.break_even_queries(meta["t_gen_vec_s"], 5 * r["train_s_idle"], rk4_big / 1e3, sur / 1e3)
        L.append(f"| {colours[key][0]} | {sur:.3f} ({'trunk cached' if variant == 'cached' else 'full'}) | "
                 f"{r['N_ivp_vec_one']:,.0f} | {n1:,.0f} | {n5:,.0f} |")
        n1s.append(n1); n5s.append(n5); sur_big[key] = sur
    ivp_ns = [r["N_ivp_vec_one"] for r in rows]
    mlp64 = min(r["ms_b64_best"] for r in rows if r["model"] == "mlp")
    don64 = min(r["ms_b64_best"] for r in rows if r["model"] == "deeponet")
    mlp_big = min(v for k, v in sur_big.items() if k.startswith("mlp"))
    don_big = min(v for k, v in sur_big.items() if k.startswith("deeponet"))
    L += ["", f"Reading: against `solve_ivp` the surrogates pay off after roughly {min(ivp_ns):,.0f} to {max(ivp_ns):,.0f} "
          f"queries; against an exact, batched RK4 after roughly {min(n1s) / 1e6:.1f} to {max(n1s) / 1e6:.1f} million "
          f"({min(n5s) / 1e6:.1f} to {max(n5s) / 1e6:.1f} million with the tuning grid), because RK4 itself becomes "
          "cheap in large batches. Training dominates the fixed cost (data generation with vectorised RK4 takes about "
          f"{meta['t_gen_vec_s']:.0f} s). The DeepONet's cached trunk makes it the cheapest per query (about "
          f"{mlp64 / don64:.0f}x below the MLPs at batch 64 with each model in its fastest thread setting, about "
          f"{mlp_big / don_big:.0f}x at batch {big} with default threads), a real "
          "architectural advantage that the MLP cannot copy; it does not make it more accurate.", ""]
    ax.set_xlabel("number of trajectories queried")
    ax.set_ylabel("total wall-clock seconds")
    ax.set_title("Break-even (markers): surrogate cost includes data generation and training\n(dotted: including the 5-run lr grid)")
    style(ax)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "bench_breakeven.png")


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    d = load(DATA_PATH_SMOOTH)
    ps = None
    if CACHE.exists() and not args.no_cache:   # keys are stored with '/' -> '__' (a '/' would be a zip folder)
        ps = {k.replace("__", "/"): v for k, v in np.load(CACHE).items()}
        if not all(f"{t}|test|err" in ps for t in ALL):
            ps = None
    if ps is None:
        ps = compute_persample(d)
        np.savez_compressed(CACHE, **{k.replace("/", "__"): v for k, v in ps.items()})

    fe = read_csv(RES / "final_eval.csv")
    runs, agg_rows = read_csv(RES / "runs.csv"), read_csv(RES / "runs_agg.csv")
    L = ["# Final analysis (after the selection rule)", "",
         "Generated by `scripts/final_analysis.py`. Headline metric everywhere: per-sample relative L2 over (time, channel), "
         "channels scaled by the train std. Test n = 1500, each OOD set n = 200. Per-sample numbers use the seed-3001 "
         "checkpoints; numbers marked '3 seeds' come from `results/final_eval*.csv`.", ""]
    tables_test(ps, fe, d, L)
    naive_baselines(d, L)
    fig_overview(fe, L)
    fig_consistency(ps, d, L)
    fig_ood(ps, d, fe, L)
    fig_sensitivity(ps, d, L)
    variants = tables_e4_test(L)
    fig_seeds(variants, L)
    fig_e4(agg_rows, runs, L)
    fig_datasize(runs, L)
    fig_psweep(runs, L)
    fig_bench(L)
    (RES / "final_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
