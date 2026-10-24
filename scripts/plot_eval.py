"""Evaluation figures from saved checkpoints. Plotting only:
every number comes from the functions in src/evaluate.py.

    py -3.10 -m scripts.plot_eval checkpoints/a.pt                         # one model, val split
    py -3.10 -m scripts.plot_eval checkpoints/a.pt checkpoints/b.pt ...    # overlay several models
    py -3.10 -m scripts.plot_eval checkpoints/*.pt --split test --final    # final evaluation ONLY

The default split is ``val``. Any other split (test, ood, ood_amp1, ood_amp2, ood_smooth) is refused
unless ``--final`` is given, so a sweep cannot peek at the test set by accident (EXPERIMENTS.md,
selection rule). Figures go to results/evalfig_<split>_<tag>_*.png:

  cases   (#10) best / median / worst sample of the FIRST checkpoint, all four channels, truth vs prediction
  time    (#11) error vs time: median line (10-90 % band for the first checkpoint), all channels together and each on its own
  cdf     (#12, #17) cumulative fraction of samples below each error, one line per checkpoint
  regime  (#13) error vs how violent the true trajectory is (peak |q|), binned median and p95
  inputs  (#13) error of the FIRST checkpoint vs b, m_p and peak torque, with Spearman rho

Error definitions (see src/evaluate.py): per-sample relative L2 with channels scaled by the TRAIN std
(the headline metric); error vs time divides by the sample's whole-trajectory rms (x0 = 0, so a
pointwise relative error is undefined at t = 0).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import DATA_PATH_SMOOTH, T_FINAL, load
from src.evaluate import CHANNELS, error_vs_time, predict, rel_l2_per_sample
from src.train import load_checkpoint, make_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
# Okabe-Ito (colour-blind safe), first two match the train/val colours in plot_training; each series also
# gets its own line style so identity never depends on colour alone.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
TRUE_C = "#222222"
TOLS = (0.01, 0.05, 0.10)
UNITS = {"q1": "rad", "q2": "rad", "q1dot": "rad/s", "q2dot": "rad/s"}


def label_for(path: Path, model, hist) -> str:
    """Short legend label, e.g. 'DeepONet, samples, lr 0.001, cosine'."""
    kind = "DeepONet" if type(model).__name__ == "DeepONet" else "MLP"
    stem = path.stem
    lr = stem.split("_lr")[1].split("_")[0] if "_lr" in stem else "?"
    sched = ", cosine" if hist.get("schedule") == "cosine" or stem.endswith("_cosine") else ""
    return f"{kind}, {hist.get('input_mode', 'samples')}, lr {lr}{sched}"


def evaluate_one(path: Path, d: dict, split: str) -> dict:
    model, in_norm, out_norm, hist = load_checkpoint(path)
    u, t, _ = make_split(d, split, in_norm, out_norm, T_FINAL, hist.get("input_mode", "samples"))
    pred = predict(model, u, t, out_norm)
    true = d[f"{split}_traj"].astype(np.float64)
    std = out_norm.std.numpy().astype(np.float64)
    return {
        "label": label_for(path, model, hist), "pred": pred, "std": std,
        "err": rel_l2_per_sample(pred, true, std),
        "err_time": error_vs_time(pred, true, std),
        # each channel on its own: pass only that channel (its scale then cancels in the ratio)
        "err_time_ch": [error_vs_time(pred[..., [j]], true[..., [j]]) for j in range(4)],
    }


def fig_cases(r: dict, true: np.ndarray, t: np.ndarray, out: Path, split: str) -> None:
    """#10: best, median and worst sample (by the headline error) of one model."""
    order = np.argsort(r["err"])
    picks = [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharex=True)
    for row, (name, i) in enumerate(picks):
        for j, ch in enumerate(CHANNELS):
            ax = axes[row, j]
            ax.plot(t, true[i, :, j], color=TRUE_C, lw=2, label="truth (RK4)")
            ax.plot(t, r["pred"][i, :, j], color=COLOURS[0], lw=2, ls="--", label="prediction")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"{ch} [{UNITS[ch]}]")
            if j == 0:
                ax.set_ylabel(f"{name}\nsample {i}, error {r['err'][i]:.3f}")
            if row == 2:
                ax.set_xlabel("t [s]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{r['label']}: best / median / worst {split} sample (relative L2, std-scaled channels)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_time(results: list, t: np.ndarray, out: Path, split: str) -> None:
    """#11: error vs time, median and 10-90 % band; all channels, then each channel on its own."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharex=True)
    panels = [("all channels (std-scaled)", lambda r: r["err_time"])] + [
        (f"{ch} only", lambda r, j=j: r["err_time_ch"][j]) for j, ch in enumerate(CHANNELS)]
    for ax, (title, get) in zip(axes, panels):
        for k, r in enumerate(results):
            e = get(r)
            if k == 0:   # band for the first model only; overlapping bands are unreadable
                ax.fill_between(t, np.percentile(e, 10, axis=0), np.percentile(e, 90, axis=0),
                                color=COLOURS[k], alpha=0.15, lw=0)
            ax.plot(t, np.median(e, axis=0), color=COLOURS[k], ls=STYLES[k], lw=2, label=r["label"])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("|error(t)| / rms of true trajectory")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Error vs time on {split} (n = {len(results[0]['err'])}): median per model; band = 10th to 90th percentile of the first model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cdf(results: list, out: Path, split: str) -> None:
    """#12 / #17: empirical CDF of the per-sample headline error, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in enumerate(results):
        e = np.sort(r["err"])
        ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=COLOURS[k], ls=STYLES[k],
                lw=2, label=f"{r['label']} (median {np.median(e):.3f})")
    for tol in TOLS:
        ax.axvline(tol, color="#888888", lw=1, ls=":")
        ax.text(tol, 1.01, f"{tol:.0%}", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-sample relative L2 error (channels scaled by train std)")
    ax.set_ylabel("fraction of samples with error below x")
    ax.set_title(f"Error distribution on {split} (n = {len(results[0]['err'])})", pad=16)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_regime(results: list, true: np.ndarray, out: Path, split: str) -> None:
    """#13: error against the peak |q| of the TRUE trajectory (calm swing ... spinning arm)."""
    peak = np.abs(true[..., :2]).max(axis=(1, 2))
    edges = np.array([0, np.pi, 2 * np.pi, 4 * np.pi, np.inf])
    bins = [(peak >= lo) & (peak < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    bins = [b for b in bins if b.sum() >= 5]                      # drop (near-)empty bins
    centres = [np.median(peak[b]) for b in bins]                  # marker at the bin's median peak
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(peak, results[0]["err"], s=8, color=COLOURS[0], alpha=0.25, lw=0,
               label=f"{results[0]['label']}: one dot per sample")
    for k, r in enumerate(results):
        med = [np.median(r["err"][b]) for b in bins]
        p95 = [np.percentile(r["err"][b], 95) for b in bins]
        ax.plot(centres, med, color=COLOURS[k], ls=STYLES[k], lw=2, marker="o", ms=6,
                label=f"{r['label']}: bin median (o) and p95 (^)")
        ax.plot(centres, p95, color=COLOURS[k], ls=STYLES[k], lw=1, marker="^", ms=6, alpha=0.7)
    names = ["< pi", "pi to 2 pi", "2 pi to 4 pi", "> 4 pi"]
    kept = [nm for nm, lo, hi in zip(names, edges[:-1], edges[1:]) if ((peak >= lo) & (peak < hi)).sum() >= 5]
    ax.text(0.99, 0.02, "samples per bin\n" + "\n".join(f"{nm}: {bb.sum()}" for nm, bb in zip(kept, bins)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak |q1|, |q2| of the true trajectory [rad]  (bin edges: pi, 2 pi, 4 pi)")
    ax.set_ylabel("per-sample relative L2 error")
    ax.set_title(f"Error vs trajectory violence on {split}", pad=16)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_inputs(r: dict, params: np.ndarray, torque: np.ndarray, out: Path, split: str) -> None:
    """#13 (inputs): error of ONE model against b, m_p and the peak torque, with Spearman rho."""
    from scipy.stats import spearmanr
    peak_tau = np.abs(torque).max(axis=(1, 2))           # max over joints and time, N m
    cols = [("friction b [N m s/rad]", params[:, 1]), ("payload m_p [kg]", params[:, 0]),
            ("peak |tau| over both joints [N m]", peak_tau)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, x) in zip(axes, cols):
        ax.scatter(x, r["err"], s=8, color=COLOURS[0], alpha=0.25, lw=0)
        edges = np.quantile(x, np.linspace(0, 1, 9))       # 8 equal-count bins
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 7)
        ax.plot([np.median(x[idx == k]) for k in range(8)], [np.median(r["err"][idx == k]) for k in range(8)],
                color=TRUE_C, lw=2, marker="o", ms=6, label="median of 8 equal-count bins")
        rho = spearmanr(x, r["err"])[0]
        ax.set_title(f"Spearman rho = {rho:+.2f}")
        ax.set_xlabel(name)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("per-sample relative L2 error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{r['label']}: error against the inputs on {split} (correlation, not cause)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--final", action="store_true", help="required for any split other than val")
    ap.add_argument("--tag", default=None, help="name used in the output files (default: first checkpoint)")
    args = ap.parse_args()
    if args.split != "val" and not args.final:
        ap.error(f"split {args.split!r} is for the final evaluation only; pass --final if this is it")
    if len(args.checkpoints) > len(COLOURS):
        ap.error(f"at most {len(COLOURS)} checkpoints per figure (use small multiples instead)")

    d = load(DATA_PATH_SMOOTH)
    true = d[f"{args.split}_traj"].astype(np.float64)
    t = np.asarray(d["t_grid"])
    results = [evaluate_one(p, d, args.split) for p in args.checkpoints]
    tag = args.tag or args.checkpoints[0].stem
    RESULTS.mkdir(exist_ok=True)
    base = RESULTS / f"evalfig_{args.split}_{tag}"
    fig_cases(results[0], true, t, Path(f"{base}_cases.png"), args.split)
    fig_time(results, t, Path(f"{base}_time.png"), args.split)
    fig_cdf(results, Path(f"{base}_cdf.png"), args.split)
    fig_regime(results, true, Path(f"{base}_regime.png"), args.split)
    fig_inputs(results[0], d[f"{args.split}_params"], d[f"{args.split}_torque"], Path(f"{base}_inputs.png"), args.split)
    for r in results:
        print(f"{r['label']:45s} median {np.median(r['err']):.4f}  p95 {np.percentile(r['err'], 95):.4f}")
    print("saved", base.name + "_{cases,time,cdf,regime,inputs}.png")


if __name__ == "__main__":
    main()
