"""Plot train and validation loss per epoch for each training run, one PNG per run.

    py -3.10 -m scripts.plot_training                  # every checkpoint in checkpoints/ and log in logs/
    py -3.10 -m scripts.plot_training logs/run.log checkpoints/x.pt   # chosen runs only

Sources:
  * checkpoints/*.pt  -> the full ``hist`` saved by ``fit`` (finished runs).
  * logs/*.log        -> lines "epoch N  train a  val b ..." printed by ``fit`` (runs still going).
If a log and a checkpoint describe the same run, the checkpoint wins (it is complete).
Output: results/training_curves/loss_<run>.png for each run, results/training_curves/loss_all_runs.png overlaying them,
and results/training_curves/loss_by_model.png (one panel per model and input; readable when there are many runs).
The loss is the channel-normalised relative L2 from ``relative_l2_loss`` (lower is better).
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CURVES = RESULTS / "training_curves"
TRAIN_C, VAL_C = "#0072B2", "#009E73"   # colours from docs/FIGURE_PLAN.md

LINE = re.compile(r"epoch\s+(\d+)\s+train\s+([\d.eE+-]+)\s+val\s+([\d.eE+-]+)")


def from_checkpoint(path: Path) -> dict:
    import torch
    h = torch.load(path, map_location="cpu", weights_only=True)["hist"]
    return {"train": list(h["train"]), "val": list(h["val"]), "complete": True}


def from_log(path: Path) -> dict:
    ep, tr, va = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            ep.append(int(m[1])); tr.append(float(m[2])); va.append(float(m[3]))
    if not ep:
        raise ValueError(f"no 'epoch ... train ... val ...' lines in {path}")
    return {"train": tr, "val": va, "epochs": ep, "complete": False}


def plot_run(name: str, run: dict, ax=None, solo=True):
    tr, va = np.asarray(run["train"]), np.asarray(run["val"])
    ep = np.asarray(run.get("epochs", np.arange(1, len(tr) + 1)))
    best = int(np.argmin(va))
    if solo:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, tr, color=TRAIN_C, label="train")
    ax.plot(ep, va, color=VAL_C, label="val")
    ax.plot(ep[best], va[best], "o", color=VAL_C, mfc="white", label=f"best val {va[best]:.4f} @ {ep[best]}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 loss (normalised)")
    state = "" if run["complete"] else f"  (in progress: {len(ep)} epochs logged)"
    ax.set_title(name + state, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    if solo:
        fig.tight_layout()
        out = CURVES / f"loss_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out.relative_to(ROOT))


LR_COLOURS = {"0.0003": "#0072B2", "0.001": "#E69F00", "0.003": "#009E73"}  # fixed order, Okabe-Ito


def plot_by_model(runs: dict) -> None:
    """Small multiples: one panel per (model, input); val loss only; colour = lr, style = schedule."""
    def family(name):
        model = "MLP" if name.startswith("mlp") else "DeepONet"
        return f"{model}, {'coef' if '_coef' in name else 'samples'} input"
    fams = sorted({family(n) for n in runs})
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.8), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for name, run in runs.items():
            if family(name) != fam:
                continue
            m = re.search(r"_lr([0-9.e-]+?)_", name)
            lr = m[1] if m else "?"
            cosine = name.endswith("_cosine")
            va = np.asarray(run["val"])
            ep = np.asarray(run.get("epochs", np.arange(1, len(va) + 1)))
            tag = "" if run["complete"] else " (running)"
            ax.plot(ep, va, color=LR_COLOURS.get(lr, "#888888"), ls="--" if cosine else "-", lw=1.5,
                    label=f"lr {lr}{', cosine' if cosine else ''}: best {va.min():.4f}{tag}")
        ax.set_title(fam, fontsize=10)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("validation loss (relative L2, normalised)")
    fig.suptitle("Validation loss by model (solid = constant lr, dashed = cosine)", fontsize=11)
    fig.tight_layout()
    out = CURVES / "loss_by_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out.relative_to(ROOT))


def main():
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted((ROOT / "checkpoints").glob("*.pt")) + sorted((ROOT / "logs").glob("*.log"))
    runs = {}
    for p in args:
        if p.suffix == ".pt":
            runs[p.stem] = from_checkpoint(p)
    for p in args:
        if p.suffix == ".log":
            # a log's header line names the run, e.g. "deeponet_p64_w128_ep200_lr0.003_...: N params"
            head = p.read_text(errors="replace").splitlines()[0]
            name = head.split(":")[0].strip() if ":" in head and " " not in head.split(":")[0] else p.stem
            if name not in runs:
                try:
                    runs[name] = from_log(p)
                except ValueError as e:   # e.g. sweep.log or a run that has not finished epoch 1
                    print("skipped:", e)
    if not runs:
        sys.exit("no runs found")
    CURVES.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        plot_run(name, run)
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        cmap = plt.get_cmap("tab10")
        for i, (name, run) in enumerate(runs.items()):
            tr, va = np.asarray(run["train"]), np.asarray(run["val"])
            ep = np.asarray(run.get("epochs", np.arange(1, len(tr) + 1)))
            ax.plot(ep, tr, color=cmap(i), ls="--", alpha=0.7)
            ax.plot(ep, va, color=cmap(i), label=name)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("relative L2 loss (normalised)")
        ax.set_title("All runs (solid = val, dashed = train)", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
        fig.tight_layout()
        out = CURVES / "loss_all_runs.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out.relative_to(ROOT))
        plot_by_model(runs)


if __name__ == "__main__":
    main()
