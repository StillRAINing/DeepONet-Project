"""Descriptive figures of the dataset for the report (no model involved).

Run: python -m scripts.data_figures        -> results/data_*.png
Colour-blind-safe palette; each figure states what the split is.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import load  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
C_TRAIN, C_OOD = "#0072B2", "#D55E00"
NAMES = ["$q_1$ (rad)", "$q_2$ (rad)", r"$\dot q_1$ (rad/s)", r"$\dot q_2$ (rad/s)"]
d = load()
t = d["t_grid"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", OUT / name)


# 1. fan of trajectories: all four channels, 40 random train samples + 40 OOD
rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for k, a in enumerate(ax.ravel()):
    for split, c, lab in (("train", C_TRAIN, "train"), ("ood", C_OOD, "OOD")):
        X = d[f"{split}_traj"]
        for i in rng.choice(len(X), 40, replace=False):
            a.plot(t, X[i, :, k], color=c, lw=0.5, alpha=0.5, label=lab if i == 0 else None)
    a.set_ylabel(NAMES[k])
    a.set_yscale("symlog", linthresh=5)  # trajectories span 3 orders of magnitude
for a in ax[1]:
    a.set_xlabel("t (s)")
ax[0, 0].plot([], [], color=C_TRAIN, label="train"); ax[0, 0].plot([], [], color=C_OOD, label="OOD")
ax[0, 0].legend(frameon=False)
fig.suptitle("40 random trajectories per set (symlog y axis)")
save(fig, "data_trajectories.png")

# 2. channel distributions (train vs ood): why per-channel normalisation matters
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for k, a in enumerate(ax):
    for split, c in (("train", C_TRAIN), ("ood", C_OOD)):
        v = d[f"{split}_traj"][..., k].ravel()
        lim = np.percentile(np.abs(d["train_traj"][..., k]), 99)
        a.hist(np.clip(v, -lim, lim), bins=60, density=True, histtype="step", color=c, label=split)
    a.set_xlabel(NAMES[k] + " (clipped at train p99)")
ax[0].legend(frameon=False)
save(fig, "data_channel_hist.png")

# 3. input-space coverage of the two scalar parameters
fig, ax = plt.subplots(figsize=(4.5, 3.8))
for split, c in (("train", C_TRAIN), ("val", "#009E73"), ("test", "#CC79A7"), ("ood", C_OOD)):
    p = d[f"{split}_params"]
    ax.scatter(p[:, 0], p[:, 1], s=6, color=c, alpha=0.6, label=split)
ax.set(xlabel="payload $m_p$ (kg)", ylabel="friction $b$ (N m s/rad)")
ax.legend(frameon=False, markerscale=2)
save(fig, "data_param_coverage.png")

# 4. how motion magnitude depends on the inputs (explains hard cases)
tr = d["train_traj"]
peak = np.abs(tr[..., 1]).max(axis=1)
A2 = d["train_tau_coef"][:, 1, 0]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
ax[0].scatter(A2, peak, s=5, color=C_TRAIN, alpha=0.5)
ax[0].set(xlabel="joint-2 torque amplitude $A_2$ (N m)", ylabel=r"peak $|q_2|$ (rad)", yscale="log")
ax[1].scatter(d["train_params"][:, 1], peak, s=5, color=C_TRAIN, alpha=0.5)
ax[1].set(xlabel="friction $b$", ylabel=r"peak $|q_2|$ (rad)", yscale="log")
save(fig, "data_peak_vs_inputs.png")
