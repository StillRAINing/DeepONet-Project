"""Descriptive statistics of the generated dataset (informs normalisation choices).

Run: python -m scripts.data_stats      (needs data/dataset.npz from ``python -m src.data``)
Read-only: prints numbers, makes no decisions. The normalisation scheme is yours to choose.
"""
import numpy as np

from src.data import load

d = load()
names = ["q1", "q2", "q1dot", "q2dot"]
for split in ("train", "val", "test", "ood"):
    X = d[f"{split}_traj"]
    print(f"\n== {split}: {X.shape}  finite={np.isfinite(X).all()}")
    flat = X.reshape(-1, 4)
    print(f"{'chan':6}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'p99|x|':>9}")
    for k, n in enumerate(names):
        c = flat[:, k]
        print(f"{n:6}{c.mean():9.3f}{c.std():9.3f}{c.min():9.3f}{c.max():9.3f}{np.percentile(np.abs(c), 99):9.3f}")
    big = (np.abs(X[..., :2]).max(axis=(1, 2)) > 2 * np.pi).mean()
    print(f"fraction of trajectories with |q| > 2 pi at some time: {big:.3f}")
    # share of total squared norm carried by each channel (what a plain relative L2 'sees')
    e = (X**2).sum(axis=(0, 1))
    print("share of sum(x^2) per channel:", np.round(e / e.sum(), 3))
    # growth of spread with time: error-prone late times
    sd_t = X.std(axis=0)
    print("per-channel std at t=0, T/2, T:", np.round(sd_t[[0, len(sd_t) // 2, -1]], 2).tolist())

tr = d["train_traj"]
print("\nTrain-only stats you would use for normalisation (do NOT use val/test/ood):")
print(" mean per channel:", np.round(tr.reshape(-1, 4).mean(0), 3))
print(" std  per channel:", np.round(tr.reshape(-1, 4).std(0), 3))
print("Branch inputs: m_p, b, torque samples")
print(" params mean/std:", np.round(d["train_params"].mean(0), 3), np.round(d["train_params"].std(0), 3))
print(" torque mean/std:", round(float(d["train_torque"].mean()), 3), round(float(d["train_torque"].std()), 3))
print(" torque samples per sample are redundant: rank of train torque matrix =",
      np.linalg.matrix_rank(d["train_torque"].reshape(len(tr), -1), tol=1e-6),
      "of", d["train_torque"][0].size)
