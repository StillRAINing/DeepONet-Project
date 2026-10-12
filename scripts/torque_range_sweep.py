"""Compare candidate torque-amplitude caps (A1, A2) before fixing RANGES in data.py.

For each cap it reports how wild the motion is and how much the torque matters:
    wrap      fraction of trajectories with |q| > 2 pi at some time
    q-p99     99th percentile of |q| over all joints/times (rad)
    influence mean relative L2 distance between the trajectory and the SAME arm with zero torque
              (near 0: torque barely matters, so the 'function input' is nearly irrelevant)
    rk4-err   mean relative L2 of substeps=8 vs substeps=32 (is the ground truth converged?)

Run: python -m scripts.torque_range_sweep      (about a minute; 40 samples per cap)
Read-only analysis; the decision about RANGES is yours.
"""
import numpy as np

from src.data import RANGES, sample_inputs, simulate

N = 40
CAPS = [(15, 15), (8, 8), (8, 2), (5, 5), (5, 2), (3, 3), (2, 2)]


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


print(f"{'A1max':>6}{'A2max':>6}{'wrap':>7}{'q-p99':>8}{'influence':>11}{'rk4-err':>10}")
for a1, a2 in CAPS:
    rng = np.random.default_rng(7)
    params, coef = sample_inputs(N, rng, RANGES)
    coef[:, 0, 0] *= a1 / RANGES["A"][1]  # rescale the drawn amplitudes to the new caps
    coef[:, 1, 0] *= a2 / RANGES["A"][1]
    trajs, infl, err = [], [], []
    for p, c in zip(params, coef):
        x8 = simulate(p, c, substeps=8)
        c0 = c.copy()
        c0[:, 0] = 0.0
        trajs.append(x8)
        infl.append(rel(x8, simulate(p, c0, substeps=8)))
        err.append(rel(x8, simulate(p, c, substeps=32)))
    X = np.stack(trajs)
    wrap = (np.abs(X[..., :2]).max(axis=(1, 2)) > 2 * np.pi).mean()
    print(f"{a1:6d}{a2:6d}{wrap:7.2f}{np.percentile(np.abs(X[..., :2]), 99):8.2f}"
          f"{np.mean(infl):11.2f}{np.mean(err):10.1e}")
