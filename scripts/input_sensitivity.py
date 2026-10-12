"""How much does the trajectory move when every input is perturbed by a tiny relative amount?

For each (A1max, A2max) regime: draw samples, perturb m_p, b, A, f, phi by +eps (relative), re-simulate,
and report the median / 90th percentile of |x_perturbed - x| / std(x) at several times. If this is already
O(1) at t = T for eps = 1e-3, no surrogate can be accurate there: the target is too sensitive to learn
from finite data, and the horizon T or amplitude caps are the thing to change.

Run: python -m scripts.input_sensitivity     (about 1 minute)
Read-only analysis; the decision is yours.
"""
import numpy as np

from src.data import RANGES, T_GRID, sample_inputs, simulate

EPS, N = 1e-3, 40
TIMES = [0.5, 1.0, 2.0, 3.0]
idx = [int(np.argmin(np.abs(T_GRID - t))) for t in TIMES]

print(f"relative input perturbation eps = {EPS:g}; entries are median / p90 of per-trajectory ||dx||/std(x) over channels")
print(f"{'A1max':>6}{'A2max':>6} " + " ".join(f"{'t=%.1fs' % t:>16}" for t in TIMES))
for a1, a2 in [(15, 15), (8, 2), (5, 2)]:
    params, coef = sample_inputs(N, np.random.default_rng(11), RANGES)
    coef[:, 0, 0] *= a1 / RANGES["A"][1]
    coef[:, 1, 0] *= a2 / RANGES["A"][1]
    D = []
    for p, c in zip(params, coef):
        x = simulate(p, c)
        xp = simulate(p * (1 + EPS), c * (1 + EPS))
        s = x.std(axis=0) + 1e-9  # per-channel scale of this trajectory
        D.append(np.linalg.norm((xp - x) / s, axis=1) / 2)  # RMS over 4 channels
    D = np.array(D)
    print(f"{a1:6d}{a2:6d} " + " ".join(f"{np.median(D[:, i]):7.1e}/{np.percentile(D[:, i], 90):7.1e}" for i in idx))
