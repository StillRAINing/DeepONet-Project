"""PROTOTYPE (not wired into data.py): a smooth random torque family with many degrees of freedom.

Why: with tau_j = A sin(2 pi f t + phi) the 'function input' has only 6 free numbers, so the problem is a
parametric regression (CLAUDE.md weak point 1). A random Fourier series with K modes per joint has
2K coefficients per joint, with a spectrum that decays like a smooth Gaussian-process sample. It is still
exactly evaluable at any t (needed by the RK4 right-hand side) and exactly sampleable on the 151-point grid.

    tau_j(t) = scale_j * sum_{k=1..K} k^(-decay) * (a_jk sin(2 pi k t / T) + b_jk cos(2 pi k t / T)),   a, b ~ N(0, 1)

Run: python -m scripts.torque_family_prototype [--plot]
Prints how the family compares with the sinusoid one (effective dimension, motion magnitude).
Decisions (K, decay, scale, whether to adopt it at all) are yours.
"""
import sys

import numpy as np

from src.data import N_T, T_FINAL, T_GRID, X0
from src.dynamics import ArmParams, state_derivative
from src.solvers import rk4_integrate

K, DECAY = 6, 1.0
SCALE = np.array([4.0, 1.0])  # N m per joint; tune with torque_range_sweep-style screens


def sample_fourier_coef(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2, 2, K): [joint, sin/cos, mode] coefficients with the decay already applied."""
    k = np.arange(1, K + 1)
    return rng.standard_normal((n, 2, 2, K)) * (k ** -DECAY) * SCALE[None, :, None, None]


def torque_fn_fourier(coef: np.ndarray):
    """coef (2, 2, K) -> t -> [tau1, tau2]."""
    k = np.arange(1, K + 1)

    def tau(t):
        w = 2 * np.pi * k * t / T_FINAL
        return coef[:, 0] @ np.sin(w) + coef[:, 1] @ np.cos(w)

    return tau


def torque_samples_fourier(coef: np.ndarray, t_grid: np.ndarray = T_GRID) -> np.ndarray:
    k = np.arange(1, K + 1)[:, None]
    w = 2 * np.pi * k * t_grid[None] / T_FINAL
    return coef[:, 0] @ np.sin(w) + coef[:, 1] @ np.cos(w)  # (2, n_t)


def simulate_fourier(params, coef, substeps: int = 8, base: ArmParams = ArmParams()) -> np.ndarray:
    from dataclasses import replace

    p = replace(base, m_p=float(params[0]), b=float(params[1]))
    tau = torque_fn_fourier(coef)
    fine = np.linspace(0.0, T_FINAL, (N_T - 1) * substeps + 1)
    return rk4_integrate(lambda t, x: state_derivative(t, x, p, tau), X0, fine)[::substeps]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 60
    coef = sample_fourier_coef(n, rng)
    params = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0.05, 0.5, n)])
    X = np.stack([simulate_fourier(p, c) for p, c in zip(params, coef)])
    T = np.stack([torque_samples_fourier(c) for c in coef]).reshape(n, -1)
    sv = np.linalg.svd(T - T.mean(0), compute_uv=False)
    cum = np.cumsum(sv**2) / np.sum(sv**2)
    print(f"free coefficients per sample: {2 * 2 * K} torque + 2 params = {2 * 2 * K + 2}")
    print(f"torque samples needed to explain 99% / 99.9% of variance: {np.searchsorted(cum, 0.99) + 1} / {np.searchsorted(cum, 0.999) + 1}")
    print(f"max|tau| per joint (p99): {np.percentile(np.abs(T.reshape(n, 2, -1)), 99, axis=(0, 2))}")
    print(f"|q| p99 = {np.percentile(np.abs(X[..., :2]), 99):.2f} rad, fraction wrapping past 2 pi = "
          f"{(np.abs(X[..., :2]).max(axis=(1, 2)) > 2 * np.pi).mean():.2f}, finite = {np.isfinite(X).all()}")
    if "--plot" in sys.argv:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(T_GRID, T.reshape(n, 2, -1)[:8, 0].T, lw=0.8)
        ax[0].set(title="joint-1 torque samples", xlabel="t (s)", ylabel="N m")
        ax[1].plot(T_GRID, X[:8, :, 1].T, lw=0.8)
        ax[1].set(title="resulting q2(t)", xlabel="t (s)", ylabel="rad")
        fig.tight_layout()
        fig.savefig("results/torque_family_prototype.png", dpi=130)
        print("saved results/torque_family_prototype.png")
