"""Seeded dataset generation: sample parameters and torque profiles, simulate, split train/val/test.

Each sample is one trajectory of the arm:
    params   (2,)       [m_p, b]
    torque   (2, N_T)   tau_j(t_i) = A_j sin(2 pi f_j t_i + phi_j) on the fixed grid
    traj     (N_T, 4)   x(t_i) = [q1, q2, q1dot, q2dot], from RK4

Ground truth is RK4 with ``substeps`` internal steps per output interval.
Besides the train/val/test split (in-range) there is an out-of-range set with
heavier payload and higher friction, used only for extrapolation tests.

Run ``python -m src.data`` to write ``data/dataset.npz``.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from .dynamics import ArmParams, state_derivative
from .solvers import rk4_integrate

SEED = 3001

# Fixed horizon, output grid and initial condition (provisional, see README).
T_FINAL = 3.0
N_T = 151
T_GRID = np.linspace(0.0, T_FINAL, N_T)
X0 = np.zeros(4)  # arm held horizontal at rest

# In-range sampling domain (provisional, see README).
RANGES = {
    "m_p": (0.0, 1.0),
    "b": (0.05, 0.5),
    "A": (0.0, 15.0),  # gravity torque at the start pose is ~12 N m, so A must be able to compete
    "f": (0.2, 1.0),
}
# Out-of-range: heavier payload, higher friction; torque family unchanged.
OOD_RANGES = {**RANGES, "m_p": (1.0, 1.5), "b": (0.5, 0.8)}

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.npz"


def sample_inputs(n: int, rng: np.random.Generator, ranges: dict = RANGES):
    """Draw n sets of inputs.

    Returns:
        params: (n, 2) array [m_p, b].
        tau_coef: (n, 2, 3) array of [A, f, phi] per joint.
    """
    params = np.column_stack(
        [rng.uniform(*ranges["m_p"], n), rng.uniform(*ranges["b"], n)]
    )
    tau_coef = np.stack(
        [
            rng.uniform(*ranges["A"], (n, 2)),
            rng.uniform(*ranges["f"], (n, 2)),
            rng.uniform(0.0, 2 * np.pi, (n, 2)),
        ],
        axis=-1,
    )
    return params, tau_coef


def torque_fn(coef: np.ndarray):
    """Return t -> [tau1, tau2] for coef of shape (2, 3) = [A, f, phi] per joint."""
    A, f, phi = coef[:, 0], coef[:, 1], coef[:, 2]
    return lambda t: A * np.sin(2 * np.pi * f * t + phi)


def torque_samples(coef: np.ndarray, t_grid: np.ndarray = T_GRID) -> np.ndarray:
    """Sample the torque profile on t_grid. Returns shape (2, len(t_grid))."""
    A, f, phi = coef[:, 0:1], coef[:, 1:2], coef[:, 2:3]
    return A * np.sin(2 * np.pi * f * t_grid + phi)


def simulate(
    params: np.ndarray,
    coef: np.ndarray,
    base: ArmParams = ArmParams(),
    substeps: int = 8,
) -> np.ndarray:
    """Simulate one trajectory with RK4. Returns shape (N_T, 4) on T_GRID."""
    p = replace(base, m_p=float(params[0]), b=float(params[1]))
    tau = torque_fn(coef)
    f = lambda t, x: state_derivative(t, x, p, tau)
    # Refine the output grid by `substeps`, then keep every substeps-th point.
    fine = np.linspace(0.0, T_FINAL, (N_T - 1) * substeps + 1)
    return rk4_integrate(f, X0, fine)[::substeps]


def make_set(n: int, rng: np.random.Generator, ranges: dict = RANGES, substeps: int = 8):
    """Generate n samples. Returns dict with params, tau_coef, torque, traj."""
    params, tau_coef = sample_inputs(n, rng, ranges)
    torque = np.stack([torque_samples(c) for c in tau_coef])
    traj = np.stack([simulate(p, c, substeps=substeps) for p, c in zip(params, tau_coef)])
    return {"params": params, "tau_coef": tau_coef, "torque": torque, "traj": traj}


def generate(
    n_total: int = 2000,
    n_ood: int = 200,
    frac: tuple = (0.7, 0.15, 0.15),
    seed: int = SEED,
    substeps: int = 8,
) -> dict:
    """Build the full dataset: train / val / test (in-range) plus ood.

    Returns a flat dict with keys like ``train_params``, ``val_traj``,
    ``ood_torque`` and the shared ``t_grid``.
    """
    rng = np.random.default_rng(seed)
    full = make_set(n_total, rng, RANGES, substeps)
    n_tr = int(frac[0] * n_total)
    n_va = int(frac[1] * n_total)
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n_total)}

    out = {"t_grid": T_GRID, "x0": X0, "seed": np.array(seed)}
    for name, (lo, hi) in bounds.items():
        for key, arr in full.items():
            out[f"{name}_{key}"] = arr[lo:hi]
    for key, arr in make_set(n_ood, rng, OOD_RANGES, substeps).items():
        out[f"ood_{key}"] = arr
    return out


def load(path: Path = DATA_PATH) -> dict:
    """Load a dataset written by ``python -m src.data``."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":
    data = generate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA_PATH, **data)
    for split in ("train", "val", "test", "ood"):
        print(split, data[f"{split}_traj"].shape)
    print("saved", DATA_PATH)
