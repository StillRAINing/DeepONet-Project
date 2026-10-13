"""Seeded dataset generation: sample parameters and torque profiles, simulate, split train/val/test.

Each sample is one trajectory of the arm:
    params   (2,)       [m_p, b]
    torque   (2, N_T)   tau_j(t_i) = A_j sin(2 pi f_j t_i + phi_j) on the fixed grid
    traj     (N_T, 4)   x(t_i) = [q1, q2, q1dot, q2dot], from RK4

Ground truth is RK4 with ``substeps`` internal steps per output interval.
Besides the train/val/test split (in-range) there is an out-of-range set with
heavier payload and higher friction, used only for extrapolation tests.

Run ``python -m src.data`` to write ``data/dataset.npz`` (sine torque family).
Run ``python -m src.data --smooth`` to write ``data/dataset_smooth.npz`` (smooth random torque
family, the main experiment: see the section below ``generate`` and docs/DESIGN_LOG.md).
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
    "A": (0.0, 15.0),  # gravity torque at the start pose is ~15 N m (m_p=0) to ~24 N m (m_p=1) on joint 1, so A must be able to compete
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
    """Simulate one trajectory with RK4. Returns shape (N_T, 4) on T_GRID.

    Raises:
        ValueError: ``params`` is not ``[m_p, b]`` with finite m_p >= 0, b >= 0, or ``coef`` is not
            a finite (2, 3) array ``[A, f, phi]`` per joint with A >= 0, f > 0 (see ``validate``).
    """
    from .validate import validate_coef, validate_params  # lazy: validate imports this module

    params = validate_params(params, ranges=None)[0]
    coef = validate_coef(coef)
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


# ---------------------------------------------------------------------------
# Smooth random torque family (docs/DESIGN_LOG.md: D10, D14, A6).
#
#   tau_j(t) = sum_{k=1..K} c_jk cos(2 pi k t / P) + s_jk sin(2 pi k t / P),   P = 2 T
#
# The coefficients are Gaussian with standard deviation exp(-(omega_k ell)^2 / 4), so the variance
# follows an RBF-kernel spectrum with correlation length ell (in normalised time t / T). Each joint's
# shape is then rescaled so that max_t |tau_j| over [0, T] equals a peak A_j drawn from a range.
# It can be evaluated at any t (RK4 needs mid-step values) and sampled on the 151-point grid.
# ---------------------------------------------------------------------------
ELL = 0.2  # correlation length in normalised time t / T (Lu et al. 2021)
K_MODES = 6  # Fourier modes per joint -> 2 K = 12 coefficients per joint
PERIOD = 2 * T_FINAL  # series period 2 T: avoids forcing tau(0) = tau(T)
SMOOTH_RANGES = {"m_p": RANGES["m_p"], "b": RANGES["b"], "A": (0.0, 6.0)}  # A: peak torque (N m)
PEAK_GRID = np.linspace(0.0, T_FINAL, 601)  # grid on which the peak of a torque is measured

# Out-of-range sets for the smooth family. Amplitude "shells" are disjoint from the training range.
OOD_SMOOTH = {
    "ood": dict(ranges={**SMOOTH_RANGES, "m_p": OOD_RANGES["m_p"], "b": OOD_RANGES["b"]}),
    "ood_amp1": dict(ranges={**SMOOTH_RANGES, "A": (6.0, 9.0)}),
    "ood_amp2": dict(ranges={**SMOOTH_RANGES, "A": (9.0, 12.0)}),
    "ood_smooth": dict(ranges=SMOOTH_RANGES, ell=0.1, k_modes=13),
}

DATA_PATH_SMOOTH = DATA_PATH.with_name("dataset_smooth.npz")


def spectral_std(ell: float = ELL, k_modes: int = K_MODES) -> np.ndarray:
    """Std of the mode-k coefficients, k = 1..k_modes, for an RBF spectrum of length ell (units of T)."""
    omega = 2 * np.pi * np.arange(1, k_modes + 1) / (PERIOD / T_FINAL)  # rad per unit of t / T
    return np.exp(-((omega * ell) ** 2) / 4)


def smooth_torque_samples(coef: np.ndarray, t_grid: np.ndarray = T_GRID) -> np.ndarray:
    """Sample the torque on t_grid. coef: (2, 2, K) = [joint, cos/sin, mode]. Returns (2, len(t_grid))."""
    k = np.arange(1, coef.shape[-1] + 1)
    w = 2 * np.pi * np.outer(k, t_grid) / PERIOD
    return coef[:, 0] @ np.cos(w) + coef[:, 1] @ np.sin(w)


def smooth_torque_fn(coef: np.ndarray):
    """Return t -> [tau1, tau2] for coef of shape (2, 2, K)."""
    k = np.arange(1, coef.shape[-1] + 1)

    def tau(t):
        w = 2 * np.pi * k * t / PERIOD
        return coef[:, 0] @ np.cos(w) + coef[:, 1] @ np.sin(w)

    return tau


def sample_smooth_inputs(
    n: int,
    rng: np.random.Generator,
    ranges: dict = SMOOTH_RANGES,
    ell: float = ELL,
    k_modes: int = K_MODES,
):
    """Draw n sets of inputs for the smooth family.

    Returns:
        params: (n, 2) array [m_p, b].
        coef: (n, 2, 2, K) Fourier coefficients [joint, cos/sin, mode], already scaled so that
            max_t |tau_j(t)| over [0, T] equals the drawn peak A_j (A_j uniform in ranges["A"]).
    """
    params = np.column_stack(
        [rng.uniform(*ranges["m_p"], n), rng.uniform(*ranges["b"], n)]
    )
    shape = rng.standard_normal((n, 2, 2, k_modes)) * spectral_std(ell, k_modes)
    peak = rng.uniform(*ranges["A"], (n, 2))
    current = np.stack([np.abs(smooth_torque_samples(c, PEAK_GRID)).max(axis=1) for c in shape])
    return params, shape * (peak / current)[:, :, None, None]


def simulate_smooth(
    params: np.ndarray,
    coef: np.ndarray,
    base: ArmParams = ArmParams(),
    substeps: int = 8,
) -> np.ndarray:
    """Simulate one trajectory under a smooth-family torque with RK4. Returns shape (N_T, 4).

    Raises:
        ValueError: ``params`` is not ``[m_p, b]`` with finite m_p >= 0, b >= 0, or ``coef`` has
            NaN/infinite entries.
    """
    from .validate import validate_params  # lazy: validate imports this module

    params = validate_params(params, ranges=None)[0]
    if not np.all(np.isfinite(coef)):
        raise ValueError("coef contains NaN or infinite values")
    p = replace(base, m_p=float(params[0]), b=float(params[1]))
    tau = smooth_torque_fn(coef)
    f = lambda t, x: state_derivative(t, x, p, tau)
    fine = np.linspace(0.0, T_FINAL, (N_T - 1) * substeps + 1)
    return rk4_integrate(f, X0, fine)[::substeps]


def _simulate_smooth_args(args):
    return simulate_smooth(*args)


def make_smooth_set(
    n: int,
    rng: np.random.Generator,
    ranges: dict = SMOOTH_RANGES,
    ell: float = ELL,
    k_modes: int = K_MODES,
    substeps: int = 8,
    workers: int = 1,
):
    """Generate n smooth-family samples. Returns dict with params, tau_coef, torque, traj.

    Inputs are drawn serially from ``rng`` (so the result does not depend on ``workers``); only the
    deterministic RK4 integrations are spread over processes.
    """
    params, coef = sample_smooth_inputs(n, rng, ranges, ell, k_modes)
    torque = np.stack([smooth_torque_samples(c) for c in coef])
    jobs = [(p, c, ArmParams(), substeps) for p, c in zip(params, coef)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            traj = np.stack(list(ex.map(_simulate_smooth_args, jobs, chunksize=max(1, n // (4 * workers)))))
    else:
        traj = np.stack([_simulate_smooth_args(j) for j in jobs])
    return {"params": params, "tau_coef": coef, "torque": torque, "traj": traj}


def generate_smooth(
    n_total: int = 10000,
    n_ood: int = 200,
    frac: tuple = (0.7, 0.15, 0.15),
    seed: int = SEED,
    substeps: int = 8,
    workers: int = 1,
) -> dict:
    """Build the smooth-family dataset: train / val / test (in-range) plus the OOD sets in ``OOD_SMOOTH``.

    Returns a flat dict with keys like ``train_params``, ``val_traj``, ``ood_amp1_torque`` and the
    shared ``t_grid``. Coefficient arrays are (n, 2, 2, K) (K = 13 for ``ood_smooth``).
    """
    rng = np.random.default_rng(seed)
    full = make_smooth_set(n_total, rng, SMOOTH_RANGES, ELL, K_MODES, substeps, workers)
    n_tr = int(frac[0] * n_total)
    n_va = int(frac[1] * n_total)
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n_total)}

    out = {"t_grid": T_GRID, "x0": X0, "seed": np.array(seed), "ell": np.array(ELL),
           "k_modes": np.array(K_MODES), "period": np.array(PERIOD)}
    for name, (lo, hi) in bounds.items():
        for key, arr in full.items():
            out[f"{name}_{key}"] = arr[lo:hi]
    for name, cfg in OOD_SMOOTH.items():
        cfg = {"ell": ELL, "k_modes": K_MODES, **cfg}
        for key, arr in make_smooth_set(n_ood, rng, substeps=substeps, workers=workers, **cfg).items():
            out[f"{name}_{key}"] = arr
    return out


def load(path: Path = DATA_PATH) -> dict:
    """Load a dataset written by ``python -m src.data``."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Generate a dataset into data/.")
    ap.add_argument("--smooth", action="store_true",
                    help="smooth random torque family -> data/dataset_smooth.npz (default: sine family)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    if args.smooth:
        data, path = generate_smooth(workers=args.workers), DATA_PATH_SMOOTH
        splits = ("train", "val", "test", *OOD_SMOOTH)
    else:
        data, path = generate(), DATA_PATH
        splits = ("train", "val", "test", "ood")
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **data)
    for split in splits:
        print(split, data[f"{split}_traj"].shape)
    print("saved", path)
