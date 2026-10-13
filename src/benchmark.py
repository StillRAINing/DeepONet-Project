"""Timing, memory and FLOP estimates for RK4, solve_ivp, DeepONet and MLP.

Scaffolding only (AI-written, see AI_LOG.md). It answers "how long does one query take?" for
four solvers, and "how many queries until the surrogate pays for its own data + training?".

Fair-comparison rules baked in (CLAUDE.md weak point 3):
    * the scalar Python-loop RK4 is the *slow* reference, not the fair competitor;
    * ``rk4_batched`` is a vectorised NumPy RK4 over B trajectories at once: the fair CPU rival
      for a batched network;
    * ``solve_ivp`` is timed at a stated tolerance; pick one that matches the surrogate's error;
    * the surrogate cost must include data generation + training (``break_even_queries``).

``batched_rhs`` re-implements the arm dynamics on arrays of shape (B, ...) purely for timing.
It must agree with ``dynamics.state_derivative`` (checked in tests/test_benchmark.py); the
scalar code in ``dynamics.py`` stays the single source of truth.

Run: ``python -m src.benchmark`` (sine family, data/dataset.npz), or
``python -m src.benchmark --checkpoint checkpoints/<name>.pt`` (smooth family, trained model, val split).
"""
import time
import tracemalloc
from dataclasses import replace

import numpy as np

from .dynamics import ArmParams, state_derivative


# --------------------------------------------------------------------------- timing helpers
def time_fn(fn, repeat: int = 5, warmup: int = 1) -> dict:
    """Wall-clock ``fn()`` ``repeat`` times after ``warmup`` calls. Returns seconds (median/min/max)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median": float(np.median(ts)), "min": min(ts), "max": max(ts), "repeat": repeat}


def peak_memory_mb(fn) -> float:
    """Peak Python-heap allocation (tracemalloc, includes NumPy buffers) while running ``fn()``.

    Does not see PyTorch's C++ allocations; for the network use ``n_params * 4 bytes`` plus the
    activation size from ``deeponet_flops`` style counting, or run in a separate process and read RSS.
    """
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


# --------------------------------------------------------------------------- vectorised NumPy reference
def batched_torque(coef: np.ndarray, t: float) -> np.ndarray:
    """Torque (B, 2) at time t for either family, told apart by the shape of ``coef``.

    (B, 2, 3): sine family, tau_j = A_j sin(2 pi f_j t + phi_j)            (data.torque_fn)
    (B, 2, 2, K): smooth family, tau_j = sum_k a_jk cos(w_k t) + b_jk sin(w_k t),
                  w_k = 2 pi k / PERIOD                                     (data.smooth_torque_fn)
    """
    if coef.ndim == 3 and coef.shape[-1] == 3:
        return coef[:, :, 0] * np.sin(2 * np.pi * coef[:, :, 1] * t + coef[:, :, 2])
    if coef.ndim == 4 and coef.shape[2] == 2:
        from .data import PERIOD
        w = 2 * np.pi * np.arange(1, coef.shape[-1] + 1) * t / PERIOD          # (K,)
        return coef[:, :, 0] @ np.cos(w) + coef[:, :, 1] @ np.sin(w)
    raise ValueError(f"unrecognised torque coefficient shape {coef.shape}")


def batched_rhs(t: float, X: np.ndarray, m_p: np.ndarray, b: np.ndarray, coef: np.ndarray,
                base: ArmParams = ArmParams()) -> np.ndarray:
    """dx/dt for B trajectories at once.

    Args:
        t: scalar time.
        X: (B, 4) states [q1, q2, q1dot, q2dot].
        m_p, b: (B,) payload mass and friction.
        coef: (B, 2, 3) sine coefficients [A, f, phi] per joint, or (B, 2, 2, K) smooth Fourier
              coefficients [joint, cos/sin, mode] (see ``batched_torque``).
    Returns:
        (B, 4).
    """
    p = base
    q1, q2, v1, v2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    # payload folded into link 2 (same algebra as dynamics.combine_payload, vectorised in m_p)
    m2e = p.m2 + m_p
    lc2e = (p.m2 * p.lc2 + m_p * p.l2) / m2e
    I2e = p.I2 + p.m2 * (lc2e - p.lc2) ** 2 + m_p * (p.l2 - lc2e) ** 2
    c2, s2 = np.cos(q2), np.sin(q2)
    M11 = p.m1 * p.lc1**2 + p.I1 + m2e * (p.l1**2 + lc2e**2 + 2 * p.l1 * lc2e * c2) + I2e
    M12 = m2e * (lc2e**2 + p.l1 * lc2e * c2) + I2e
    M22 = m2e * lc2e**2 + I2e
    h = m2e * p.l1 * lc2e * s2
    g1 = (p.m1 * p.lc1 + m2e * p.l1) * p.g * np.cos(q1) + m2e * lc2e * p.g * np.cos(q1 + q2)
    g2 = m2e * lc2e * p.g * np.cos(q1 + q2)
    tau = batched_torque(coef, t)  # (B, 2)
    r1 = tau[:, 0] + h * (2 * v1 * v2 + v2**2) - g1 - b * v1
    r2 = tau[:, 1] - h * v1**2 - g2 - b * v2
    det = M11 * M22 - M12**2  # closed-form 2x2 solve
    a1 = (M22 * r1 - M12 * r2) / det
    a2 = (-M12 * r1 + M11 * r2) / det
    return np.stack([v1, v2, a1, a2], axis=1)


def rk4_batched(params: np.ndarray, coef: np.ndarray, t_grid: np.ndarray, x0: np.ndarray,
                substeps: int = 8, base: ArmParams = ArmParams()) -> np.ndarray:
    """Vectorised RK4 over all B trajectories. Same step size as ``data.simulate`` by default.

    Returns (B, len(t_grid), 4).
    """
    B = params.shape[0]
    m_p, b = params[:, 0], params[:, 1]
    f = lambda t, X: batched_rhs(t, X, m_p, b, coef, base)
    X = np.tile(np.asarray(x0, float), (B, 1))
    out = np.empty((B, len(t_grid), 4))
    out[:, 0] = X
    for i in range(len(t_grid) - 1):
        h = (t_grid[i + 1] - t_grid[i]) / substeps
        t = t_grid[i]
        for _ in range(substeps):
            k1 = f(t, X)
            k2 = f(t + h / 2, X + h / 2 * k1)
            k3 = f(t + h / 2, X + h / 2 * k2)
            k4 = f(t + h, X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[:, i + 1] = X
    return out


def solve_ivp_one(params: np.ndarray, coef: np.ndarray, t_grid: np.ndarray, x0: np.ndarray,
                  rtol: float = 1e-6, atol: float = 1e-9, method: str = "RK45",
                  base: ArmParams = ArmParams()) -> np.ndarray:
    """SciPy ``solve_ivp`` for one sample at a stated tolerance. Returns (len(t_grid), 4)."""
    from scipy.integrate import solve_ivp

    from .data import smooth_torque_fn, torque_fn

    p = replace(base, m_p=float(params[0]), b=float(params[1]))
    tau = smooth_torque_fn(coef) if np.ndim(coef) == 3 else torque_fn(coef)
    sol = solve_ivp(lambda t, x: state_derivative(t, x, p, tau), (t_grid[0], t_grid[-1]), x0,
                    t_eval=t_grid, rtol=rtol, atol=atol, method=method)
    return sol.y.T


# --------------------------------------------------------------------------- cost accounting
def mlp_flops(d_in: int, width: int, depth: int, d_out: int) -> int:
    """Multiply-add FLOPs (2 per MAC) of one forward pass of ``model.mlp`` for ONE input row."""
    dims = [d_in] + [width] * depth + [d_out]
    return int(sum(2 * a * c for a, c in zip(dims[:-1], dims[1:])))


def deeponet_flops(d_u: int, n_t: int, p: int, width: int, branch_depth: int, trunk_depth: int,
                   n_out: int = 4) -> int:
    """FLOPs for ONE sample at n_t query times (activations ignored).

    Branch runs once per sample, trunk once per query time, combination is a length-p dot
    product per (time, channel). (The trunk does not depend on the sample, so over a batch on
    a shared grid it could be cached; this counts the un-cached cost.)
    """
    branch = mlp_flops(d_u, width, branch_depth, n_out * p)
    trunk = n_t * mlp_flops(1, width, trunk_depth, n_out * p)
    combine = n_t * n_out * 2 * p
    return branch + trunk + combine


def mlp_baseline_flops(d_u: int, n_t: int, width: int, depth: int, n_out: int = 4) -> int:
    """FLOPs for ONE sample at n_t query times: the full MLP runs once per (sample, time) row."""
    return n_t * mlp_flops(d_u + 1, width, depth, n_out)


def break_even_queries(t_data_s: float, t_train_s: float, t_solver_per_query_s: float,
                       t_surrogate_per_query_s: float) -> float:
    """Number of trajectories after which the surrogate is cheaper in total.

    N* = (data generation + training) / (solver cost per query - surrogate cost per query).
    Returns inf if the surrogate is not faster per query.
    """
    gain = t_solver_per_query_s - t_surrogate_per_query_s
    return float("inf") if gain <= 0 else (t_data_s + t_train_s) / gain


# --------------------------------------------------------------------------- driver
def run(n_samples: int = 64, model=None, repeat: int = 3) -> dict:
    """Time every method on ``n_samples`` test trajectories. Returns seconds per trajectory.

    ``model`` is any ``model(u, t) -> (B, n_t, 4)`` module (untrained weights time identically).
    """
    from .data import T_GRID, X0, load, simulate
    from .model import branch_input

    d = load()
    params, coef, torque = d["test_params"][:n_samples], d["test_tau_coef"][:n_samples], d["test_torque"][:n_samples]
    res = {}

    res["rk4_scalar"] = time_fn(lambda: [simulate(p, c) for p, c in zip(params[:8], coef[:8])], repeat)["median"] / 8
    res["rk4_batched"] = time_fn(lambda: rk4_batched(params, coef, T_GRID, X0), repeat)["median"] / n_samples
    for rtol in (1e-3, 1e-6):
        res[f"solve_ivp_rtol{rtol:g}"] = time_fn(
            lambda: [solve_ivp_one(p, c, T_GRID, X0, rtol=rtol) for p, c in zip(params[:8], coef[:8])], repeat
        )["median"] / 8

    if model is not None:
        import torch

        u = branch_input(params, torque)
        t = torch.as_tensor(T_GRID / T_GRID[-1], dtype=torch.float32)  # use the SAME time scaling as training
        model.eval()
        with torch.no_grad():
            res["surrogate_batch"] = time_fn(lambda: model(u, t), repeat * 3)["median"] / n_samples
            res["surrogate_single"] = time_fn(lambda: model(u[:1], t), repeat * 10)["median"]
    return res


def batch_sweep(model, batches=(1, 8, 64, 300), repeat: int = 3) -> dict:
    """Seconds per trajectory vs batch size for vectorised RK4 and the surrogate.

    The speed-up is NOT a constant: batched NumPy RK4 amortises its Python overhead over the batch,
    so the ratio shrinks as the batch grows. Report the batch size next to every speed-up.
    """
    import torch

    from .data import T_GRID, X0, load
    from .model import branch_input

    d = load()
    t = torch.as_tensor(T_GRID / T_GRID[-1], dtype=torch.float32)
    model.eval()
    out = {}
    for B in batches:
        params, coef, torque = d["test_params"][:B], d["test_tau_coef"][:B], d["test_torque"][:B]
        u = branch_input(params, torque)
        rk4 = time_fn(lambda: rk4_batched(params, coef, T_GRID, X0), repeat)["median"] / B
        with torch.no_grad():
            sur = time_fn(lambda: model(u, t), repeat * 3)["median"] / B
        out[B] = {"rk4_batched": rk4, "surrogate": sur, "speedup": rk4 / sur}
    return out


def run_checkpoint(path, n_samples: int = 64, repeat: int = 3, rtols=(1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-6)) -> dict:
    """Time a TRAINED checkpoint against the solvers on the smooth family, VALIDATION split.

    The surrogate time covers the whole query path: build the raw input, normalise it, run the
    network, decode to raw units (what a user of the surrogate would actually pay). For each
    ``solve_ivp`` tolerance it also reports the median headline error against the stored RK4 truth
    (``evaluate.rel_l2_per_sample``, channels scaled by the train std), so a tolerance whose error
    matches the surrogate's can be chosen (CLAUDE.md weak point 3).
    Returns {name: {"s_per_traj": ..., "median_err": ...}}.
    """
    import torch

    from .data import DATA_PATH_SMOOTH, T_FINAL, T_GRID, X0
    from .data import load as load_data
    from .evaluate import rel_l2_per_sample
    from .train import load_checkpoint, raw_inputs

    model, in_norm, out_norm, hist = load_checkpoint(path)
    mode = hist.get("input_mode", "samples")
    d = load_data(DATA_PATH_SMOOTH)
    sub = {k: v[:n_samples] for k, v in d.items() if k.startswith("val_")}
    params, coef, true = sub["val_params"], sub["val_tau_coef"], sub["val_traj"].astype(np.float64)
    std = out_norm.std.numpy().astype(np.float64)
    t = torch.as_tensor(T_GRID / T_FINAL, dtype=torch.float32)
    res = {}

    def surrogate(rows):
        u = in_norm.encode(torch.as_tensor(raw_inputs({k: v[rows] for k, v in sub.items()}, "val", mode),
                                           dtype=torch.float32))
        with torch.no_grad():
            return out_norm.decode(model(u, t)).numpy()

    pred = surrogate(slice(None))
    res["surrogate_batch"] = {"s_per_traj": time_fn(lambda: surrogate(slice(None)), repeat * 3)["median"] / n_samples,
                              "median_err": float(np.median(rel_l2_per_sample(pred, true, std)))}
    res["surrogate_single"] = {"s_per_traj": time_fn(lambda: surrogate(slice(0, 1)), repeat * 10)["median"],
                               "median_err": res["surrogate_batch"]["median_err"]}
    res["rk4_batched"] = {"s_per_traj": time_fn(lambda: rk4_batched(params, coef, T_GRID, X0), repeat)["median"] / n_samples,
                          "median_err": float(np.median(rel_l2_per_sample(rk4_batched(params, coef, T_GRID, X0), true, std)))}
    n_ivp = min(8, n_samples)
    for rtol in rtols:
        run_ivp = lambda: np.stack([solve_ivp_one(p, c, T_GRID, X0, rtol=rtol, atol=rtol * 1e-3)
                                    for p, c in zip(params[:n_ivp], coef[:n_ivp])])
        err = rel_l2_per_sample(run_ivp(), true[:n_ivp], std)
        res[f"solve_ivp_rtol{rtol:g}"] = {"s_per_traj": time_fn(run_ivp, repeat)["median"] / n_ivp,
                                          "median_err": float(np.median(err))}
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Timing. Close other heavy programs (e.g. a training sweep) first.")
    ap.add_argument("--checkpoint", default=None, help="time a trained model on the smooth family (val split)")
    ap.add_argument("--n", type=int, default=64, help="number of trajectories in the batch")
    args = ap.parse_args()
    if args.checkpoint:
        print(f"{'method':24s} {'ms / trajectory':>16s} {'median rel. L2':>16s}")
        for k, v in run_checkpoint(args.checkpoint, n_samples=args.n).items():
            print(f"{k:24s} {v['s_per_traj'] * 1e3:16.3f} {v['median_err']:16.2e}")
    else:
        out = run()
        for k, v in out.items():
            print(f"{k:24s} {v * 1e3:10.3f} ms / trajectory")
