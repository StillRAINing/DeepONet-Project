"""Metrics (relative L2, error vs time) and physical-consistency checks (qdot = dq/dt, energy balance).

Written by the AI on request (AI_LOG row 18). Every function here returns PER-SAMPLE arrays, not a
single number, so that means, percentiles and worst cases can all be taken afterwards (CLAUDE.md
weak point 5: never report a mean alone). Everything is in RAW units (rad, rad/s, J) unless a
``scale`` argument says otherwise. Use only checkpoints and the test / OOD splits here; never
train in this module.

Shapes: N samples, n_t = 151 times, 4 channels [q1, q2, q1dot, q2dot].
"""
from dataclasses import replace

import numpy as np
import torch

from .data import T_FINAL
from .dynamics import ArmParams, energy

CHANNELS = ("q1", "q2", "q1dot", "q2dot")


# --------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, u, t, out_norm, batch_size: int = 500) -> np.ndarray:
    """Run the model on normalised inputs and return RAW-unit trajectories as a NumPy array.

    u: (N, d_u) normalised branch input; t: (n_t,) time already rescaled to [0, 1]
    (both from ``train.make_split``). Returns (N, n_t, 4).
    """
    model.eval()
    out = [out_norm.decode(model(u[i:i + batch_size], t)) for i in range(0, len(u), batch_size)]
    return torch.cat(out).numpy().astype(np.float64)


# --------------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------------
def rel_l2_per_sample(pred, true, scale) -> np.ndarray:
    """||pred - true|| / ||true|| per sample, norm over (time, channel). Returns (N,).

    ``scale`` decides what "size" means and is DELIBERATELY not defaulted (decision point):
      * ``None``      : raw units. q (rad) and qdot (rad/s) are added up as they are, so the
                        channel with the largest numbers dominates.
      * a (4,) array  : divide each channel by it first (pass ``out_norm.std.numpy()`` to match
                        the training loss's channel scaling).
    Like the training loss, the denominator is the size of the trajectory itself (not centred).
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if scale is not None:
        pred, true = pred / np.asarray(scale), true / np.asarray(scale)
    num = np.sqrt(((pred - true) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((true ** 2).sum(axis=(1, 2)))
    return num / np.maximum(den, 1e-12)


def rel_l2_per_channel(pred, true) -> np.ndarray:
    """Relative L2 for each channel on its own: ||pred_c - true_c|| / ||true_c||, over time. (N, 4).

    Needs no scale choice, because a ratio within one channel does not depend on that channel's units.
    A channel that is near zero for a whole trajectory (a calm q2, say) gives a large ratio.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    num = np.sqrt(((pred - true) ** 2).sum(axis=1))
    den = np.sqrt((true ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def error_vs_time(pred, true, scale=None) -> np.ndarray:
    """||pred(t) - true(t)|| at each time, divided by the sample's own RMS size. Returns (N, n_t).

    The denominator is the rms of the WHOLE true trajectory (one number per sample), NOT the
    size at time t: every trajectory starts at x0 = 0, so a pointwise relative error would divide
    by zero at t = 0. (Decision point: this is my choice; the alternative is an absolute error.)
    ``scale`` is as in ``rel_l2_per_sample``.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if scale is not None:
        pred, true = pred / np.asarray(scale), true / np.asarray(scale)
    num = np.sqrt(((pred - true) ** 2).sum(axis=2))                      # (N, n_t)
    rms = np.sqrt((true ** 2).sum(axis=2).mean(axis=1, keepdims=True))   # (N, 1)
    return num / np.maximum(rms, 1e-12)


def summarise(errors) -> dict:
    """Mean, median, 90th / 95th percentile and max of a 1-D array of per-sample errors."""
    e = np.asarray(errors, dtype=np.float64)
    return {"mean": e.mean(), "median": np.median(e), "p90": np.percentile(e, 90),
            "p95": np.percentile(e, 95), "max": e.max()}


def fraction_below(errors, tolerances=(0.01, 0.05, 0.10)) -> dict:
    """Fraction of samples whose error is below each tolerance, e.g. {0.05: 0.92}. Tolerances are decision points."""
    e = np.asarray(errors, dtype=np.float64)
    return {tol: float((e < tol).mean()) for tol in tolerances}


def worst_cases(errors, k: int = 5) -> np.ndarray:
    """Indices of the k largest errors, worst first."""
    return np.argsort(np.asarray(errors))[::-1][:k]


# --------------------------------------------------------------------------------------------
# Physical consistency
# --------------------------------------------------------------------------------------------
def qdot_consistency(model, u, t, out_norm, t_final: float = T_FINAL, batch_size: int = 200) -> np.ndarray:
    """(d q_j / dt from autodiff of the q outputs) - (the qdot_j outputs), j = 1, 2. Shape (N, n_t, 2), rad/s.

    The loss never links the q channels to the qdot channels, so nothing forces them to agree.
    The model takes t rescaled to [0, 1] (tau = t / t_final), so
        dq_raw/dt = std_q * dz_q/dtau / t_final      (z = normalised output; the mean drops out).
    Each sample gets its own copy of the time grid, so the gradient of the summed output with
    respect to that copy is the per-sample, per-time derivative.
    """
    model.eval()
    std = out_norm.std
    res = []
    for i in range(0, len(u), batch_size):
        ub = u[i:i + batch_size]
        tb = t.unsqueeze(0).repeat(len(ub), 1).requires_grad_(True)       # (B, n_t)
        z = model(ub, tb)                                                  # (B, n_t, 4) normalised
        dq = []
        for j in range(2):
            (g,) = torch.autograd.grad(z[..., j].sum(), tb, retain_graph=True)
            dq.append(g * std[j] / t_final)                                # (B, n_t)
        dq = torch.stack(dq, dim=-1)
        qdot_out = z[..., 2:4].detach() * std[2:4] + out_norm.mean[2:4]
        res.append((dq.detach() - qdot_out).numpy())
    return np.concatenate(res).astype(np.float64)


def energy_balance_residual(traj, params, torque, t_grid, base: ArmParams = ArmParams()) -> np.ndarray:
    """How badly a trajectory breaks  E(t) - E(0) = integral_0^t (tau . qdot - b |qdot|^2) dt.  (N, n_t), J.

    Raw energy is not conserved (torque adds energy, friction removes it), so what a trajectory
    must satisfy is the BALANCE dE/dt = tau . qdot - b |qdot|^2. E comes from ``dynamics.energy``,
    the integral is a cumulative trapezoid on the output grid, using the trajectory's own qdot.

    Call it on the TRUE trajectories too: that residual is the floor (RK4 error plus the trapezoid
    rule on a 0.02 s grid). Only a prediction's residual ABOVE that floor is the network's fault.

    traj: (N, n_t, 4) raw; params: (N, 2) [m_p, b]; torque: (N, 2, n_t); t_grid: (n_t,).
    """
    traj, torque = np.asarray(traj, dtype=np.float64), np.asarray(torque, dtype=np.float64)
    out = np.empty(traj.shape[:2])
    dt = np.diff(t_grid)
    for n in range(len(traj)):
        p = replace(base, m_p=float(params[n, 0]), b=float(params[n, 1]))
        E = np.array([energy(x, p) for x in traj[n]])
        qd = traj[n, :, 2:4]
        power = (torque[n].T * qd).sum(axis=1) - p.b * (qd ** 2).sum(axis=1)   # (n_t,)
        work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * dt)])
        out[n] = (E - E[0]) - work
    return out


def energy_range(traj, params, base: ArmParams = ArmParams()) -> np.ndarray:
    """max_t E - min_t E for each trajectory, in J. Shape (N,). The yardstick for the balance residual."""
    traj = np.asarray(traj, dtype=np.float64)
    out = np.empty(len(traj))
    for n in range(len(traj)):
        p = replace(base, m_p=float(params[n, 0]), b=float(params[n, 1]))
        E = np.array([energy(x, p) for x in traj[n]])
        out[n] = E.max() - E.min()
    return out


def energy_balance_summary(pred, true, params, torque, t_grid) -> dict:
    """Per-sample energy-balance numbers, three ways (which to headline is the student's decision).

    All are per-sample (N,) arrays built from the worst-in-time |residual|:
      pred_J / true_J : raw max_t |residual| of the prediction / of the true trajectory (J)
      pred_rel        : pred_J / energy range of the TRUE trajectory (fraction of the energy swing)
      true_rel        : true_J / energy range (the floor: RK4 + trapezoid error on the grid)
      excess_J        : max(pred_J - true_J, 0), what is left after removing the floor
      excess_rel      : excess_J / energy range
    """
    pred_J = np.abs(energy_balance_residual(pred, params, torque, t_grid)).max(axis=1)
    true_J = np.abs(energy_balance_residual(true, params, torque, t_grid)).max(axis=1)
    rng = np.maximum(energy_range(true, params), 1e-12)
    excess = np.maximum(pred_J - true_J, 0.0)
    return {"pred_J": pred_J, "true_J": true_J, "pred_rel": pred_J / rng, "true_rel": true_J / rng,
            "excess_J": excess, "excess_rel": excess / rng}
