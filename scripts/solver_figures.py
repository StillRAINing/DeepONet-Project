"""Ground-truth verification figures (FIGURE_PLAN #2, #3, #4) and their numbers.

    py -3.10 -m scripts.solver_figures          # about 1 minute

Uses the project's own ``solvers.rk4_integrate`` / ``data.simulate_smooth`` (the code that made the
labels) and SciPy DOP853 at rtol = atol = 1e-12 as the reference. No model is involved, so using test /
OOD inputs here does not touch model selection. Writes:

  results/solver_convergence.png  (#2) max error vs step h, log-log, on y' = -y and on two arm samples
  results/solver_energy.png       (#3) energy drift with b = 0, tau = 0; energy BALANCE with torque + friction
  results/solver_vs_dop853.png    (#4) stored-label scheme vs DOP853 over time, calm and spinning sample
  results/solver_checks.md        the numbers behind the three figures

Plotting and measurement only (AI-written, assistant mode). The RK4 code itself is the student's.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from src.data import DATA_PATH_SMOOTH, T_FINAL, T_GRID, X0, load, simulate_smooth, smooth_torque_fn
from src.dynamics import ArmParams, energy, state_derivative
from src.evaluate import energy_balance_residual, rel_l2_per_sample
from src.solvers import rk4_integrate

RESULTS = Path(__file__).resolve().parent.parent / "results"
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
H_DATA = T_FINAL / ((len(T_GRID) - 1) * 8)       # 0.0025 s, the step that made the labels


def arm_rhs(params, coef):
    p = replace(ArmParams(), m_p=float(params[0]), b=float(params[1]))
    tau = smooth_torque_fn(coef)
    return (lambda t, x: state_derivative(t, x, p, tau)), p, tau


def reference(f, t_eval, x0=X0):
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), x0, method="DOP853", t_eval=t_eval, rtol=1e-12, atol=1e-12)
    assert sol.success, sol.message
    return sol.y.T


def orders(e):
    e = np.asarray(e)
    return np.log2(e[:-1] / e[1:])


def convergence(d, lines):
    """#2: error of RK4 at t on the coarse output grid vs h, for a decay ODE and two arm trajectories."""
    peak = np.abs(d["test_traj"][..., :2]).max(axis=(1, 2))
    calm = int(np.argsort(peak)[len(peak) // 2])       # median-violence test sample
    wild = int(np.argmax(peak))                         # the most violent test sample
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ns = 10 * 2 ** np.arange(0, 8)                      # 10 .. 1280 steps on [0, 2]
    hs = 2.0 / ns
    err = [abs(rk4_integrate(lambda t, y: -y, np.array([1.0]), np.linspace(0, 2, n + 1))[-1, 0] - np.exp(-2)) for n in ns]
    ax[0].loglog(hs, err, "o-", color=C[0], lw=2, label="RK4 on y' = -y (error at t = 2)")
    ax[0].loglog(hs, err[0] * (hs / hs[0]) ** 4, ":", color="#555555", label="slope 4 reference")
    ax[0].set_xlabel("step h")
    ax[0].set_ylabel("|y_RK4(2) - exp(-2)|")
    ax[0].set_title("Scalar test problem")
    p0 = orders(err)
    lines += ["## #2 Convergence order", "",
              f"y' = -y on [0, 2], error at t = 2, h = {hs[0]:.3g} .. {hs[-1]:.3g}: observed orders "
              + ", ".join(f"{p:.2f}" for p in p0) + " (round-off takes over when the error nears 1e-15).", ""]

    arm_n = 75 * 2 ** np.arange(0, 8)                   # h = 0.04 .. 0.0003125 s
    arm_h = T_FINAL / arm_n
    for k, (name, i) in enumerate((("calm test sample", calm), ("most violent test sample", wild))):
        f, _, _ = arm_rhs(d["test_params"][i], d["test_tau_coef"][i])
        e = []
        for n in arm_n:
            t = np.linspace(0, T_FINAL, n + 1)
            stride = n // 75
            e.append(np.abs(rk4_integrate(f, X0, t)[::stride] - reference(f, t[::stride])).max())
        ax[1].loglog(arm_h, e, "o-", color=C[k + 1], lw=2,
                     label=f"{name} #{i} (peak |q| {peak[i]:.1f} rad)")
        p = orders(e)
        lines.append(f"Arm, {name} (test #{i}, peak |q| {peak[i]:.1f} rad), max abs state error over 76 times "
                     f"vs DOP853, h = {arm_h[0]:.3g} .. {arm_h[-1]:.3g} s: errors "
                     + ", ".join(f"{x:.1e}" for x in e) + "; observed orders " + ", ".join(f"{x:.2f}" for x in p) + ".")
    ax[1].loglog(arm_h, e[2] * (arm_h / arm_h[2]) ** 4, ":", color="#555555", label="slope 4 reference")
    ax[1].axvline(H_DATA, color="#888888", lw=1, ls="--")
    ax[1].text(H_DATA, ax[1].get_ylim()[1], " dataset h = 2.5 ms", va="top", fontsize=8, color="#555555")
    ax[1].set_xlabel("step h [s]")
    ax[1].set_ylabel("max |x_RK4 - x_DOP853| over the trajectory")
    ax[1].set_title("Arm (smooth torque family), reference DOP853 rtol 1e-12")
    for a in ax:
        a.grid(alpha=0.25, which="both")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("#2  RK4 global error: slope 4 (16x per halving) once h is in the asymptotic range")
    fig.tight_layout()
    fig.savefig(RESULTS / "solver_convergence.png", dpi=150)
    plt.close(fig)
    lines += ["", "Reading: the calm sample settles to order 4 as h shrinks (3.8 to 4.0 at the smallest steps). The violent "
              "sample is not in the asymptotic regime for h >= 10 ms (orders 0.5 and 1.3: the trajectory is wrong, not just "
              "inaccurate), then shows orders of about 5 between 10 ms and 0.3 ms: faster than 4 because the leading h^4 "
              "term is still being overtaken by fast-shrinking higher-order terms. 4 is the asymptotic order; the test in "
              "`tests/test_solvers.py` therefore uses the calm sample. At the dataset step (2.5 ms) the violent sample's "
              "error is about 1e-2, the calm one's about 1e-6 (max over states and times).", ""]


def energy_checks(d, lines):
    """#3: (a) conservation with b = 0, tau = 0 for several h; (b) energy balance with torque + friction."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    p = replace(ArmParams(), m_p=0.5, b=0.0)
    f = lambda t, x: state_derivative(t, x, p, lambda s: np.zeros(2))
    T_LONG = 10.0
    lines += ["## #3 Energy", "", f"(a) b = 0, tau = 0, m_p = 0.5 kg, released from x0 = 0 (horizontal), {T_LONG:.0f} s:", ""]
    drift = {}
    for k, h in enumerate((0.02, 0.01, 0.005, H_DATA)):
        t = np.linspace(0, T_LONG, int(round(T_LONG / h)) + 1)
        X = rk4_integrate(f, X0, t)
        E = np.array([energy(x, p) for x in X])
        dE = np.abs(E - E[0])
        drift[h] = dE.max()
        ax[0].semilogy(t[1:], np.maximum(dE[1:], 1e-16), color=C[k], lw=1.5, label=f"h = {h * 1e3:g} ms (max {dE.max():.1e} J)")
    hs = sorted(drift, reverse=True)
    lines.append("| h (ms) | max \\|E(t) - E(0)\\| (J) | ratio to previous h |")
    lines.append("|---|---|---|")
    for a, b in zip([None] + hs[:-1], hs):
        lines.append(f"| {b * 1e3:g} | {drift[b]:.2e} | {'' if a is None else f'{drift[a] / drift[b]:.1f} (h ratio {a / b:g})'} |")
    # yardstick: the largest kinetic energy reached (E - potential at the same angles), finest run
    E_scale = max(energy(x, p) - energy(np.r_[x[:2], 0.0, 0.0], p) for x in X)
    lines += ["", f"Largest kinetic energy in this motion: {E_scale:.1f} J, so the drift at the dataset step is a "
              f"relative {drift[H_DATA] / E_scale:.0e}. RK4 is not symplectic, so the drift is small but not zero. Per halving of h "
              "it falls by " + ", ".join(f"{drift[a] / drift[b]:.0f}x" for a, b in zip(hs[:-1], hs[1:]))
              + ": at least the 16x of a 4th-order method, but not a clean 16 (the drift is the maximum of an oscillating "
              "error, so cancellation makes the ratio uneven).", ""]
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("|E(t) - E(0)| [J]")
    ax[0].set_title("(a) conservation: b = 0, tau = 0 (energy must stay constant)")

    # (b) energy balance on a real test sample: E(t) - E(0) = W(t) = int tau.qdot - b |qdot|^2 dt
    i = int(np.argsort(np.abs(d["test_traj"][..., :2]).max(axis=(1, 2)))[len(d["test_traj"]) // 2])
    f, pb, tau = arm_rhs(d["test_params"][i], d["test_tau_coef"][i])
    t = np.linspace(0, T_FINAL, int(round(T_FINAL / H_DATA)) + 1)
    X = rk4_integrate(f, X0, t)
    E = np.array([energy(x, pb) for x in X])
    tq = np.array([tau(s) for s in t])
    power = (tq * X[:, 2:]).sum(1) - pb.b * (X[:, 2:] ** 2).sum(1)
    W = np.concatenate([[0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    res_fine = (E - E[0]) - W
    res_grid = energy_balance_residual(d["test_traj"][i:i + 1], d["test_params"][i:i + 1],
                                       d["test_torque"][i:i + 1], T_GRID)[0]
    ax[1].plot(t, E - E[0], color=C[0], lw=2, label="E(t) - E(0)")
    ax[1].plot(t, W, color=C[1], lw=2, ls="--", label="work in: int (tau . qdot - b |qdot|^2) dt")
    ax2 = ax[1].twinx()
    ax2.semilogy(t[1:], np.maximum(np.abs(res_fine[1:]), 1e-16), color=C[2], lw=1, label="|residual|, 2.5 ms grid")
    ax2.semilogy(T_GRID[1:], np.maximum(np.abs(res_grid[1:]), 1e-16), color=C[3], lw=1, ls=":",
                 label="|residual|, 20 ms output grid (evaluation floor)")
    ax2.set_ylabel("|residual| [J] (log)")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("energy [J]")
    ax[1].set_title(f"(b) balance on test sample #{i} (b = {pb.b:.2f}, m_p = {pb.m_p:.2f})")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    for a in ax:
        a.grid(alpha=0.25, which="both")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("#3  Raw energy is conserved only without torque and friction; with them the BALANCE dE/dt = tau.qdot - b|qdot|^2 holds")
    fig.tight_layout()
    fig.savefig(RESULTS / "solver_energy.png", dpi=150)
    plt.close(fig)
    lines += [f"(b) test sample #{i}: energy swing {np.ptp(E):.2f} J; max |balance residual| {np.abs(res_fine).max():.1e} J "
              f"with the trapezoid rule on the 2.5 ms step, {np.abs(res_grid).max():.1e} J on the 20 ms output grid. The second "
              "number is the floor `evaluate.energy_balance_residual` subtracts before blaming a network (it is quadrature "
              "error, not RK4 error).", ""]


def label_accuracy(d, lines):
    """#4: the label-generating scheme (RK4, 8 substeps) against DOP853 over time, calm vs spinning."""
    peak_test = np.abs(d["test_traj"][..., :2]).max(axis=(1, 2))
    peak_amp2 = np.abs(d["ood_amp2_traj"][..., :2]).max(axis=(1, 2))
    cases = [("calm test sample", "test", int(np.argsort(peak_test)[len(peak_test) // 2])),
             ("most violent test sample", "test", int(np.argmax(peak_test))),
             ("most violent ood_amp2 sample", "ood_amp2", int(np.argmax(peak_amp2)))]
    fig, ax = plt.subplots(1, len(cases), figsize=(15, 4.2), sharey=True)
    std = d["train_traj"].reshape(-1, 4).std(0)
    lines += ["## #4 Stored labels vs DOP853 (rtol 1e-12)", "",
              "| sample | peak \\|q\\| (rad) | headline rel. L2 of the label | max abs error q1, q2, q1dot, q2dot |", "|---|---|---|---|"]
    for a, (name, split, i) in zip(ax, cases):
        f, _, _ = arm_rhs(d[f"{split}_params"][i], d[f"{split}_tau_coef"][i])
        lab = simulate_smooth(d[f"{split}_params"][i], d[f"{split}_tau_coef"][i])
        assert np.allclose(lab, d[f"{split}_traj"][i], atol=1e-5), "simulate_smooth no longer reproduces the stored label"
        ref = reference(f, T_GRID)
        err = np.abs(lab - ref)
        for j, ch in enumerate(("q1", "q2", "q1dot", "q2dot")):
            a.semilogy(T_GRID[1:], np.maximum(err[1:, j], 1e-16), color=C[j], lw=1.5, label=ch)
        rel = rel_l2_per_sample(lab[None], ref[None], std)[0]
        pk = np.abs(ref[:, :2]).max()
        a.set_title(f"{name} #{i}\npeak |q| {pk:.1f} rad, headline rel. L2 {rel:.1e}", fontsize=10)
        a.set_xlabel("t [s]")
        a.grid(alpha=0.25, which="both")
        lines.append(f"| {split} #{i} ({name}) | {pk:.1f} | {rel:.1e} | " + ", ".join(f"{x:.1e}" for x in err.max(0)) + " |")
    ax[0].set_ylabel("|label - DOP853| (rad or rad/s)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("#4  Label error (RK4, h = 2.5 ms) vs a tight DOP853 reference: tiny when calm, grows on spinning trajectories, "
                 "below every surrogate error on the same set")
    fig.tight_layout()
    fig.savefig(RESULTS / "solver_vs_dop853.png", dpi=150)
    plt.close(fig)
    lines += ["", "In range (test) the label error is 1e-8 to 5e-5, three or more orders of magnitude below the surrogates' "
              "median error (0.07 to 0.13). The worst trajectory of the most extreme OOD set (ood_amp2, peak 122 rad) has a "
              "label error of 2.4e-2: no longer negligible in absolute terms, but still about 30x below the surrogates' error "
              "on that set (median about 0.8). So the label floor explains none of the reported surrogate errors. "
              "Sample-wide statistics: `EXPERIMENTS.md` 'Truth check'.", ""]


def main():
    d = load(DATA_PATH_SMOOTH)
    lines = ["# Ground-truth solver checks (FIGURE_PLAN #2 to #4)", "",
             "Generated by `scripts/solver_figures.py`. Reference: SciPy DOP853, rtol = atol = 1e-12. The RK4 under test is "
             "`src/solvers.rk4_integrate`, the routine that produced every label.", ""]
    convergence(d, lines)
    energy_checks(d, lines)
    label_accuracy(d, lines)
    (RESULTS / "solver_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
