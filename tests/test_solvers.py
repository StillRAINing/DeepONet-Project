"""RK4 solver tests (see docs/STUDY_GUIDE.md section 2).

Written by the AI on request (AI_LOG row to add); the questions from the original skeleton are kept
in each docstring with a short answer, so the reasoning can be checked and defended.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.data import T_FINAL, T_GRID, X0, simulate, torque_fn
from src.dynamics import ArmParams, state_derivative
from src.solvers import rk4_integrate, rk4_step

MILD = (np.array([0.3, 0.2]), np.array([[2.0, 0.5, 0.0], [1.0, 0.7, 1.0]]))
SPINNING = (np.array([0.0, 0.05]), np.array([[15.0, 1.0, 0.0], [15.0, 1.0, np.pi / 2]]))


def _arm_rhs(params, coef):
    p = ArmParams(m_p=float(params[0]), b=float(params[1]))
    tau = torque_fn(coef)
    return lambda t, x: state_derivative(t, x, p, tau)


def _reference(f, t_eval):
    """solve_ivp with DOP853 at tolerances far below any RK4 run in this file."""
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), X0, method="DOP853", t_eval=t_eval, rtol=1e-12, atol=1e-12)
    assert sol.success
    return sol.y.T


def _observed_orders(errors):
    """p_i = log2(err(h) / err(h/2)) for consecutive halvings."""
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def test_rk4_step_exact_for_cubic_polynomial_rhs():
    """y' = t**3 has an exact quartic solution. Why must one RK4 step reproduce it exactly (to round-off)?

    When f depends on t only, an RK4 step is Simpson's rule on [t, t+h] (weights 1/6, 4/6, 1/6, the two
    midpoint stages coincide). Simpson's rule is exact for polynomials up to degree 3, so the step is
    exact for any cubic RHS, even with a large h.
    """
    f = lambda t, y: np.array([t**3 - 2 * t**2 + 0.5 * t + 1.0])
    exact = lambda t: t**4 / 4 - 2 * t**3 / 3 + t**2 / 4 + t
    for t0, h in [(0.0, 1.0), (0.7, 2.5), (-1.0, 0.3)]:
        y1 = rk4_step(f, t0, np.array([exact(t0)]), h)
        assert y1[0] == pytest.approx(exact(t0 + h), abs=1e-12)
    # degree 4 is NOT integrated exactly: this is what makes the test sharp
    f4 = lambda t, y: np.array([t**4])
    assert abs(rk4_step(f4, 0.0, np.array([0.0]), 1.0)[0] - 0.2) > 1e-3


def test_rk4_exponential_decay_against_exact_solution():
    """y' = -y, y(0) = 1. Compare rk4_integrate with exp(-t) on a fine grid; pick a tolerance you can justify.

    Global error is about C h^4 with C ~ t e^{-t} / 120 (the local error constant of RK4 on y' = -y is 1/120),
    so for h = 0.01 the error is ~3e-11 at most. A tolerance of 1e-9 leaves room for round-off but would
    catch an order drop to O(h^3) (~1e-6).
    """
    t = np.linspace(0.0, 5.0, 501)
    y = rk4_integrate(lambda s, y: -y, np.array([1.0]), t)[:, 0]
    assert np.max(np.abs(y - np.exp(-t))) < 1e-9


def test_rk4_global_error_order_is_four():
    """Halve h repeatedly on y' = -y. Estimate p = log2(err(h)/err(h/2)). Assert p is close to 4.

    'Close' = within [3.8, 4.2]. The h^5 term contributes a relative correction of order h, so the
    estimate approaches 4 only in the asymptotic regime (h small enough that h^4 dominates) but before
    round-off (~1e-16 per step) takes over. h from 0.1 to 0.0125 sits in that window: errors run
    from ~1e-7 down to ~1e-11.
    """
    errs = []
    for n in (20, 40, 80, 160):
        t = np.linspace(0.0, 2.0, n + 1)
        errs.append(abs(rk4_integrate(lambda s, y: -y, np.array([1.0]), t)[-1, 0] - np.exp(-2.0)))
    p = _observed_orders(errs)
    assert np.all((p > 3.8) & (p < 4.2)), p


def test_rk4_order_on_the_arm_against_a_fine_reference():
    """Same order check on the real arm (a mild sample), with solve_ivp(rtol=1e-12) as reference.

    The reference must be far better than the finest RK4 run: the measured error is
    |rk4 - ref| <= |rk4 - exact| + |ref - exact|. If the reference error is comparable to the finest
    RK4 error, the last halving cannot drop by 16 and the estimated order falls below 4.
    """
    f = _arm_rhs(*MILD)
    errs = []
    for n in (75, 150, 300, 600):  # h = 0.04 .. 0.005 s; the dataset uses h = 3 / 1200 = 0.0025 s
        t = np.linspace(0.0, T_FINAL, n + 1)
        ref = _reference(f, t)
        errs.append(np.max(np.abs(rk4_integrate(f, X0, t) - ref)))
    p = _observed_orders(errs)
    assert errs[-1] > 1e-10, "finest RK4 error too close to the reference error to measure an order"
    assert np.all((p > 3.6) & (p < 4.4)), (errs, p)


def test_rk4_agrees_with_solve_ivp_on_mild_arm_sample():
    """Dataset RK4 (8 substeps, h = 2.5 ms) vs solve_ivp(DOP853, rtol = atol = 1e-12) on the 151-point grid.

    Measured (2026-10-15): mild sample (|qdot| <= 12 rad/s) 1.6e-8, far below any surrogate error.
    Spinning sample (A = 15 N m, b = 0.05, |qdot| up to ~120 rad/s, q up to ~34 rad) 2.4e-3: the
    fixed h = 2.5 ms is coarse relative to the motion, and the truth error is no longer negligible
    next to a ~1e-2 surrogate error. docs/REVIEW_NOTES.md 2.2. The smooth-family dataset (peak 6 N m)
    is checked sample by sample in scripts/sanity_checks.py.
    """
    def rel_err(params, coef):
        got = simulate(params, coef)  # the dataset's own ground-truth routine
        ref = _reference(_arm_rhs(params, coef), T_GRID)
        return np.linalg.norm(got - ref) / np.linalg.norm(ref)

    e_mild, e_spin = rel_err(*MILD), rel_err(*SPINNING)
    assert e_mild < 1e-7
    assert e_spin < 1e-2
    assert e_spin > e_mild  # the harder trajectory is harder for RK4 too


def test_rk4_output_grid_and_initial_condition():
    """Output shape (len(t_grid), n); first row equals x0; x0 is not modified in place."""
    x0 = np.array([0.3, -0.2, 1.0])
    keep = x0.copy()
    t = np.linspace(0.0, 1.0, 11)
    out = rk4_integrate(lambda s, y: -y, x0, t)
    assert out.shape == (11, 3)
    assert np.array_equal(out[0], keep)
    assert np.array_equal(x0, keep)
    # integer x0 must not truncate the solution to integers
    assert rk4_integrate(lambda s, y: -y, np.array([1]), t).dtype == np.float64
