"""Solver tests: SKELETON for you to fill in (see docs/STUDY_GUIDE.md section 2).

Every test below is skipped until you replace the body. Each docstring says what to check, not how.
Delete the ``@todo`` decorator from a test once you have written and run it. Suggested order is top to bottom.
"""
import numpy as np  # noqa: F401
import pytest

from src.solvers import rk4_integrate, rk4_step  # noqa: F401

todo = pytest.mark.skip(reason="TODO (student): write this test")


@todo
def test_rk4_step_exact_for_cubic_polynomial_rhs():
    """y' = t**3 has an exact quartic solution. Why must one RK4 step reproduce it exactly (to round-off)?"""


@todo
def test_rk4_exponential_decay_against_exact_solution():
    """y' = -y, y(0) = 1. Compare rk4_integrate with exp(-t) on a fine grid; pick a tolerance you can justify."""


@todo
def test_rk4_global_error_order_is_four():
    """Halve h repeatedly on y' = -y. Estimate p = log2(err(h)/err(h/2)). Assert p is close to 4.
    Decide what 'close' means and why the asymptotic regime matters."""


@todo
def test_rk4_order_on_the_arm_against_a_fine_reference():
    """Same order check on the real arm (a mild sample), with a much finer RK4 or solve_ivp(rtol=1e-12) as reference.
    Why must the reference be far better than your finest run?"""


@todo
def test_rk4_agrees_with_solve_ivp_on_mild_arm_sample():
    """Compare against scipy.integrate.solve_ivp (tight tolerance) for a low-torque sample. Then try a spinning sample
    and explain the difference (docs/REVIEW_NOTES.md 2.2)."""


@todo
def test_rk4_output_grid_and_initial_condition():
    """Output shape (len(t_grid), n); first row equals x0; x0 is not modified in place."""
