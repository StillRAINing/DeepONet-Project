"""Tests for the adaptive Dormand-Prince RK45 (the RK4 tests are yours: see tests/test_solvers.py)."""
import numpy as np
import pytest

from src.data import T_GRID, X0, simulate, torque_fn
from src.dynamics import ArmParams, state_derivative
from src.solvers import rk45_integrate, rk45_step


def test_exponential_decay_matches_exact_to_tolerance():
    t = np.linspace(0.0, 2.0, 21)
    out = rk45_integrate(lambda s, y: -y, np.array([1.0]), t, rtol=1e-9, atol=1e-12)
    assert np.allclose(out[:, 0], np.exp(-t), rtol=1e-7, atol=1e-9)


def test_tighter_tolerance_gives_smaller_error():
    # oscillator with a coarse output grid, so the step size is set by the tolerance, not by the grid
    w = 5.0
    f = lambda s, y: np.array([y[1], -w**2 * y[0]])
    t = np.linspace(0.0, 5.0, 11)
    exact = np.cos(w * t)
    errs = [np.max(np.abs(rk45_integrate(f, np.array([1.0, 0.0]), t, rtol=r, atol=r * 1e-3)[:, 0] - exact))
            for r in (1e-3, 1e-6, 1e-9)]
    assert errs[0] > errs[1] > errs[2]


def test_embedded_error_estimate_scales_like_h5():
    # local error of the 4th-order companion is O(h^5): halving h should cut the estimate by about 32
    f = lambda s, y: np.array([np.cos(s)]) * y + 1.0
    e = [np.abs(rk45_step(f, 0.3, np.array([1.0]), h)[1][0]) for h in (0.2, 0.1)]
    assert 16 < e[0] / e[1] < 64


def test_arm_agrees_with_rk4_on_a_mild_sample():
    params, coef = np.array([0.3, 0.2]), np.array([[1.0, 0.5, 0.0], [0.5, 0.7, 1.0]])
    p = ArmParams(m_p=0.3, b=0.2)
    tau = torque_fn(coef)
    got = rk45_integrate(lambda t, x: state_derivative(t, x, p, tau), X0, T_GRID, rtol=1e-9, atol=1e-12)
    ref = simulate(params, coef, substeps=16)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6


@pytest.mark.parametrize("n", [1, 3])
def test_output_shape_and_start(n):
    out = rk45_integrate(lambda s, y: -y, np.ones(n), np.linspace(0, 1, 5))
    assert out.shape == (5, n) and np.all(out[0] == 1.0)
