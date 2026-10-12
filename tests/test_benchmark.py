"""Tests for the benchmark helpers: the vectorised reference must agree with the scalar code."""
from dataclasses import replace

import numpy as np
import pytest

from src.benchmark import (
    batched_rhs,
    break_even_queries,
    deeponet_flops,
    mlp_flops,
    rk4_batched,
    solve_ivp_one,
)
from src.data import N_T, T_GRID, X0, sample_inputs, simulate, torque_fn
from src.dynamics import ArmParams, state_derivative


@pytest.fixture(scope="module")
def inputs():
    return sample_inputs(4, np.random.default_rng(0))


def test_batched_rhs_matches_scalar(inputs):
    params, coef = inputs
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, (4, 4))
    t = 0.37
    got = batched_rhs(t, X, params[:, 0], params[:, 1], coef)
    for i in range(4):
        p = replace(ArmParams(), m_p=params[i, 0], b=params[i, 1])
        want = state_derivative(t, X[i], p, torque_fn(coef[i]))
        assert np.allclose(got[i], want, rtol=1e-10, atol=1e-10)


def test_rk4_batched_matches_scalar_rk4(inputs):
    params, coef = inputs
    out = rk4_batched(params, coef, T_GRID, X0)
    assert out.shape == (4, N_T, 4)
    for i in range(4):
        assert np.allclose(out[i], simulate(params[i], coef[i]), rtol=1e-8, atol=1e-8)


def test_solve_ivp_agrees_with_rk4_on_a_mild_sample():
    params, coef = np.array([0.3, 0.2]), np.array([[1.0, 0.5, 0.0], [0.5, 0.7, 1.0]])
    ref = simulate(params, coef, substeps=16)
    got = solve_ivp_one(params, coef, T_GRID, X0, rtol=1e-9, atol=1e-12)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6


def test_flop_counts():
    assert mlp_flops(3, 4, 1, 2) == 2 * (3 * 4 + 4 * 2)
    # trunk cost grows with n_t, branch cost does not
    assert deeponet_flops(10, 100, 8, 16, 2, 2) > deeponet_flops(10, 10, 8, 16, 2, 2)


def test_break_even():
    assert break_even_queries(100, 100, 0.1, 0.0) == pytest.approx(2000)
    assert break_even_queries(100, 100, 0.1, 0.2) == float("inf")
