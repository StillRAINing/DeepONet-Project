"""Tests for src.dynamics. They fail until the functions are implemented (that is the point)."""
import numpy as np
import pytest

from src.dynamics import (
    ArmParams,
    combine_payload,
    energy,
    gravity_vector,
    mass_matrix,
    state_derivative,
)
from src.solvers import rk4_integrate


def test_zero_payload_leaves_link2_unchanged():
    p = ArmParams()
    m, lc, I = combine_payload(p.m2, p.lc2, p.I2, p.l2, 0.0)
    assert m == pytest.approx(p.m2)
    assert lc == pytest.approx(p.lc2)
    assert I == pytest.approx(p.I2)


def test_huge_payload_moves_com_to_tip():
    p = ArmParams()
    m, lc, _ = combine_payload(p.m2, p.lc2, p.I2, p.l2, 1e9)
    assert lc == pytest.approx(p.l2, rel=1e-6)
    assert m == pytest.approx(1e9, rel=1e-6)


def test_mass_matrix_symmetric_positive_definite():
    rng = np.random.default_rng(0)
    p = ArmParams(m_p=0.7)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, size=2)
        M = mass_matrix(q, p)
        assert M.shape == (2, 2)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_hanging_straight_down_is_equilibrium():
    p = ArmParams(m_p=0.5)
    q = np.array([-np.pi / 2, 0.0])
    assert np.allclose(gravity_vector(q, p), 0.0, atol=1e-12)
    x = np.array([q[0], q[1], 0.0, 0.0])
    assert np.allclose(state_derivative(0.0, x, p, lambda t: np.zeros(2)), 0.0, atol=1e-12)


def test_energy_conserved_without_friction_or_torque():
    p = ArmParams(b=0.0, m_p=0.5)
    x0 = np.array([-np.pi / 2 + 0.8, 0.5, 0.0, 0.0])
    f = lambda t, x: state_derivative(t, x, p, lambda s: np.zeros(2))
    t = np.arange(0.0, 2.0 + 1e-9, 1e-3)
    X = rk4_integrate(f, x0, t)
    E = np.array([energy(x, p) for x in X])
    assert np.max(np.abs(E - E[0])) / abs(E[0] - energy(np.array([-np.pi/2, 0, 0, 0]), p)) < 1e-6
