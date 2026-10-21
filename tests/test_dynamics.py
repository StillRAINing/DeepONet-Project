"""Tests for src.dynamics. They fail until the functions are implemented (that is the point)."""
import numpy as np
import pytest

from src.dynamics import (
    ArmParams,
    combine_payload,
    coriolis_vector,
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


# --- Tests the mutation audit showed were missing (docs/REVIEW_NOTES.md section 1). ---
# Written by the AI on request (AI_LOG row to add). Each uses a reference that does NOT go through
# combine_payload / mass_matrix, so a wrong formula there cannot cancel itself out.


def _brute_force_energy(x, p: ArmParams):
    """T and V of link 1, link 2 and the point payload as three separate bodies, from geometry only."""
    q1, q2, q1d, q2d = x
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    w1, w12 = q1d, q1d + q2d
    # positions and velocities (d/dt of the positions, by the chain rule)
    pos = {
        "c1": (p.lc1 * c1, p.lc1 * s1),
        "c2": (p.l1 * c1 + p.lc2 * c12, p.l1 * s1 + p.lc2 * s12),
        "tip": (p.l1 * c1 + p.l2 * c12, p.l1 * s1 + p.l2 * s12),
    }
    vel = {
        "c1": (-p.lc1 * s1 * w1, p.lc1 * c1 * w1),
        "c2": (-p.l1 * s1 * w1 - p.lc2 * s12 * w12, p.l1 * c1 * w1 + p.lc2 * c12 * w12),
        "tip": (-p.l1 * s1 * w1 - p.l2 * s12 * w12, p.l1 * c1 * w1 + p.l2 * c12 * w12),
    }
    mass = {"c1": p.m1, "c2": p.m2, "tip": p.m_p}
    T = sum(0.5 * mass[k] * (vel[k][0] ** 2 + vel[k][1] ** 2) for k in mass)
    T += 0.5 * p.I1 * w1**2 + 0.5 * p.I2 * w12**2  # rotation of each link about its own COM
    V = sum(mass[k] * p.g * pos[k][1] for k in mass)
    return T, V


@pytest.mark.parametrize("m_p", [0.3, 0.5, 0.7, 1.3])
def test_payload_inertia_matches_brute_force_at_mid_range_payload(m_p):
    """0.5 qdot^T M qdot and energy() equal the three-body kinetic + potential energy at random states.
    Kills the 'drop a parallel-axis term' mutations (they only show up for 0 < m_p < infinity)."""
    rng = np.random.default_rng(1)
    p = ArmParams(m_p=m_p)
    for _ in range(50):
        x = np.concatenate([rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 2)])
        T_ref, V_ref = _brute_force_energy(x, p)
        T = 0.5 * x[2:] @ mass_matrix(x[:2], p) @ x[2:]
        assert T == pytest.approx(T_ref, rel=1e-12, abs=1e-12)
        assert energy(x, p) == pytest.approx(T_ref + V_ref, rel=1e-12, abs=1e-12)


def _mdot(q, qdot, p, eps=1e-6):
    """dM/dt = sum_i dM/dq_i * qdot_i, by a central difference along the direction qdot."""
    return (mass_matrix(q + eps * qdot, p) - mass_matrix(q - eps * qdot, p)) / (2 * eps)


def _dT_dq(q, qdot, p, eps=1e-6):
    """Partial derivative of T = 0.5 qdot^T M(q) qdot w.r.t. q at fixed qdot (central differences)."""
    T = lambda qq: 0.5 * qdot @ mass_matrix(qq, p) @ qdot
    return np.array([(T(q + eps * e) - T(q - eps * e)) / (2 * eps) for e in np.eye(2)])


def test_coriolis_skew_symmetry_identity():
    """qdot^T (Mdot - 2 C) qdot = 0, in the power form qdot . (C qdot) = 0.5 qdot^T Mdot qdot.

    Plus the stronger, full check from Euler-Lagrange: C qdot = Mdot qdot - dT/dq. The power identity
    only sees the component of C qdot along qdot, so a wrong term perpendicular to qdot would survive it.
    """
    rng = np.random.default_rng(2)
    p = ArmParams(m_p=0.6)
    for _ in range(50):
        q, qdot = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 2)
        Cq = coriolis_vector(q, qdot, p)
        Md = _mdot(q, qdot, p)
        assert qdot @ Cq == pytest.approx(0.5 * qdot @ Md @ qdot, rel=1e-6, abs=1e-8)
        np.testing.assert_allclose(Cq, Md @ qdot - _dT_dq(q, qdot, p), rtol=1e-6, atol=1e-8)


def test_gravity_is_gradient_of_potential():
    """G(q) = dV/dq, with V from the brute-force geometry (finite differences)."""
    rng = np.random.default_rng(3)
    p = ArmParams(m_p=0.4)
    eps = 1e-6
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 2)
        V = lambda qq: _brute_force_energy(np.concatenate([qq, [0.0, 0.0]]), p)[1]
        dV = np.array([(V(q + eps * e) - V(q - eps * e)) / (2 * eps) for e in np.eye(2)])
        np.testing.assert_allclose(gravity_vector(q, p), dV, rtol=1e-6, atol=1e-8)


def test_energy_balance_with_torque_and_friction():
    """dE/dt = tau . qdot - b |qdot|^2 along an RK4 trajectory (central-difference dE/dt).

    Central difference error is O(h^2) * |E'''|; with h = 1e-3 that is far below 1e-3 of the power scale.
    """
    p = ArmParams(b=0.3, m_p=0.5)
    tau_fn = lambda t: np.array([4.0 * np.sin(2.0 * t), -2.0 * np.cos(3.0 * t + 0.4)])
    f = lambda t, x: state_derivative(t, x, p, tau_fn)
    h = 1e-3
    t = np.arange(0.0, 3.0 + h / 2, h)
    X = rk4_integrate(f, np.zeros(4), t)
    E = np.array([energy(x, p) for x in X])
    dEdt = (E[2:] - E[:-2]) / (2 * h)
    qd = X[1:-1, 2:]
    power = np.array([tau_fn(s) @ v for s, v in zip(t[1:-1], qd)]) - p.b * np.sum(qd**2, axis=1)
    assert np.abs(power).max() > 1.0  # not a trivial trajectory
    assert np.max(np.abs(dEdt - power)) < 1e-3 * np.abs(power).max()
    # with the friction sign flipped (energy injected) the balance must fail clearly
    assert np.max(np.abs(dEdt - (power + 2 * p.b * np.sum(qd**2, axis=1)))) > 1e-2 * np.abs(power).max()


def test_small_oscillations_about_hanging_equilibrium_match_linear_theory():
    """Pendulum limit, done on the whole arm. About q* = (-pi/2, 0), linearise: M(q*) d'' + K d = 0, with

        K = d^2 V / dq^2 at q*  (hand-derived, V = g [m1 lc1 s1 + m2e (l1 s1 + lc2e s12)]):
        K11 = g (m1 lc1 + m2e l1 + m2e lc2e),  K12 = K21 = K22 = g m2e lc2e.

    Each generalised eigenvector v of K v = w^2 M v is a normal mode: start at q* + eps v at rest and
    the arm must swing as q* + eps v cos(w t). Errors are O(eps^2) (nonlinearity) + RK4 error.
    Zero torque on joint 2 is NOT a lock, so the arm is not a single pendulum; the single-pendulum
    formula is what you get when link 2 is very short and light (lower mode -> sqrt(g m l / I_pivot)).
    """
    p = ArmParams(b=0.0, m_p=0.5)
    m2e = p.m2 + p.m_p
    lc2e = (p.m2 * p.lc2 + p.m_p * p.l2) / m2e
    K = p.g * np.array([[p.m1 * p.lc1 + m2e * p.l1 + m2e * lc2e, m2e * lc2e],
                        [m2e * lc2e, m2e * lc2e]])
    q_star = np.array([-np.pi / 2, 0.0])
    M = mass_matrix(q_star, p)
    w2, V = np.linalg.eig(np.linalg.solve(M, K))
    assert np.all(w2.real > 0)  # hanging down is a stable equilibrium
    eps = 1e-3
    f = lambda t, x: state_derivative(t, x, p, lambda s: np.zeros(2))
    for w, v in zip(np.sqrt(w2.real), V.real.T):
        v = v / np.abs(v).max()
        t = np.linspace(0.0, 4 * 2 * np.pi / w, 2001)  # four periods
        X = rk4_integrate(f, np.concatenate([q_star + eps * v, [0.0, 0.0]]), t)
        expected = q_star + eps * np.outer(np.cos(w * t), v)
        assert np.max(np.abs(X[:, :2] - expected)) < 1e-2 * eps
