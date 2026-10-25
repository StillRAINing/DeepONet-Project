"""2-link planar arm model (vertical plane, gravity on, viscous joint friction).

Conventions (keep these fixed everywhere in the project):
    * State x = [q1, q2, q1dot, q2dot]. Units: rad, rad, rad/s, rad/s.
    * q1 is measured from the horizontal +x axis, q2 is relative to link 1.
    * Gravity acts in the -y direction, g = 9.81 m/s^2.
    * Potential energy is zero at the base joint height.
    * Equation of motion: M(q) qddot + C(q, qdot) qdot + G(q) + b qdot = tau.

Payload m_p is a point mass at the tip of link 2 and is folded into link 2's
mass, COM distance and inertia (see ``combine_payload``).
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class ArmParams:
    """Physical parameters of the arm (SI units).

    Attributes:
        l1, l2: Link lengths (m).
        m1, m2: Link masses (kg), excluding payload.
        lc1, lc2: Distance from each joint to its link's COM (m).
        I1, I2: Link inertias about their own COM (kg m^2), excluding payload.
        g: Gravitational acceleration (m/s^2).
        m_p: Tip payload mass (kg). Varies per sample.
        b: Viscous joint friction coefficient (N m s/rad). Varies per sample.
    """

    l1: float = 0.5
    l2: float = 0.4
    m1: float = 2.0
    m2: float = 1.5
    lc1: float = 0.25
    lc2: float = 0.2
    I1: float = 2.0 * 0.5**2 / 12
    I2: float = 1.5 * 0.4**2 / 12
    g: float = 9.81
    m_p: float = 0.0
    b: float = 0.1


def combine_payload(
    m2: float, lc2: float, I2: float, l2: float, m_p: float
) -> Tuple[float, float, float]:
    """Combine link 2 with a point-mass payload at its tip.

    Args:
        m2: Link 2 mass (kg).
        lc2: Link 2 COM distance from joint 2 (m).
        I2: Link 2 inertia about its own COM (kg m^2).
        l2: Link 2 length (m); the payload sits at distance l2 from joint 2.
        m_p: Payload mass (kg).

    Returns:
        (m2_eff, lc2_eff, I2_eff): combined mass, COM distance from joint 2,
        and inertia about the combined COM (use the parallel-axis theorem).
    """

    m2_eff = m2 + m_p #treating payload as the tip of link 2

    # new COM of link 2 + payload (mass-weighted average of the two)
    lc2_eff = (m2*lc2 + m_p*l2) / (m2 + m_p)

    I2_eff = I2 + m2*(lc2_eff - lc2)**2 + m_p*(l2 - lc2_eff)**2 # Using the parrallel axis theorem


    return m2_eff, lc2_eff, I2_eff


def mass_matrix(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the 2x2 mass matrix M(q), including the payload.

    Args:
        q: Joint angles [q1, q2] (rad).
        p: Arm parameters.

    Returns:
        Symmetric positive definite 2x2 array (kg m^2).
    """
    q1, q2 = q
    m2e, lc2e, I2e = combine_payload(p.m2, p.lc2, p.I2, p.l2, p.m_p)
    c2 = np.cos(q2)  # M only depends on q2 (the elbow angle), not on q1

    # M11: whole arm about joint 1, M12: coupling between the joints, M22: link 2 about joint 2
    M11 = p.m1*p.lc1**2 + p.I1 + m2e*(p.l1**2 + lc2e**2 + 2*p.l1*lc2e*c2) + I2e
    M12 = m2e*(lc2e**2 + p.l1*lc2e*c2) + I2e
    M22 = m2e*lc2e**2 + I2e

    return np.array([[M11, M12],
                     [M12, M22]])



def coriolis_vector(q: np.ndarray, qdot: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the Coriolis/centripetal term C(q, qdot) @ qdot as a length-2 vector (N m)."""
    q1, q2 = q
    q1d, q2d = qdot
    m2e, lc2e, I2e = combine_payload(p.m2, p.lc2, p.I2, p.l2, p.m_p)
    h = m2e * p.l1 * lc2e * np.sin(q2)  # common factor in all the velocity-product terms
    # row 1: Coriolis (2 q1d q2d) + centripetal (q2d^2), row 2: centripetal from q1d
    return np.array([-h*(2*q1d*q2d + q2d**2),
                      h*q1d**2])


def gravity_vector(q: np.ndarray, p: ArmParams) -> np.ndarray:
    """Return the gravity torque vector G(q) as a length-2 vector (N m)."""
    q1, q2 = q
    m2e, lc2e, I2e = combine_payload(p.m2, p.lc2, p.I2, p.l2, p.m_p)
    # torque = weight x horizontal lever arm; q1 is from the horizontal so the lever arm uses cos
    g1 = (p.m1*p.lc1 + m2e*p.l1) * p.g * np.cos(q1) + m2e*lc2e*p.g*np.cos(q1 + q2)
    g2 = m2e*lc2e*p.g*np.cos(q1 + q2)
    
    return np.array([g1, g2])


def state_derivative(
    t: float,
    x: np.ndarray,
    p: ArmParams,
    tau_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Return dx/dt for x = [q1, q2, q1dot, q2dot].

    Solves M qddot = tau - C qdot - G - b qdot (use np.linalg.solve, not an
    explicit inverse).

    Args:
        t: Time (s).
        x: State, shape (4,).
        p: Arm parameters.
        tau_fn: Function t -> joint torques [tau1, tau2] (N m), shape (2,).

    Returns:
        Array of shape (4,): [q1dot, q2dot, q1ddot, q2ddot].
    """
    q = x[:2]
    qdot = x[2:]
    tau = np.asarray(tau_fn(t), dtype=float)  # applied joint torques at this time

    M = mass_matrix(q, p)
    # move everything except M qddot to the right-hand side
    rhs = tau - coriolis_vector(q, qdot, p) - gravity_vector(q, p) - p.b * qdot
    qddot = np.linalg.solve(M, rhs)  # solve the 2x2 system instead of inverting M

    return np.concatenate([qdot, qddot])


def energy(x: np.ndarray, p: ArmParams) -> float:
    """Return total mechanical energy T + V (J) for state x (payload included)."""
    q = x[:2]
    qdot = x[2:]
    q1, q2 = q
    m2e, lc2e, _ = combine_payload(p.m2, p.lc2, p.I2, p.l2, p.m_p)

    T = 0.5 * qdot @ mass_matrix(q, p) @ qdot  # kinetic energy
    # potential energy: heights use sin because q1 is from the horizontal
    V = p.g * (
        p.m1 * p.lc1 * np.sin(q1)
        + m2e * (p.l1 * np.sin(q1) + lc2e * np.sin(q1 + q2))
    )
    return float(T + V)
