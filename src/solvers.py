"""Fixed-step RK4 and adaptive RK45 (Dormand-Prince) integrators for the arm ODE."""
from typing import Callable

import numpy as np


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray, h: float
) -> np.ndarray:
    """Advance one classical RK4 step.

    Args:
        f: Right-hand side f(t, x) -> dx/dt.
        t: Current time (s).
        x: Current state.
        h: Step size (s).

    Returns:
        State at t + h.
    """
    # four slopes: at the start, two at the midpoint, one at the end
    k1 = f(t, x)
    k2 = f(t + h / 2, x + h / 2 * k1)
    k3 = f(t + h / 2, x + h / 2 * k2)
    k4 = f(t + h, x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # weighted average of the slopes (1, 2, 2, 1) / 6


def rk4_integrate(
    f: Callable[[float, np.ndarray], np.ndarray], x0: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    """Integrate from x0 over a uniform time grid.

    Args:
        f: Right-hand side f(t, x) -> dx/dt.
        x0: Initial state, shape (n,).
        t_grid: Uniformly spaced times, shape (N,). Step size is t_grid[1] - t_grid[0].

    Returns:
        Array of shape (N, n) with the state at every time in t_grid.
    """
    x0 = np.asarray(x0, dtype=float)
    out = np.empty((len(t_grid), x0.size))
    out[0] = x0
    h = t_grid[1] - t_grid[0]
    # fixed step: march along the grid one step at a time
    for i in range(len(t_grid) - 1):
        out[i + 1] = rk4_step(f, t_grid[i], out[i], h)
    return out


# Dormand-Prince coefficients (the pair behind MATLAB's ode45 / scipy's RK45).
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)


def rk45_step(
    f: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray, h: float
) -> tuple:
    """One Dormand-Prince step.

    Returns:
        (x_new, err): the 5th-order state at t + h and the embedded error
        estimate (5th-order minus 4th-order solution), same shape as x.
    """
    k = []
    for i in range(7):
        # 7 stages, each built from the earlier slopes with the Butcher-table weights
        xi = x + h * sum(a * kj for a, kj in zip(_A[i], k)) if i else x
        k.append(f(t + _C[i] * h, xi))
    K = np.array(k)
    return x + h * (_B5 @ K), h * ((_B5 - _B4) @ K)  # 5th-order answer, and 5th minus 4th as the error estimate


def rk45_integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_grid: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float = None,
) -> np.ndarray:
    """Adaptive RK45 integration, reported at the times in t_grid.

    Internal steps are chosen from the error estimate and clipped so they land
    exactly on each grid time (no interpolation needed).

    Args:
        f: Right-hand side f(t, x) -> dx/dt.
        x0: Initial state, shape (n,).
        t_grid: Increasing output times, shape (N,).
        rtol, atol: Relative / absolute error tolerances.
        h0: Initial step guess (defaults to the grid spacing).

    Returns:
        Array of shape (N, n) with the state at every time in t_grid.
    """
    x = np.asarray(x0, dtype=float)
    out = np.empty((len(t_grid), x.size))
    out[0] = x
    h = h0 if h0 is not None else t_grid[1] - t_grid[0]
    t = t_grid[0]
    for i in range(1, len(t_grid)):
        t_end = t_grid[i]
        while t < t_end:
            h = min(h, t_end - t)  # don't step past the next output time
            x_new, err = rk45_step(f, t, x, h)
            # scale the error by the tolerance, so e <= 1 means the step is accurate enough
            scale = atol + rtol * np.maximum(np.abs(x), np.abs(x_new))
            e = np.sqrt(np.mean((err / scale) ** 2))
            if e <= 1.0:  # accept
                t += h
                x = x_new
                if t_end - t < 1e-12 * max(1.0, abs(t_end)):
                    t = t_end
            # if the step was rejected, t and x stay the same and it is retried with a smaller h
            # standard step-size controller (order 5 -> exponent 1/5)
            h *= min(5.0, max(0.2, 0.9 * (1.0 / max(e, 1e-10)) ** 0.2))
        out[i] = x
    return out
