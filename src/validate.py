"""Input validation for the arm simulator and the surrogate.

Two kinds of bad input are handled differently:

* **Invalid** (cannot be simulated or fed to the network at all): wrong shape, non-numeric, NaN or
  infinite values, negative payload mass, negative friction. These raise ``ValueError``.
* **Unsupported** (valid physics, but outside the region the surrogate was trained on): payload,
  friction or torque peak outside the training ranges, or query times outside ``[0, T_FINAL]``.
  These raise ``OutOfRangeWarning`` by default (the answer is still returned, but it is an
  extrapolation and should not be trusted), or ``ValueError`` with ``strict=True``.

The trained ranges are ``data.SMOOTH_RANGES`` (the main dataset). The torque limit is a limit on the
peak of ``|tau|`` over the horizon, because that is what the smooth family caps (``A``).
"""
import warnings

import numpy as np

from .data import N_T, SMOOTH_RANGES, T_FINAL


class OutOfRangeWarning(UserWarning):
    """Inputs are physically valid but outside the surrogate's training range (extrapolation)."""


def _as_float_array(x, name: str) -> np.ndarray:
    """Convert to a float64 array or raise a ValueError that names the argument."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {type(x).__name__}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _report(message: str, strict: bool) -> None:
    """Raise (strict) or warn (default) about an out-of-range input."""
    if strict:
        raise ValueError(message)
    warnings.warn(message, OutOfRangeWarning, stacklevel=3)


def validate_params(params, *, ranges: dict = SMOOTH_RANGES, strict: bool = False) -> np.ndarray:
    """Check payload mass and friction. Returns ``params`` as a float64 array of shape (B, 2).

    Args:
        params: ``[m_p, b]`` of shape (2,), or (B, 2). m_p in kg, b in N m s/rad.
        ranges: training ranges, keys ``m_p`` and ``b`` (low, high); ``None`` checks physical
            validity only (used by the simulators, which legitimately run out-of-range cases).
        strict: raise instead of warn when a value is outside the training range.

    Raises:
        ValueError: wrong shape, NaN/inf, ``m_p < 0`` or ``b < 0``, or out of range with ``strict``.
    """
    arr = _as_float_array(params, "params")
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"params must have shape (2,) or (B, 2) = [m_p, b]; got {arr.shape}")
    if np.any(arr[:, 0] < 0):
        raise ValueError("payload mass m_p must be >= 0 kg")
    if np.any(arr[:, 1] < 0):
        raise ValueError("friction b must be >= 0 N m s/rad")
    # range check only for the surrogate; the simulators pass ranges=None
    for col, key, unit in (((0, "m_p", "kg"), (1, "b", "N m s/rad")) if ranges is not None else ()):
        lo, hi = ranges[key]
        bad = (arr[:, col] < lo) | (arr[:, col] > hi)
        if bad.any():
            _report(f"{key} outside the trained range [{lo}, {hi}] {unit} in {int(bad.sum())} of "
                    f"{len(arr)} rows: extrapolation, treat the prediction as unreliable", strict)
    return arr


def validate_torque(torque, *, n_t: int = N_T, ranges: dict = SMOOTH_RANGES,
                    strict: bool = False) -> np.ndarray:
    """Check torque samples on the fixed grid. Returns a float64 array of shape (B, 2, n_t).

    Args:
        torque: samples of tau_1, tau_2 in N m on the fixed 151-point grid, shape (2, n_t) or
            (B, 2, n_t).
        n_t: number of grid points the model expects.
        ranges: training ranges; ``A`` is the cap on ``max |tau|`` over the horizon.
        strict: raise instead of warn when the peak exceeds the training cap.

    Raises:
        ValueError: wrong shape, NaN/inf, or peak above the cap with ``strict``.
    """
    arr = _as_float_array(torque, "torque")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (2, n_t):
        raise ValueError(f"torque must have shape (2, {n_t}) or (B, 2, {n_t}); got {arr.shape}")
    cap = ranges["A"][1]  # training cap on the torque peak (6 N m)
    peak = np.abs(arr).max(axis=(1, 2))  # peak |tau| of each sample over both joints and all times
    if np.any(peak > cap):
        _report(f"torque peak |tau| up to {peak.max():.2f} N m exceeds the trained cap of {cap} N m "
                f"in {int((peak > cap).sum())} of {len(arr)} rows: extrapolation", strict)
    return arr


def validate_time(t, *, t_final: float = T_FINAL, strict: bool = False) -> np.ndarray:
    """Check query times in seconds. Returns a 1-D float64 array.

    Raises:
        ValueError: NaN/inf, not 1-D, empty, negative, or beyond ``t_final`` with ``strict``.
    """
    arr = _as_float_array(t, "t")
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"t must be a non-empty 1-D array of seconds; got shape {arr.shape}")
    if np.any(arr < 0):
        raise ValueError("query times must be >= 0 s")
    # beyond the horizon is allowed but warned: the network never saw those times
    if np.any(arr > t_final):
        _report(f"query time beyond the horizon T = {t_final} s: the network was only trained on "
                f"[0, {t_final}] s", strict)
    return arr


def validate_coef(coef) -> np.ndarray:
    """Check the sine-family coefficients ``[A, f, phi]`` per joint used by ``data.simulate``.

    Returns a float64 array of shape (2, 3).

    Raises:
        ValueError: wrong shape, NaN/inf, negative amplitude or negative frequency.
    """
    arr = _as_float_array(coef, "coef")
    if arr.shape != (2, 3):
        raise ValueError(f"coef must have shape (2, 3) = [A, f, phi] per joint; got {arr.shape}")
    if np.any(arr[:, 0] < 0):
        raise ValueError("torque amplitude A must be >= 0 N m")
    if np.any(arr[:, 1] < 0):
        raise ValueError("torque frequency f must be >= 0 Hz")
    return arr
