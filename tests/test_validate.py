"""Invalid inputs raise ValueError; unsupported (out-of-range) inputs warn or, with strict, raise."""
import warnings

import numpy as np
import pytest

from src.data import N_T, simulate, simulate_smooth
from src.model import DeepONet
from src.surrogate import Surrogate
from src.train import COEF_MODES, ChannelNormaliser, InputNormaliser
from src.validate import (OutOfRangeWarning, validate_coef, validate_params, validate_time,
                          validate_torque)

GOOD_COEF = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])


@pytest.mark.parametrize("bad", [[np.nan, 0.1], [0.5, np.inf], [-0.1, 0.1], [0.5, -0.1], [0.5],
                                 [[0.5, 0.1, 0.2]], "abc", None])
def test_params_invalid_raise(bad):
    with pytest.raises(ValueError):
        validate_params(bad)


def test_params_shapes_and_in_range_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate_params([0.5, 0.1]).shape == (1, 2)
        assert validate_params(np.full((4, 2), [0.5, 0.1])).shape == (4, 2)


def test_params_out_of_range_warns_or_raises_strict():
    with pytest.warns(OutOfRangeWarning):
        validate_params([1.5, 0.1])
    with pytest.warns(OutOfRangeWarning):
        validate_params([0.5, 0.9])
    with pytest.raises(ValueError):
        validate_params([1.5, 0.1], strict=True)


def test_params_ranges_none_checks_physics_only():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_params([5.0, 9.0], ranges=None)  # absurd but physical: no warning
    with pytest.raises(ValueError):
        validate_params([-1.0, 0.1], ranges=None)


def test_torque_shape_nan_and_cap():
    ok = np.zeros((2, N_T))
    assert validate_torque(ok).shape == (1, 2, N_T)
    with pytest.raises(ValueError):
        validate_torque(np.zeros((2, N_T - 1)))
    bad = ok.copy()
    bad[0, 3] = np.nan
    with pytest.raises(ValueError):
        validate_torque(bad)
    big = ok.copy()
    big[1, 5] = 50.0
    with pytest.warns(OutOfRangeWarning):
        validate_torque(big)
    with pytest.raises(ValueError):
        validate_torque(big, strict=True)


def test_time_checks():
    assert validate_time([0.0, 1.5, 3.0]).shape == (3,)
    for bad in ([-0.1, 1.0], [np.nan], [], [[1.0]]):
        with pytest.raises(ValueError):
            validate_time(bad)
    with pytest.warns(OutOfRangeWarning):
        validate_time([0.0, 4.0])


def test_coef_checks():
    assert validate_coef(GOOD_COEF).shape == (2, 3)
    assert validate_coef(np.zeros((2, 3))).shape == (2, 3)  # zero torque (used by viz) is valid
    for bad in (np.zeros((3, 2)), GOOD_COEF * np.nan, GOOD_COEF * [-1, 1, 1], GOOD_COEF * [1, -1, 1]):
        with pytest.raises(ValueError):
            validate_coef(bad)


def test_simulate_rejects_bad_input_but_runs_ood():
    with pytest.raises(ValueError):
        simulate([-1.0, 0.1], GOOD_COEF)
    with pytest.raises(ValueError):
        simulate([0.5, 0.1], np.full((2, 3), np.nan))
    with pytest.raises(ValueError):
        simulate_smooth([0.5, np.nan], np.zeros((2, 2, 6)))
    with warnings.catch_warnings():  # out-of-range physics is allowed in the simulator (OOD sets)
        warnings.simplefilter("error")
        assert simulate([1.4, 0.7], GOOD_COEF).shape == (N_T, 4)


def _surrogate(mode: str) -> Surrogate:
    d_u = 2 + 2 * N_T if mode == "samples" else 2 + 4 * COEF_MODES
    model = DeepONet(d_u=d_u, p=8, width=16, branch_depth=2, trunk_depth=2).eval()
    return Surrogate(model, InputNormaliser(np.zeros(d_u), np.ones(d_u)),
                     ChannelNormaliser(np.zeros(4), np.ones(4)), mode)


def test_surrogate_samples_mode_end_to_end():
    sur = _surrogate("samples")
    out = sur.predict([0.5, 0.1], np.zeros((2, N_T)))
    assert out.shape == (1, N_T, 4) and np.all(np.isfinite(out))
    assert sur.predict([0.5, 0.1], np.zeros((2, N_T)), t=[0.0, 1.0]).shape == (1, 2, 4)
    with pytest.raises(ValueError):
        sur.predict([0.5, np.nan], np.zeros((2, N_T)))
    with pytest.raises(ValueError):
        sur.predict([0.5, 0.1], np.zeros((2, 10)))
    with pytest.raises(ValueError):  # row-count mismatch
        sur.predict(np.full((3, 2), [0.5, 0.1]), np.zeros((2, 2, N_T)))
    with pytest.warns(OutOfRangeWarning):
        sur.predict([1.5, 0.1], np.zeros((2, N_T)))


def test_surrogate_coef_mode_pads_and_validates():
    sur = _surrogate("coef")
    assert sur.predict([0.5, 0.1], np.zeros((2, 2, 6))).shape == (1, N_T, 4)
    with pytest.raises(ValueError):
        sur.predict([0.5, 0.1], np.zeros((2, 2, COEF_MODES + 1)))
    with pytest.raises(ValueError):
        sur.predict([0.5, 0.1], np.full((2, 2, 6), np.nan))
