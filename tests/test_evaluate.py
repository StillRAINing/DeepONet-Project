"""Tests for the evaluation metrics and physical-consistency checks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data import N_T, T_FINAL, T_GRID, simulate, torque_samples  # noqa: E402
from src.evaluate import (energy_balance_residual, energy_balance_summary, error_vs_time,  # noqa: E402
                          fraction_below, qdot_consistency,
                          rel_l2_per_channel, rel_l2_per_sample, summarise, worst_cases)
from src.train import ChannelNormaliser  # noqa: E402


@pytest.fixture
def true():
    rng = np.random.default_rng(0)
    return rng.standard_normal((6, 20, 4)) * np.array([1.0, 2.0, 10.0, 20.0]) + 1.0


def test_exact_prediction_has_zero_error(true):
    assert rel_l2_per_sample(true, true, None).max() == 0
    assert rel_l2_per_channel(true, true).max() == 0
    assert error_vs_time(true, true).max() == 0


def test_scaled_prediction_gives_that_relative_error(true):
    np.testing.assert_allclose(rel_l2_per_sample(1.1 * true, true, None), 0.1)
    np.testing.assert_allclose(rel_l2_per_sample(1.1 * true, true, [1, 2, 10, 20]), 0.1)
    np.testing.assert_allclose(rel_l2_per_channel(1.1 * true, true), 0.1)


def test_scale_choice_changes_the_answer_when_one_channel_is_wrong(true):
    """Error only in the big qdot2 channel: raw units are dominated by it, scaled units are not."""
    pred = true.copy()
    pred[..., 3] *= 1.5
    raw = rel_l2_per_sample(pred, true, None)
    scaled = rel_l2_per_sample(pred, true, [1, 2, 10, 20])
    assert not np.allclose(raw, scaled)
    per_ch = rel_l2_per_channel(pred, true)
    np.testing.assert_allclose(per_ch[:, :3], 0, atol=1e-12)
    np.testing.assert_allclose(per_ch[:, 3], 0.5)


def test_error_vs_time_is_finite_at_t0_and_localised(true):
    true[:, 0] = 0.0  # x0 = 0, as in the dataset
    pred = true.copy()
    pred[:, 10] += 1.0
    e = error_vs_time(pred, true)
    assert np.isfinite(e).all()
    assert (e[:, 10] > 0).all() and np.delete(e, 10, axis=1).max() == 0


def test_summarise_and_worst_cases():
    e = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    s = summarise(e)
    assert s["max"] == 0.9 and s["median"] == 0.3 and s["mean"] == pytest.approx(0.4)
    assert list(worst_cases(e, 2)) == [3, 1]


class _Known(torch.nn.Module):
    """q1 = q2 = tau^2 + c (tau = t / T), qdot = dq/dt exactly (+ offset ``bias`` on qdot)."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, u, t):
        t = t.unsqueeze(0).expand(len(u), -1) if t.ndim == 1 else t
        q = t ** 2
        qd = 2 * t / T_FINAL + self.bias       # d(tau^2)/dt = 2 tau / T
        return torch.stack([q, q, qd, qd], dim=-1)


def test_qdot_consistency_is_zero_for_a_consistent_model_and_detects_an_offset():
    norm = ChannelNormaliser(np.zeros(4), np.ones(4))
    u, t = torch.zeros(3, 5), torch.linspace(0, 1, 11)
    assert np.abs(qdot_consistency(_Known(), u, t, norm)).max() < 1e-5
    off = qdot_consistency(_Known(bias=0.3), u, t, norm)
    np.testing.assert_allclose(off, -0.3, atol=1e-5)  # d/dt q minus qdot output = -bias


def test_qdot_consistency_applies_the_std_scaling():
    """Normalised z = tau^2 with std 3 means raw q = 3 tau^2, so dq/dt = 6 tau / T."""
    norm = ChannelNormaliser(np.zeros(4), np.array([3.0, 3.0, 1.0, 1.0]))
    t = torch.linspace(0, 1, 11)
    model = _Known()
    res = qdot_consistency(model, torch.zeros(2, 5), t, norm)
    expected = 6 * t.numpy() / T_FINAL - 2 * t.numpy() / T_FINAL  # raw dq/dt minus raw qdot out
    np.testing.assert_allclose(res[0, :, 0], expected, atol=1e-5)


def test_energy_balance_holds_for_the_true_trajectory_and_catches_a_wrong_one():
    params = np.array([[0.5, 0.2], [0.0, 0.4]])
    coef = np.array([[[3.0, 0.5, 0.0], [2.0, 0.8, 1.0]], [[5.0, 0.3, 2.0], [1.0, 0.6, 0.5]]])
    traj = np.stack([simulate(p, c) for p, c in zip(params, coef)])
    torque = np.stack([torque_samples(c) for c in coef])
    res = energy_balance_residual(traj, params, torque, T_GRID)
    assert res.shape == (2, N_T) and np.all(res[:, 0] == 0)
    swing = np.abs(traj[:, :, 2:]).max()
    assert swing > 0.5                                       # not a trivial trajectory
    # Energy changes by many J (torque + gravity); the balance residual must be far smaller.
    assert np.abs(res).max() < 0.01 * 100
    bad = traj.copy()
    bad[:, :, 2:] *= 1.2                                     # velocities no longer match the energy change
    assert np.abs(energy_balance_residual(bad, params, torque, T_GRID)).max() > 5 * np.abs(res).max()


def test_fraction_below():
    f = fraction_below(np.array([0.005, 0.02, 0.07, 0.3]), (0.01, 0.05, 0.10))
    assert f == {0.01: 0.25, 0.05: 0.5, 0.10: 0.75}


def test_energy_balance_summary_floor_and_excess():
    params = np.array([[0.5, 0.2]])
    coef = np.array([[[3.0, 0.5, 0.0], [2.0, 0.8, 1.0]]])
    true = np.stack([simulate(p, c) for p, c in zip(params, coef)])
    torque = np.stack([torque_samples(c) for c in coef])
    same = energy_balance_summary(true, true, params, torque, T_GRID)
    assert same["excess_J"][0] == 0 and same["true_rel"][0] < 0.05
    bad = true.copy()
    bad[:, :, 2:] *= 1.2
    worse = energy_balance_summary(bad, true, params, torque, T_GRID)
    assert worse["excess_J"][0] > 0 and worse["pred_rel"][0] > same["pred_rel"][0]
