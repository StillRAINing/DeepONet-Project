"""Smooth random torque family: shapes, reproducibility, amplitude scheme, OOD shells, spectrum."""
import numpy as np
import pytest

from src import data
from src.data import (K_MODES, N_T, OOD_SMOOTH, PEAK_GRID, SMOOTH_RANGES, T_GRID, generate_smooth,
                      sample_smooth_inputs, smooth_torque_fn, smooth_torque_samples, spectral_std)


@pytest.fixture(scope="module")
def small():
    return generate_smooth(n_total=30, n_ood=4, seed=7, substeps=2)


def test_spectral_std_decays_and_tail_is_negligible():
    s = spectral_std(data.ELL, 50)
    assert np.all(np.diff(s) < 0)
    var = s**2
    assert var[K_MODES:].sum() / var.sum() < 1e-3  # modes beyond K carry almost no variance


def test_split_shapes(small):
    n = {"train": 21, "val": 4, "test": 5, **{name: 4 for name in OOD_SMOOTH}}
    for split, k in n.items():
        km = 13 if split == "ood_smooth" else K_MODES
        assert small[f"{split}_params"].shape == (k, 2)
        assert small[f"{split}_tau_coef"].shape == (k, 2, 2, km)
        assert small[f"{split}_torque"].shape == (k, 2, N_T)
        assert small[f"{split}_traj"].shape == (k, N_T, 4)


def test_seed_reproduces_and_workers_do_not_change_result(small):
    again = generate_smooth(n_total=30, n_ood=4, seed=7, substeps=2)
    assert all(np.array_equal(small[k], again[k]) for k in small)
    parallel = generate_smooth(n_total=30, n_ood=4, seed=7, substeps=2, workers=2)
    assert all(np.array_equal(small[k], parallel[k]) for k in small)
    other = generate_smooth(n_total=30, n_ood=4, seed=8, substeps=2)
    assert not np.array_equal(small["train_tau_coef"], other["train_tau_coef"])


def test_torque_samples_match_torque_fn():
    _, coef = sample_smooth_inputs(3, np.random.default_rng(0))
    for c in coef:
        grid = np.array([smooth_torque_fn(c)(t) for t in T_GRID]).T
        assert np.allclose(smooth_torque_samples(c), grid)


def test_torque_peak_lies_in_the_drawn_range():
    lo, hi = SMOOTH_RANGES["A"]
    _, coef = sample_smooth_inputs(200, np.random.default_rng(1))
    peaks = np.stack([np.abs(smooth_torque_samples(c, PEAK_GRID)).max(axis=1) for c in coef])
    assert peaks.min() >= lo and peaks.max() <= hi + 1e-9
    assert peaks.max() > 0.9 * hi and peaks.min() < 0.1 * hi  # the range is actually covered


@pytest.mark.parametrize("name,lo,hi", [("ood_amp1", 6.0, 9.0), ("ood_amp2", 9.0, 12.0)])
def test_amplitude_shells_are_outside_the_training_range(name, lo, hi):
    cfg = OOD_SMOOTH[name]
    _, coef = sample_smooth_inputs(100, np.random.default_rng(2), cfg["ranges"])
    peaks = np.stack([np.abs(smooth_torque_samples(c, PEAK_GRID)).max(axis=1) for c in coef])
    assert peaks.min() >= lo - 1e-9 and peaks.max() <= hi + 1e-9


def test_params_in_and_out_of_range(small):
    for s in ("train", "val", "test", "ood_amp1", "ood_amp2", "ood_smooth"):
        mp, b = small[f"{s}_params"].T
        assert mp.min() >= data.RANGES["m_p"][0] and mp.max() <= data.RANGES["m_p"][1]
        assert b.min() >= data.RANGES["b"][0] and b.max() <= data.RANGES["b"][1]
    mp, b = small["ood_params"].T
    assert mp.min() >= data.OOD_RANGES["m_p"][0] and b.min() >= data.OOD_RANGES["b"][0]


def test_initial_condition_and_finiteness(small):
    for s in ("train", "val", "test", *OOD_SMOOTH):
        traj = small[f"{s}_traj"]
        assert np.all(traj[:, 0] == data.X0)
        assert np.isfinite(traj).all()
