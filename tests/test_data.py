"""Data-pipeline tests: shapes, reproducibility, no leakage between splits, torque consistency."""
import numpy as np
import pytest

from src import data
from src.data import N_T, T_GRID, generate, sample_inputs, torque_fn, torque_samples


@pytest.fixture(scope="module")
def small():
    return generate(n_total=40, n_ood=6, seed=123, substeps=2)


def test_split_shapes(small):
    n = {"train": 28, "val": 6, "test": 6, "ood": 6}
    for split, k in n.items():
        assert small[f"{split}_params"].shape == (k, 2)
        assert small[f"{split}_tau_coef"].shape == (k, 2, 3)
        assert small[f"{split}_torque"].shape == (k, 2, N_T)
        assert small[f"{split}_traj"].shape == (k, N_T, 4)
    assert small["t_grid"].shape == (N_T,)


def test_same_seed_reproduces_and_other_seed_differs(small):
    again = generate(n_total=40, n_ood=6, seed=123, substeps=2)
    assert all(np.array_equal(small[k], again[k]) for k in small)
    other = generate(n_total=40, n_ood=6, seed=124, substeps=2)
    assert not np.array_equal(small["train_params"], other["train_params"])


def test_splits_are_disjoint(small):
    keys = [
        {tuple(np.round(r, 12)) for r in small[f"{s}_tau_coef"].reshape(len(small[f"{s}_tau_coef"]), -1)}
        for s in ("train", "val", "test", "ood")
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (keys[i] & keys[j])


def test_in_range_and_out_of_range_parameters(small):
    for s in ("train", "val", "test"):
        mp, b = small[f"{s}_params"].T
        assert mp.min() >= data.RANGES["m_p"][0] and mp.max() <= data.RANGES["m_p"][1]
        assert b.min() >= data.RANGES["b"][0] and b.max() <= data.RANGES["b"][1]
    mp, b = small["ood_params"].T
    assert mp.min() >= data.OOD_RANGES["m_p"][0] and b.min() >= data.OOD_RANGES["b"][0]


def test_initial_condition_and_finiteness(small):
    for s in ("train", "val", "test", "ood"):
        traj = small[f"{s}_traj"]
        assert np.all(traj[:, 0] == data.X0)
        assert np.isfinite(traj).all()


def test_torque_samples_match_torque_fn():
    _, coef = sample_inputs(3, np.random.default_rng(0))
    for c in coef:
        grid = np.array([torque_fn(c)(t) for t in T_GRID]).T
        assert np.allclose(torque_samples(c), grid)
