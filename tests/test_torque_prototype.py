"""Consistency of the prototype Fourier torque family (scripts/torque_family_prototype.py)."""
import numpy as np

from scripts.torque_family_prototype import sample_fourier_coef, torque_fn_fourier, torque_samples_fourier
from src.data import T_GRID


def test_samples_match_function_and_seed_is_reproducible():
    c1 = sample_fourier_coef(3, np.random.default_rng(5))
    c2 = sample_fourier_coef(3, np.random.default_rng(5))
    assert np.array_equal(c1, c2)
    for c in c1:
        grid = np.array([torque_fn_fourier(c)(t) for t in T_GRID]).T
        assert np.allclose(torque_samples_fourier(c), grid)
