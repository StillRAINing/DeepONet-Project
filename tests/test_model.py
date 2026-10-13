"""Shape and input-validation tests for the DeepONet and MLP baseline."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model import DeepONet, MLPBaseline, branch_input, n_params  # noqa: E402

B, N_T = 5, 151
D_U = 2 + 2 * N_T


@pytest.fixture
def batch():
    u = branch_input(np.random.rand(B, 2), np.random.rand(B, 2, N_T))
    return u, torch.linspace(0, 1, N_T)


@pytest.mark.parametrize("make", [lambda: DeepONet(D_U, p=8, width=16), lambda: MLPBaseline(D_U, width=16)])
def test_output_shape(make, batch):
    u, t = batch
    model = make()
    assert model(u, t).shape == (B, N_T, 4)
    assert model(u, t.expand(B, -1)).shape == (B, N_T, 4)  # per-sample time grids
    assert n_params(model) > 0


def test_branch_input_shape(batch):
    assert batch[0].shape == (B, D_U)


def test_deeponet_rejects_bad_shapes(batch):
    u, t = batch
    model = DeepONet(D_U, p=8, width=16)
    with pytest.raises(ValueError):
        model(u[:, :-1], t)
    with pytest.raises(ValueError):
        model(u, t.expand(B + 1, -1))


def test_deeponet_is_differentiable(batch):
    u, t = batch
    model = DeepONet(D_U, p=8, width=16)
    model(u, t).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("make", [lambda k: DeepONet(D_U, p=8, width=16, n_fourier=k),
                                  lambda k: MLPBaseline(D_U, width=16, n_fourier=k)])
def test_fourier_time_features_keep_the_interface(make, batch):
    u, t = batch
    plain, ff = make(0), make(6)
    assert ff(u, t).shape == plain(u, t).shape == (B, N_T, 4)
    assert n_params(ff) > n_params(plain)          # wider first layer of the trunk / MLP


def test_time_features_values():
    from src.model import time_features
    t = torch.tensor([[[0.0], [0.5], [1.0]]])      # (1, 3, 1)
    f = time_features(t, 2)
    assert f.shape == (1, 3, 5)                     # [t, sin pi t, sin 2 pi t, cos pi t, cos 2 pi t]
    torch.testing.assert_close(f[0, 1], torch.tensor([0.5, 1.0, 0.0, 0.0, -1.0]), atol=1e-6, rtol=0)
    assert time_features(t, 0) is t


# --- TODO (student): semantic tests the shape tests cannot give (docs/STUDY_GUIDE.md section 3) ---
todo = pytest.mark.skip(reason="TODO (student): write this test")


@todo
def test_einsum_equals_explicit_loop():
    """out[b,t,k] = sum_p branch[b,k,p] * trunk[t,k,p] + bias[k], written as plain loops on tiny sizes."""


@todo
def test_trunk_is_independent_of_u():
    """Changing u must not change the trunk output. State what that implies for caching."""


@todo
def test_changing_one_sample_leaves_others_unchanged():
    """Perturb u[0] only; outputs for b >= 1 are identical."""


@todo
def test_parameter_count_matches_hand_calculation():
    """Derive n_params for one small DeepONet config by hand and compare with n_params(model)."""
