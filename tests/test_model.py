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


# --- Semantic tests the shape tests cannot give ---
# Written by the AI on request (AI_LOG row to add).


def test_einsum_equals_explicit_loop():
    """out[b,t,k] = sum_p branch[b,k,p] * trunk[t,k,p] + bias[k], written as plain loops on tiny sizes."""
    torch.manual_seed(0)
    Bs, n_t, d_u, p = 2, 3, 5, 4
    model = DeepONet(d_u, p=p, width=8)
    with torch.no_grad():
        model.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))  # non-zero, so the bias is checked too
    u, t = torch.randn(Bs, d_u), torch.linspace(0, 1, n_t)
    out = model(u, t).detach()
    with torch.no_grad():
        br = model.branch(u).view(Bs, 4, p)
        tr = model.trunk(t[:, None]).view(n_t, 4, p)
    ref = torch.zeros(Bs, n_t, 4)
    for b in range(Bs):
        for i in range(n_t):
            for k in range(4):
                s = 0.0
                for j in range(p):
                    s += br[b, k, j] * tr[i, k, j]
                ref[b, i, k] = s + model.bias[k]
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_trunk_is_independent_of_u():
    """Changing u must not change the trunk output.

    Implication: for a fixed query grid the trunk basis (n_t, 4, p) can be computed once and reused for
    every new input; each new query then costs one branch pass plus a (4, p) x (n_t, 4, p) contraction.
    """
    torch.manual_seed(0)
    model = DeepONet(D_U, p=8, width=16)
    seen = []
    model.trunk.register_forward_hook(lambda mod, inp, out: seen.append(out.detach().clone()))
    t = torch.linspace(0, 1, N_T)
    u1, u2 = torch.randn(B, D_U), 10 * torch.randn(B, D_U)
    out1, out2 = model(u1, t), model(u2, t)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert not torch.allclose(out1, out2)  # the output does depend on u, through the branch only


def test_changing_one_sample_leaves_others_unchanged():
    """Perturb u[0] only; outputs for b >= 1 are identical (no mixing across the batch dimension)."""
    torch.manual_seed(0)
    t = torch.linspace(0, 1, N_T)
    for model in (DeepONet(D_U, p=8, width=16), MLPBaseline(D_U, width=16)):
        u = torch.randn(B, D_U)
        u2 = u.clone()
        u2[0] += 5.0
        a, b = model(u, t), model(u2, t)
        torch.testing.assert_close(a[1:], b[1:], rtol=0, atol=0)
        assert not torch.allclose(a[0], b[0])


def test_output_at_t_depends_only_on_t():
    """Permuting the time grid permutes the outputs: the trunk sees each time on its own."""
    torch.manual_seed(0)
    model = DeepONet(D_U, p=8, width=16, n_fourier=3)
    u, t = torch.randn(B, D_U), torch.rand(N_T)
    perm = torch.randperm(N_T)
    torch.testing.assert_close(model(u, t)[:, perm], model(u, t[perm]))


def test_parameter_count_matches_hand_calculation():
    """DeepONet(d_u=6, p=2, width=3, branch_depth=1, trunk_depth=1), by hand:

    branch: Linear(6 -> 3) = 6*3 + 3 = 21, Linear(3 -> 4*2 = 8) = 3*8 + 8 = 32   -> 53
    trunk:  Linear(1 -> 3) = 1*3 + 3 = 6,  Linear(3 -> 8)          = 3*8 + 8 = 32 -> 38
    bias:   4 (one per output channel)                                            -> 4
    total 95. With n_fourier = 2 the trunk input is 1 + 2*2 = 5, so its first layer is 5*3 + 3 = 18: total 107.
    MLPBaseline(d_u=6, width=3, depth=1): Linear(7 -> 3) = 24, Linear(3 -> 4) = 16 -> 40.
    """
    assert n_params(DeepONet(6, p=2, width=3, branch_depth=1, trunk_depth=1)) == 95
    assert n_params(DeepONet(6, p=2, width=3, branch_depth=1, trunk_depth=1, n_fourier=2)) == 107
    assert n_params(MLPBaseline(6, width=3, depth=1)) == 40
