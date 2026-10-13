"""Tests for the per-channel normaliser and the per-sample relative L2 loss."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.train import ChannelNormaliser, relative_l2_loss  # noqa: E402

N, N_T = 12, 20


@pytest.fixture
def raw():
    """Trajectories whose size varies by 100x across samples, and channels on different scales."""
    rng = np.random.default_rng(0)
    amp = np.logspace(0, 2, N)[:, None, None]
    chan = np.array([1.0, 2.0, 10.0, 20.0])
    return (rng.standard_normal((N, N_T, 4)) + 3.0) * amp * chan


def test_fit_matches_numpy_and_roundtrips(raw):
    norm = ChannelNormaliser.fit(raw)
    np.testing.assert_allclose(norm.mean.numpy(), raw.mean(axis=(0, 1)), rtol=1e-5)
    np.testing.assert_allclose(norm.std.numpy(), raw.std(axis=(0, 1)), rtol=1e-5)
    x = torch.as_tensor(raw, dtype=torch.float32)
    z = norm.encode(x)
    np.testing.assert_allclose(z.mean(dim=(0, 1)).numpy(), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(dim=(0, 1), unbiased=False).numpy(), 1, rtol=1e-4)
    torch.testing.assert_close(norm.decode(z), x, rtol=1e-4, atol=1e-3)


def test_loss_is_zero_when_prediction_is_exact(raw):
    norm = ChannelNormaliser.fit(raw)
    z = norm.encode(torch.as_tensor(raw, dtype=torch.float32))
    assert relative_l2_loss(z, z, norm).item() < 1e-5


def test_loss_is_blind_to_trajectory_size(raw):
    """A 10% over-prediction of the raw trajectory costs ~0.1 whether it is calm or spinning."""
    norm = ChannelNormaliser.fit(raw)
    x = torch.as_tensor(raw, dtype=torch.float32)
    per_sample = [
        relative_l2_loss(norm.encode(1.1 * x[i:i + 1]), norm.encode(x[i:i + 1]), norm).item()
        for i in range(N)
    ]
    np.testing.assert_allclose(per_sample, 0.1, rtol=1e-3)
    # contrast: plain MSE on the same normalised data differs by orders of magnitude across samples
    mse = [(norm.encode(1.1 * x[i:i + 1]) - norm.encode(x[i:i + 1])).pow(2).mean().item() for i in range(N)]
    assert max(mse) / min(mse) > 100


def test_loss_has_finite_gradient_at_zero_error(raw):
    norm = ChannelNormaliser.fit(raw)
    z = norm.encode(torch.as_tensor(raw, dtype=torch.float32))
    pred = z.clone().requires_grad_(True)
    relative_l2_loss(pred, z, norm).backward()
    assert torch.isfinite(pred.grad).all()


def test_fit_overfits_a_handful_of_samples():
    """Sanity check: with 5 samples the loss must go well below its starting value."""
    from src.model import DeepONet
    from src.train import InputNormaliser, fit

    rng = np.random.default_rng(1)
    n, n_t, d_u = 5, 16, 10
    u = rng.standard_normal((n, d_u)).astype(np.float32)
    t = torch.linspace(0, 1, n_t)
    traj = np.stack([np.sin((i + 1) * t.numpy() * 3)[:, None] * np.array([1, 2, 3, 4]) for i in range(n)])
    out_norm = ChannelNormaliser.fit(traj)
    y = out_norm.encode(torch.as_tensor(traj, dtype=torch.float32))
    uu = InputNormaliser.fit(u).encode(torch.as_tensor(u))
    model = DeepONet(d_u, p=16, width=32, branch_depth=2, trunk_depth=2)
    _, hist = fit(model, (uu, t, y), (uu, t, y), out_norm, epochs=1500, lr=3e-3, batch_size=5, verbose=0)
    assert hist["best_val"] < 0.3 * hist["train"][0]


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    from src.model import DeepONet
    from src.train import InputNormaliser, load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    cfg = {"d_u": 6, "p": 8, "width": 16}
    model = DeepONet(**cfg)
    in_norm = InputNormaliser(np.arange(6.0), np.arange(1.0, 7.0))
    out_norm = ChannelNormaliser(np.arange(4.0), np.arange(1.0, 5.0))
    save_checkpoint(tmp_path / "m.pt", model, "deeponet", cfg, in_norm, out_norm, {"best_val": 0.1})
    m2, i2, o2, hist = load_checkpoint(tmp_path / "m.pt")
    u, t = torch.randn(3, 6), torch.linspace(0, 1, 7)
    model.eval()
    torch.testing.assert_close(m2(u, t), model(u, t))
    torch.testing.assert_close(i2.std, in_norm.std)
    torch.testing.assert_close(o2.mean, out_norm.mean)
    assert hist["best_val"] == 0.1


def test_constant_input_feature_gets_unit_std():
    """A zero-padded feature must not be divided by a tiny std (OOD values would explode)."""
    from src.train import InputNormaliser
    u = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    norm = InputNormaliser.fit(u)
    assert norm.std[1].item() == 1.0
    assert norm.encode(torch.tensor([[0.5, 2.0]]))[0, 1].item() == 2.0


def test_coef_inputs_are_zero_padded_to_a_common_length():
    from src.data import smooth_torque_samples
    from src.train import COEF_MODES, raw_inputs
    rng = np.random.default_rng(0)
    short = rng.standard_normal((3, 2, 2, 6))
    data = {"a_params": np.ones((3, 2)), "a_tau_coef": short,
            "b_params": np.ones((3, 2)), "b_tau_coef": rng.standard_normal((3, 2, 2, COEF_MODES))}
    ua, ub = raw_inputs(data, "a", "coef"), raw_inputs(data, "b", "coef")
    assert ua.shape == ub.shape == (3, 2 + 4 * COEF_MODES)
    padded = ua[0, 2:].reshape(2, 2, COEF_MODES)
    assert np.all(padded[..., 6:] == 0)
    # padding is exact: the same torque comes out
    np.testing.assert_allclose(smooth_torque_samples(padded), smooth_torque_samples(short[0]), atol=1e-12)


def test_subset_train_cuts_every_train_array():
    """--n-train must cut ALL train_* arrays (it once missed train_tau_coef, breaking coef inputs)."""
    from scripts.train_quick import subset_train
    from src.train import raw_inputs
    rng = np.random.default_rng(0)
    d = {"train_params": rng.random((10, 2)), "train_torque": rng.random((10, 2, 7)),
         "train_tau_coef": rng.random((10, 2, 2, 6)), "train_traj": rng.random((10, 7, 4)),
         "val_params": rng.random((5, 2)), "t_grid": np.linspace(0, 1, 7)}
    sub = subset_train(d, 4)
    assert all(len(v) == 4 for k, v in sub.items() if k.startswith("train_"))
    assert sub["val_params"] is d["val_params"] and sub["t_grid"] is d["t_grid"]
    assert raw_inputs(sub, "train", "coef").shape[0] == 4


class _Fixed(torch.nn.Module):
    """Stand-in model: q1 = a * tau, q2 = tau^2 (normalised outputs); qdot given by ``qd1``, ``qd2``."""

    def __init__(self, qd1, qd2):
        super().__init__()
        self.qd1, self.qd2, self.w = qd1, qd2, torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, u, t):
        z = torch.stack([self.w * t, t ** 2, self.qd1(t), self.qd2(t)], dim=-1)
        return z + 0 * u.sum()


def test_consistency_loss_is_zero_when_qdot_is_the_derivative_of_q():
    """q1 = tau, q2 = tau^2 -> dq1/dtau = 1, dq2/dtau = 2 tau. With unit std, zero mean, t_final 1."""
    from src.train import loss_with_consistency
    norm = ChannelNormaliser(np.zeros(4), np.ones(4))
    model = _Fixed(lambda t: torch.ones_like(t), lambda t: 2 * t)
    u, t = torch.zeros(3, 2), torch.linspace(0, 1, 9)
    y = torch.ones(3, 9, 4)
    data, cons = loss_with_consistency(model, u, t, y, norm, t_final=1.0)
    assert cons.item() < 1e-5


def test_consistency_loss_detects_a_mismatch_and_has_gradient():
    from src.train import loss_with_consistency
    norm = ChannelNormaliser(np.zeros(4), np.ones(4))
    model = _Fixed(lambda t: 3 * torch.ones_like(t), lambda t: 2 * t)   # qdot1 should be 1, is 3
    u, t = torch.zeros(3, 2), torch.linspace(0, 1, 9)
    _, cons = loss_with_consistency(model, u, t, torch.ones(3, 9, 4), norm, t_final=1.0)
    assert cons.item() > 0.1
    cons.backward()
    assert model.w.grad is not None and torch.isfinite(model.w.grad)


def test_consistency_term_trains_and_reports_data_loss_only():
    from src.model import DeepONet
    from src.train import InputNormaliser, fit
    rng = np.random.default_rng(2)
    n, n_t, d_u = 6, 12, 5
    u = rng.standard_normal((n, d_u)).astype(np.float32)
    t = torch.linspace(0, 1, n_t)
    traj = np.stack([np.sin((i + 1) * t.numpy() * 2)[:, None] * np.array([1, 2, 3, 4]) for i in range(n)])
    out_norm = ChannelNormaliser.fit(traj)
    y = out_norm.encode(torch.as_tensor(traj, dtype=torch.float32))
    uu = InputNormaliser.fit(u).encode(torch.as_tensor(u))
    model = DeepONet(d_u, p=8, width=16, branch_depth=2, trunk_depth=2, n_fourier=2)
    _, hist = fit(model, (uu, t, y), (uu, t, y), out_norm, epochs=40, lr=3e-3, batch_size=6,
                  verbose=0, consistency_weight=0.1)
    assert np.isfinite(hist["train"]).all() and hist["best_val"] < hist["train"][0]
