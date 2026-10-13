"""Training pieces: per-channel normalisation and the per-sample relative L2 loss.

The train / validation loop (``fit``) was written by the AI on request: see AI_LOG row 17.

Why these two together (docs/DESIGN_LOG.md, A8):
  * channels [q1, q2, q1dot, q2dot] live on different scales -> scale each channel by its own std;
  * trajectory size varies by ~100x (calm swing vs. spinning arm) and MSE grows with the square of
    the size, so a few spinners would dominate -> divide each sample's error by its own size.
"""
import copy

import numpy as np
import torch

from .data import K_MODES, OOD_SMOOTH, T_FINAL

# Largest number of Fourier modes per joint in any split of the smooth dataset (ood_smooth has 13).
COEF_MODES = max([K_MODES] + [cfg.get("k_modes", K_MODES) for cfg in OOD_SMOOTH.values()])


class ChannelNormaliser:
    """z-score per output channel: z = (x - mean) / std, statistics over (samples, time).

    Fit on the TRAIN split only, then reuse the same object for val, test, OOD and evaluation.
    """

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)  # (4,)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)    # (4,)

    @classmethod
    def fit(cls, traj) -> "ChannelNormaliser":
        """traj: (N, n_t, 4) training trajectories (NumPy, raw units)."""
        traj = np.asarray(traj, dtype=np.float64)
        return cls(traj.mean(axis=(0, 1)), traj.std(axis=(0, 1)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (..., 4) -> normalised."""
        return (x - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised (..., 4) -> raw units."""
        return z * self.std + self.mean


def relative_l2_loss(pred: torch.Tensor, true: torch.Tensor, norm: ChannelNormaliser) -> torch.Tensor:
    """Mean over the batch of ||pred - true|| / ||true||, per sample, on per-channel-scaled data.

    pred, true: (B, n_t, 4), both normalised by ``norm``. The norm is taken over (time, channel)
    of each sample, so every trajectory counts equally whatever its size, and a 5% error on a calm
    trajectory costs the same as a 5% error on a spinning one.

    The denominator uses the channel-SCALED but not centred truth, ``true_raw / std`` (= true +
    mean / std), so it is the size of the trajectory itself, not its distance from the channel mean.
    The numerator is unaffected by the mean (it cancels in pred - true).
    """
    diff = pred - true
    size = true + norm.mean / norm.std
    num = torch.sqrt(diff.pow(2).sum(dim=(1, 2)) + 1e-12)  # +1e-12: finite gradient at diff = 0
    den = torch.sqrt(size.pow(2).sum(dim=(1, 2))).clamp_min(1e-8)
    return (num / den).mean()


def loss_with_consistency(model, u, t, y, norm: ChannelNormaliser, t_final: float = T_FINAL):
    """(relative L2 loss, consistency loss) from ONE forward pass.

    The consistency loss measures how far the model's own q outputs are from its own qdot outputs:
        r_j = dq_j/dt (autodiff of the q output w.r.t. time) - qdot_j (the qdot output),  j = 1, 2
    in raw units, divided by std of the qdot channel, then the same per-sample norm and truth-size
    denominator as ``relative_l2_loss``. It uses no labels in the numerator: it asks the network to
    agree with itself (the same quantity as ``evaluate.qdot_consistency``, but differentiable).
    The model takes tau = t / t_final, hence the division by ``t_final``; each sample gets its own
    copy of the time grid so the gradient of the summed output is the per-sample, per-time derivative.
    """
    tb = t.unsqueeze(0).repeat(len(u), 1).requires_grad_(True)               # (B, n_t)
    z = model(u, tb)                                                           # (B, n_t, 4)
    dq = torch.stack([torch.autograd.grad(z[..., j].sum(), tb, create_graph=True)[0]
                      * norm.std[j] / t_final for j in range(2)], dim=-1)      # (B, n_t, 2) rad/s
    qdot = z[..., 2:4] * norm.std[2:4] + norm.mean[2:4]                         # raw rad/s
    res = (dq - qdot) / norm.std[2:4]
    size = y + norm.mean / norm.std
    num = torch.sqrt(res.pow(2).sum(dim=(1, 2)) + 1e-12)
    den = torch.sqrt(size.pow(2).sum(dim=(1, 2))).clamp_min(1e-8)
    return relative_l2_loss(z, y, norm), (num / den).mean()


class InputNormaliser:
    """z-score per branch-input feature (m_p, b, each torque sample). Fit on the TRAIN split only."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)  # (d_u,)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)    # (d_u,)

    @classmethod
    def fit(cls, u) -> "InputNormaliser":
        """u: (N, d_u) training branch inputs (raw units)."""
        u = np.asarray(u, dtype=np.float64)
        std = u.std(axis=0)
        # A feature that is constant in training (e.g. a zero-padded Fourier mode) gets std 1, not a
        # tiny floor: dividing by 1e-8 would turn any non-zero OOD value into a huge input.
        return cls(u.mean(axis=0), np.where(std < 1e-8, 1.0, std))

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Raw branch input (N, d_u) -> z-scored input using the stored training mean and std."""
        return (u - self.mean) / self.std


def raw_inputs(data: dict, name: str, mode: str = "samples") -> np.ndarray:
    """Raw (un-normalised) branch input for one split, (N, d_u).

    mode "samples": [m_p, b, tau_1(t_i), tau_2(t_i)] on the 151-point grid (d_u = 304).
    mode "coef":    [m_p, b, the Fourier coefficients of both torques] (CLAUDE.md weak point 2,
                    DESIGN_LOG W6). Coefficients are zero-padded to COEF_MODES modes per joint, so the
                    input has the same meaning in every split (d_u = 2 + 2 * 2 * 13 = 54). Mode k has
                    the same frequency in every set (fixed period), so padding is exact: the 6-mode
                    training torques simply have zero amplitude in modes 7..13. Those inputs are
                    always zero in training, so a coef model has never learnt what they do: the
                    ``ood_smooth`` set is a genuine extrapolation for it.
    """
    params = np.asarray(data[f"{name}_params"])
    if mode == "samples":
        extra = np.asarray(data[f"{name}_torque"])
    elif mode == "coef":
        extra = np.asarray(data[f"{name}_tau_coef"])
        k = extra.shape[-1]
        if k > COEF_MODES:
            raise ValueError(f"{name} has {k} Fourier modes; COEF_MODES is {COEF_MODES}")
        extra = np.pad(extra, [(0, 0)] * (extra.ndim - 1) + [(0, COEF_MODES - k)])
    else:
        raise ValueError(f"unknown input mode {mode!r}")
    return np.concatenate([params, extra.reshape(len(params), -1)], axis=1)


def make_split(data: dict, name: str, in_norm: InputNormaliser, out_norm: ChannelNormaliser,
               t_final: float, mode: str = "samples"):
    """Turn one split of the dataset dict into normalised tensors (u, t, y).

    u: (N, d_u) normalised branch input; t: (n_t,) time rescaled to [0, 1]; y: (N, n_t, 4) normalised.
    """
    u = in_norm.encode(torch.as_tensor(raw_inputs(data, name, mode), dtype=torch.float32))
    t = torch.as_tensor(data["t_grid"], dtype=torch.float32) / t_final
    y = out_norm.encode(torch.as_tensor(data[f"{name}_traj"], dtype=torch.float32))
    return u, t, y


@torch.no_grad()
def evaluate_loss(model, u, t, y, norm: ChannelNormaliser, batch_size: int = 500) -> float:
    """Mean relative L2 loss over a whole split (sample-weighted, so batch size does not matter)."""
    model.eval()
    total = 0.0
    for i in range(0, len(u), batch_size):
        pred = model(u[i:i + batch_size], t)
        total += relative_l2_loss(pred, y[i:i + batch_size], norm).item() * len(pred)
    return total / len(u)


def train_epoch(model, opt, u, t, y, norm: ChannelNormaliser, batch_size: int,
                gen: torch.Generator, consistency_weight: float = 0.0) -> float:
    """One pass over the training set in shuffled mini-batches; returns the mean data loss.

    With ``consistency_weight`` > 0 the optimised loss is data + weight * consistency, but the
    returned (and logged) number is the data loss alone, so curves stay comparable across runs.
    """
    model.train()
    perm = torch.randperm(len(u), generator=gen)
    total = 0.0
    for i in range(0, len(u), batch_size):
        idx = perm[i:i + batch_size]
        if consistency_weight > 0:
            data, cons = loss_with_consistency(model, u[idx], t, y[idx], norm)
            loss = data + consistency_weight * cons
        else:
            data = loss = relative_l2_loss(model(u[idx], t), y[idx], norm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += data.item() * len(idx)
    return total / len(u)


def fit(model, train, val, norm: ChannelNormaliser, epochs: int = 200, lr: float = 1e-3,
        batch_size: int = 100, seed: int = 3001, verbose: int = 10, schedule: str = "none",
        consistency_weight: float = 0.0):
    """Train with Adam and keep the weights with the lowest VALIDATION loss.

    schedule: "none" (constant lr) or "cosine" (lr decays from ``lr`` to ``lr / 100`` over the epochs).
    consistency_weight: weight of the qdot = dq/dt self-consistency term (0 = off). Validation and
    model selection always use the plain relative L2, whatever the training loss was.

    train, val: (u, t, y) tuples from ``make_split``. The test and OOD splits are never passed in.
    Returns (model with best weights loaded, history dict with train/val loss per epoch).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
             if schedule == "cosine" else None)
    hist = {"train": [], "val": [], "lr": [], "best_epoch": 0, "best_val": float("inf")}
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["train"].append(train_epoch(model, opt, *train, norm, batch_size, gen, consistency_weight))
        hist["val"].append(evaluate_loss(model, *val, norm))
        if sched is not None:
            sched.step()
        if hist["val"][-1] < hist["best_val"]:
            hist["best_val"], hist["best_epoch"] = hist["val"][-1], ep
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (ep % verbose == 0 or ep == 1):
            print(f"epoch {ep:4d}  train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}"
                  f"  best {hist['best_val']:.4f} @ {hist['best_epoch']}")
    model.load_state_dict(best_state)
    return model, hist


def save_checkpoint(path, model, kind: str, config: dict, in_norm: InputNormaliser,
                    out_norm: ChannelNormaliser, hist: dict) -> None:
    """Save everything needed to use the model later without retraining.

    kind: "deeponet" or "mlp"; config: constructor kwargs (d_u, p, width, ...). The normalisers
    are saved too: predictions are meaningless without the train-split statistics they came from.
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": kind, "config": config, "state_dict": model.state_dict(),
        "in_norm": {"mean": in_norm.mean, "std": in_norm.std},
        "out_norm": {"mean": out_norm.mean, "std": out_norm.std},
        "hist": hist,
    }, path)


def build_model(kind: str, config: dict):
    """Instantiate ``model.DeepONet`` (kind ``"deeponet"``) or ``model.MLPBaseline`` (``"mlp"``) from its config dict."""
    from .model import DeepONet, MLPBaseline
    return {"deeponet": DeepONet, "mlp": MLPBaseline}[kind](**config)


def load_checkpoint(path):
    """Rebuild a saved model. Returns (model in eval mode, in_norm, out_norm, hist)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["kind"], ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return (model, InputNormaliser(**ck["in_norm"]), ChannelNormaliser(**ck["out_norm"]), ck["hist"])
