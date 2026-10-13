"""DeepONet and MLP baseline models (PyTorch).

Both models share one interface so train/evaluate/benchmark can treat them alike:

    model(u, t) -> (B, n_t, 4)

    u: (B, d_u)           branch input [m_p, b, tau_1 samples, tau_2 samples], see ``branch_input``
    t: (n_t,) or (B, n_t) query times (same normalisation convention as the training code)

Output channels are [q1, q2, q1dot, q2dot]. Inputs are expected to be normalised
by the caller (e.g. to [0, 1] or zero mean / unit variance).
"""
import numpy as np
import torch
from torch import nn

N_OUT = 4  # q1, q2, q1dot, q2dot


def branch_input(params, torque) -> torch.Tensor:
    """Flatten (params (B, 2), torque (B, 2, N_T)) into the branch input (B, 2 + 2 N_T)."""
    params = torch.as_tensor(np.asarray(params), dtype=torch.float32)
    torque = torch.as_tensor(np.asarray(torque), dtype=torch.float32)
    return torch.cat([params, torque.flatten(1)], dim=1)


def mlp(d_in: int, width: int, depth: int, d_out: int, act=nn.Tanh) -> nn.Sequential:
    """``depth`` hidden layers of size ``width`` followed by a linear output layer."""
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _as_time_batch(t: torch.Tensor, batch: int) -> torch.Tensor:
    """Return t as (B, n_t, 1)."""
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(batch, -1)
    if t.ndim != 2 or t.shape[0] != batch:
        raise ValueError(f"t must have shape (n_t,) or (B, n_t); got {tuple(t.shape)}")
    return t.unsqueeze(-1)


def time_features(t: torch.Tensor, n_fourier: int) -> torch.Tensor:
    """(B, n_t, 1) -> (B, n_t, 1 + 2 n_fourier): [t, sin(k pi t), cos(k pi t)] for k = 1..n_fourier.

    t is the rescaled time tau = t / T in [0, 1]. The frequencies k pi are the harmonics of the
    smooth torque family (period 2T, so mode k has angular frequency k pi in tau), a fixed prior
    rather than learnt. Without them a tanh MLP on a bare scalar t learns oscillation slowly
    (spectral bias). ``n_fourier = 0`` returns t unchanged.
    """
    if n_fourier == 0:
        return t
    w = torch.pi * torch.arange(1, n_fourier + 1, dtype=t.dtype, device=t.device)   # (K,)
    ang = t * w                                                                      # (B, n_t, K)
    return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=-1)


class DeepONet(nn.Module):
    """Unstacked DeepONet with one set of p basis functions per output channel.

    branch: u -> (N_OUT, p) coefficients
    trunk:  t -> (N_OUT, p) basis functions
    output_k(u, t) = sum_i branch_{k,i}(u) trunk_{k,i}(t) + bias_k
    """

    def __init__(self, d_u: int, p: int = 64, width: int = 128,
                 branch_depth: int = 3, trunk_depth: int = 3, n_fourier: int = 0):
        super().__init__()
        self.d_u, self.p, self.n_fourier = d_u, p, n_fourier
        self.branch = mlp(d_u, width, branch_depth, N_OUT * p)
        self.trunk = mlp(1 + 2 * n_fourier, width, trunk_depth, N_OUT * p)
        self.bias = nn.Parameter(torch.zeros(N_OUT))

    def forward(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if u.ndim != 2 or u.shape[1] != self.d_u:
            raise ValueError(f"u must have shape (B, {self.d_u}); got {tuple(u.shape)}")
        B = u.shape[0]
        b = self.branch(u).view(B, N_OUT, self.p)                  # (B, 4, p)
        tt = _as_time_batch(t, B)
        k = self.trunk(time_features(tt, self.n_fourier)).view(B, tt.shape[1], N_OUT, self.p)  # (B, n_t, 4, p)
        return torch.einsum("bkp,btkp->btk", b, k) + self.bias


class MLPBaseline(nn.Module):
    """Plain MLP on the concatenated input [u, t] -> 4 outputs, one row per (sample, time)."""

    def __init__(self, d_u: int, width: int = 128, depth: int = 4, n_fourier: int = 0):
        super().__init__()
        self.d_u, self.n_fourier = d_u, n_fourier
        self.net = mlp(d_u + 1 + 2 * n_fourier, width, depth, N_OUT)

    def forward(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if u.ndim != 2 or u.shape[1] != self.d_u:
            raise ValueError(f"u must have shape (B, {self.d_u}); got {tuple(u.shape)}")
        B = u.shape[0]
        tt = time_features(_as_time_batch(t, B), self.n_fourier)
        x = torch.cat([u.unsqueeze(1).expand(-1, tt.shape[1], -1), tt], dim=-1)
        return self.net(x)


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
