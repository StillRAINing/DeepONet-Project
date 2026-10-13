"""Query-time generalisation: error at the training times vs at times BETWEEN them (validation split).

    py -3.10 -m scripts.offgrid_eval checkpoints/a.pt checkpoints/b.pt [--n 300]

Both models take the time t as a continuous input, so both can be asked about any t. Training only
ever used the 151 grid times (spacing 0.02 s). Here the truth is recomputed with RK4 on a grid of 301
times (spacing 0.01 s, 4 substeps = the same step size as the stored labels), so the EVEN points are
the training times and the ODD points are midpoints the networks have never been asked about.

Reading: a ratio (odd error / even error) near 1 means the model interpolates in time as well as it
fits; well above 1 means it memorised the grid. This is NOT a test of DeepONet vs MLP specifically
(both interpolate in t); it checks that the 'continuous-time surrogate' claim is earned. Validation
split only, no training.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.data import DATA_PATH_SMOOTH, T_FINAL, X0, load, smooth_torque_fn
from src.dynamics import ArmParams, state_derivative
from src.evaluate import predict, rel_l2_per_sample
from src.model import n_params
from src.solvers import rk4_integrate
from src.train import load_checkpoint, raw_inputs

FINE_N = 301      # 2 x 150 + 1 times: even indices = training grid, odd = midpoints
SUBSTEPS = 4      # 0.01 s / 4 = 0.0025 s, the step of the stored labels (0.02 s / 8)


def fine_truth(params, coef) -> np.ndarray:
    """RK4 truth for one sample on the 301-point grid. Returns (301, 4)."""
    p = replace(ArmParams(), m_p=float(params[0]), b=float(params[1]))
    tau = smooth_torque_fn(coef)
    f = lambda t, x: state_derivative(t, x, p, tau)
    fine = np.linspace(0.0, T_FINAL, (FINE_N - 1) * SUBSTEPS + 1)
    return rk4_integrate(f, X0, fine)[::SUBSTEPS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=300, help="number of validation samples (first n)")
    args = ap.parse_args()

    d = load(DATA_PATH_SMOOTH)
    n = args.n
    truth = np.stack([fine_truth(d["val_params"][i], d["val_tau_coef"][i]) for i in range(n)])  # (n, 301, 4)
    t_fine = torch.linspace(0.0, 1.0, FINE_N)
    # sanity: the even points must reproduce the stored labels
    stored = d["val_traj"][:n]
    assert np.allclose(truth[:, ::2], stored, atol=1e-6 * np.abs(stored).max()), "fine truth != stored labels"

    print(f"{n} val samples; error = per-sample relative L2, channels scaled by train std\n")
    print("| checkpoint | params | on-grid median | off-grid median | ratio | on-grid p95 | off-grid p95 |")
    print("|---|---|---|---|---|---|---|")
    for path in args.checkpoints:
        model, in_norm, out_norm, h = load_checkpoint(path)
        mode = h.get("input_mode", "samples")
        u = in_norm.encode(torch.as_tensor(raw_inputs({k: v[:n] if k.startswith("val_") else v
                                                       for k, v in d.items()}, "val", mode), dtype=torch.float32))
        pred = predict(model, u, t_fine, out_norm)
        s = out_norm.std.numpy()
        on = rel_l2_per_sample(pred[:, ::2], truth[:, ::2], s)
        off = rel_l2_per_sample(pred[:, 1::2], truth[:, 1::2], s)
        print(f"| {path.stem} | {n_params(model):,} | {np.median(on):.4f} | {np.median(off):.4f} "
              f"| {np.median(off) / np.median(on):.3f} | {np.percentile(on, 95):.4f} | {np.percentile(off, 95):.4f} |")


if __name__ == "__main__":
    main()
