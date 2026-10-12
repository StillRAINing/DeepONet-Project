"""Check the smooth random torque family before generating the full dataset (docs/DESIGN_LOG.md, A6/A7).

Reports, for a few hundred samples at the training settings:
  * the share of torque variance beyond the K retained Fourier modes (analytic, from the spectrum),
  * that every sample's torque peak equals its drawn peak A (the amplitude scheme),
  * motion magnitude and spin statistics (fraction with max |q| past pi and 2 pi, 95th percentile),
  * the effective dimension of the sampled torques (SVD), for comparison with the sine family,
  * the same motion statistics for each out-of-range set.

Run: python -m scripts.smooth_family_check [--n 300] [--plot] [--workers 4]
"""
import argparse

import numpy as np

from src.data import ELL, K_MODES, OOD_SMOOTH, SMOOTH_RANGES, T_GRID, make_smooth_set, spectral_std


def tail_variance_share(ell: float, k_modes: int, k_max: int = 400) -> float:
    """Share of total coefficient variance in modes k_modes+1..k_max (k_max stands in for infinity)."""
    var = spectral_std(ell, k_max) ** 2
    return float(var[k_modes:].sum() / var.sum())


def spin_stats(traj: np.ndarray) -> dict:
    q = np.abs(traj[..., :2]).max(axis=(1, 2))
    return {
        "median max|q|": float(np.median(q)),
        "frac > pi": float((q > np.pi).mean()),
        "frac > 2pi": float((q > 2 * np.pi).mean()),
        "p95 max|q|": float(np.percentile(q, 95)),
        "max |q|": float(q.max()),
        "max |qdot|": float(np.abs(traj[..., 2:]).max()),
        "finite": bool(np.isfinite(traj).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"tail variance beyond K={K_MODES} at ell={ELL}: {tail_variance_share(ELL, K_MODES):.2e}")
    print(f"tail variance beyond K=13 at ell=0.1: {tail_variance_share(0.1, 13):.2e}")

    rng = np.random.default_rng(0)
    d = make_smooth_set(args.n, rng, workers=args.workers)
    peaks = np.abs(d["torque"]).max(axis=2)  # on the 151-point grid; the scheme fixes the peak on a finer grid
    print(f"torque peak on the 151-point grid: min {peaks.min():.2f}, max {peaks.max():.2f} (cap {SMOOTH_RANGES['A'][1]})")

    T = d["torque"].reshape(args.n, -1)
    sv = np.linalg.svd(T - T.mean(0), compute_uv=False)
    cum = np.cumsum(sv**2) / np.sum(sv**2)
    print(f"sampled-torque dimension for 99% / 99.9% of variance: {np.searchsorted(cum, 0.99) + 1} / {np.searchsorted(cum, 0.999) + 1}")

    print("\nmotion statistics")
    rows = {"in-range (training settings)": spin_stats(d["traj"])}
    for name, cfg in OOD_SMOOTH.items():
        cfg = {"ell": ELL, "k_modes": K_MODES, **cfg}
        rows[name] = spin_stats(make_smooth_set(min(args.n, 100), rng, workers=args.workers, **cfg)["traj"])
    for name, s in rows.items():
        print(f"  {name:30s}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, grid = "#1f2937", "#d1d5db"
        hue = "#2a6fb0"
        plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink,
                             "ytick.color": ink, "text.color": ink, "font.size": 9})
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

        for i in range(10):
            ax[0].plot(T_GRID, d["torque"][i, 0], color=hue, lw=0.9, alpha=0.7)
        ax[0].set(title="Joint-1 torque, 10 samples", xlabel="t (s)", ylabel="torque (N m)")

        k = np.arange(1, 16)
        var = spectral_std(ELL, 15) ** 2
        ax[1].semilogy(k, var / var[0], color=hue, lw=1.5, marker="o", ms=3)
        ax[1].axvline(K_MODES + 0.5, color=ink, lw=0.8, ls="--")
        ax[1].text(K_MODES + 0.8, 0.3, f"keep K = {K_MODES}", fontsize=8)
        ax[1].set(title=f"Coefficient variance by mode (ell = {ELL})", xlabel="mode k",
                  ylabel="variance / variance of mode 1", ylim=(1e-8, 2))

        q = np.abs(d["traj"][..., :2]).max(axis=(1, 2))
        ax[2].hist(q, bins=np.arange(0, 46, 1.5), color=hue, edgecolor="white", linewidth=0.6)
        ax[2].set_yscale("log")
        for x, lab, ha in ((np.pi, "pi", "right"), (2 * np.pi, "2 pi", "left")):
            ax[2].axvline(x, color=ink, lw=0.8, ls="--")
            ax[2].text(x + (0.4 if ha == "left" else -0.4), 150, lab, fontsize=8, ha=ha)
        ax[2].set(title="Largest |q| per trajectory", xlabel="rad", ylabel="count (log)")

        for a in ax:
            a.grid(axis="y", color=grid, lw=0.5)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig("results/smooth_family_check.png", dpi=140)
        print("saved results/smooth_family_check.png")


if __name__ == "__main__":
    main()
