"""Naive reference errors on the VALIDATION split: how much better than "no model" is a trained surrogate?

    py -3.10 -m scripts.naive_baselines

Two baselines, no training, train statistics only:
  * mean trajectory : predict the average training trajectory for every input (ignores the input);
  * 1-NN            : predict the training trajectory whose compact input (m_p, b, Fourier
                      coefficients, z-scored with train statistics) is nearest.
Same headline metric as everywhere else (src/evaluate.rel_l2_per_sample, channels scaled by train std).
"""
import numpy as np

from src.data import DATA_PATH_SMOOTH, load
from src.evaluate import fraction_below, rel_l2_per_channel, rel_l2_per_sample


def compact(d: dict, name: str) -> np.ndarray:
    params = np.asarray(d[f"{name}_params"])
    return np.concatenate([params, np.asarray(d[f"{name}_tau_coef"]).reshape(len(params), -1)], axis=1)


def report(name: str, pred: np.ndarray, true: np.ndarray, std: np.ndarray) -> None:
    e = rel_l2_per_sample(pred, true, std)
    f = fraction_below(e)
    ch = np.median(rel_l2_per_channel(pred, true), axis=0)
    print(f"{name:16s} median {np.median(e):.3f}  p95 {np.percentile(e, 95):.3f}  max {e.max():.3f}  "
          f"under 5% {f[0.05]:.1%}  under 10% {f[0.10]:.1%}  per-channel medians {np.round(ch, 3)}")


def main():
    d = load(DATA_PATH_SMOOTH)
    train, val = d["train_traj"].astype(np.float64), d["val_traj"].astype(np.float64)
    std = train.std(axis=(0, 1))
    report("mean trajectory", np.broadcast_to(train.mean(axis=0), val.shape), val, std)

    xt, xv = compact(d, "train"), compact(d, "val")
    mu, sd = xt.mean(axis=0), xt.std(axis=0)
    xt, xv = (xt - mu) / sd, (xv - mu) / sd
    nearest = np.array([np.argmin(((xt - x) ** 2).sum(axis=1)) for x in xv])
    report("1-NN (compact)", train[nearest], val, std)


if __name__ == "__main__":
    main()
