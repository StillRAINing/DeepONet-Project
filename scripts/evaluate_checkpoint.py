"""Load a saved checkpoint and print its evaluation tables (CLI glue around src/evaluate.py).

    py -3.10 -m scripts.evaluate_checkpoint checkpoints/<name>.pt                 # validation split
    py -3.10 -m scripts.evaluate_checkpoint checkpoints/<name>.pt --split test    # final numbers only
    py -3.10 -m scripts.evaluate_checkpoint checkpoints/<name>.pt --split ood_amp1 --no-physics

The default split is ``val`` so that hyperparameter sweeps never touch the test set. Run ``--split test``
and the OOD sets once, at the end. Normalisers come from the checkpoint (train-split statistics).
Per-sample arrays are saved to results/eval_<checkpoint>_<split>.npz for later figures.
"""
import argparse
from pathlib import Path

import numpy as np

from src.data import DATA_PATH_SMOOTH, T_FINAL, load
from src.evaluate import (CHANNELS, energy_balance_summary, error_vs_time, fraction_below,
                          predict, qdot_consistency, rel_l2_per_channel, rel_l2_per_sample,
                          worst_cases)
from src.train import load_checkpoint, make_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOLS = (0.01, 0.05, 0.10)
SECONDS = (0.5, 1.0, 2.0, 3.0)


def med_p95(x):
    return np.median(x), np.percentile(x, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="val", help="val (default), test, ood, ood_amp1, ood_amp2, ood_smooth")
    ap.add_argument("--n", type=int, default=None, help="evaluate only the first n samples")
    ap.add_argument("--no-physics", action="store_true", help="skip the slower consistency / energy checks")
    args = ap.parse_args()

    model, in_norm, out_norm, hist = load_checkpoint(args.checkpoint)
    d = load(DATA_PATH_SMOOTH)
    if args.n:
        for k in [k for k in d if k.startswith(args.split + "_")]:
            d[k] = d[k][:args.n]
    mode = hist.get("input_mode", "samples")
    u, t, y_norm = make_split(d, args.split, in_norm, out_norm, T_FINAL, mode)
    true = d[f"{args.split}_traj"].astype(np.float64)
    pred = predict(model, u, t, out_norm)
    std = out_norm.std.numpy()
    n = len(true)
    print(f"{Path(args.checkpoint).name} on {args.split} (n = {n}); selected at epoch {hist['best_epoch']}"
          f" with val loss {hist['best_val']:.4f}")

    err = rel_l2_per_sample(pred, true, std)
    m, p = med_p95(err)
    print(f"\nHeadline relative L2 (channels scaled by train std): median {m:.4f}   p95 {p:.4f}   max {err.max():.4f}")
    print("Fraction of trajectories under tolerance: "
          + "   ".join(f"<{tol:.0%}: {f:.1%}" for tol, f in fraction_below(err, TOLS).items()))
    print("Worst 5 samples (index: error): " + ", ".join(f"{i}: {err[i]:.3f}" for i in worst_cases(err, 5)))

    ch = rel_l2_per_channel(pred, true)
    print("\nPer channel (own-scale relative L2)      median     p95")
    for j, name in enumerate(CHANNELS):
        m, p = med_p95(ch[:, j])
        print(f"  {name:6s}                              {m:8.4f}  {p:8.4f}")

    ev = error_vs_time(pred, true, std)
    idx = [int(np.argmin(np.abs(d["t_grid"] - s))) for s in SECONDS]
    print("\nError vs time (fraction of the trajectory's rms)  median / p95")
    for s, i in zip(SECONDS, idx):
        m, p = med_p95(ev[:, i])
        print(f"  t = {d['t_grid'][i]:.1f} s : {m:.4f} / {p:.4f}")

    saved = {"err": err, "err_channel": ch, "err_time": ev}
    if not args.no_physics:
        res = qdot_consistency(model, u, t, out_norm)
        rms_res = np.sqrt((res ** 2).mean(axis=(1, 2)))
        rms_qd = np.sqrt((true[..., 2:] ** 2).mean(axis=(1, 2)))
        cons = rms_res / np.maximum(rms_qd, 1e-12)
        m, p = med_p95(cons)
        print(f"\nqdot vs dq/dt (rms residual / rms true qdot): median {m:.4f}   p95 {p:.4f}")
        eb = energy_balance_summary(pred, true, d[f"{args.split}_params"], d[f"{args.split}_torque"], d["t_grid"])
        print("Energy balance, max over time, as a fraction of the true energy range   median / p95")
        for key, label in (("pred_rel", "prediction"), ("true_rel", "truth (floor)"), ("excess_rel", "excess over floor")):
            m, p = med_p95(eb[key])
            print(f"  {label:18s}: {m:.4f} / {p:.4f}")
        saved.update(consistency=cons, **{f"energy_{k}": v for k, v in eb.items()})

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{Path(args.checkpoint).stem}_{args.split}.npz"
    np.savez_compressed(out, **saved)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
