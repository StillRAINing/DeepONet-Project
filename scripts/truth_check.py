"""How accurate are the stored RK4 labels? Compares them with a tight DOP853 reference, per split.

    py -3.10 -m scripts.truth_check            # 20 random samples from each split
    py -3.10 -m scripts.truth_check --n 50

No model is involved, so checking the test / OOD LABELS here does not leak anything into model
selection. Metric: the headline relative L2 (channels scaled by the train std), so the numbers are
directly comparable with surrogate errors: a surrogate cannot be judged below this floor.
Extends DESIGN_LOG A8, which checked training samples only.
"""
import argparse

import numpy as np

from src.benchmark import solve_ivp_one
from src.data import DATA_PATH_SMOOTH, SEED, T_GRID, X0, load
from src.evaluate import rel_l2_per_sample

SPLITS = ("train", "val", "test", "ood", "ood_amp1", "ood_amp2", "ood_smooth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    d = load(DATA_PATH_SMOOTH)
    std = d["train_traj"].std(axis=(0, 1))
    rng = np.random.default_rng(SEED)
    print(f"{'split':12s} {'n':>4s} {'median':>10s} {'max':>10s}   (stored RK4 vs DOP853 rtol 1e-10)")
    for s in SPLITS:
        if f"{s}_traj" not in d:
            continue
        idx = rng.choice(len(d[f"{s}_traj"]), size=min(args.n, len(d[f"{s}_traj"])), replace=False)
        ref = np.stack([solve_ivp_one(d[f"{s}_params"][i], d[f"{s}_tau_coef"][i], T_GRID, X0,
                                      rtol=1e-10, atol=1e-12, method="DOP853") for i in idx])
        e = rel_l2_per_sample(d[f"{s}_traj"][idx], ref, std)
        print(f"{s:12s} {len(idx):4d} {np.median(e):10.2e} {e.max():10.2e}")


if __name__ == "__main__":
    main()
