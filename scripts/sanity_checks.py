"""Pre-flight sanity checks on the real smooth dataset (docs/STUDY_GUIDE.md sections 4 and 5, DESIGN_LOG E).

    py -3.10 -m scripts.sanity_checks                 # everything (~10-15 min on a laptop CPU)
    py -3.10 -m scripts.sanity_checks --only metric float32 substeps energy params leak
    py -3.10 -m scripts.sanity_checks --only overfit shuffle easy lr

Uses the TRAIN and VAL splits only: the test and OOD sets are never loaded into a model here.
Each check prints PASS / FAIL against a criterion stated in its docstring.
Written by the AI on request (AI_LOG row to add).

  metric    relative L2 of truth vs truth = 0, of zeros vs truth = 1
  float32   rounding error of storing trajectories in float32 (D5)
  substeps  RK4 truth at 4 substeps vs 8: the label's own error (the floor a surrogate cannot beat)
  energy    energy-balance residual of the TRUE trajectories ~0 (before trusting it on predictions)
  params    n_params of the final configs (W1) and that matched models really are matched
  leak      normalisers stored in the final checkpoints equal statistics of the train split only
  overfit   both final models overfit 5 real samples to near-zero loss
  shuffle   with labels shuffled across samples, val loss stays at the predict-the-mean level
  easy      zero torque, zero payload (only b varies): the easy sub-problem is learnt well
  lr        1-epoch learning-rate range test for both models
"""
import argparse
import time
from dataclasses import replace

import numpy as np
import torch

from src.data import (DATA_PATH_SMOOTH, N_T, T_FINAL, T_GRID, X0, load, simulate_smooth,
                      smooth_torque_fn, smooth_torque_samples)
from src.dynamics import ArmParams, state_derivative
from src.solvers import rk4_integrate
from src.evaluate import energy_balance_residual, energy_range, predict, rel_l2_per_sample
from src.model import DeepONet, MLPBaseline, n_params
from src.train import (ChannelNormaliser, InputNormaliser, evaluate_loss, fit, load_checkpoint,
                       make_split, raw_inputs, relative_l2_loss)

SEED = 3001
# The four final configurations (results/final_eval_ff6.md), seed 3001.
FINAL = {
    "deeponet/samples": "deeponet_p64_w127_ep200_lr0.001_n7000_seed3001_cosine_ff6",
    "mlp/samples": "mlp_w191_ep200_lr0.001_n7000_seed3001_cosine_ff6",
    "deeponet/coef": "deeponet_p64_w146_ep200_lr0.003_n7000_seed3001_coef_cosine_ff6",
    "mlp/coef": "mlp_w227_ep200_lr0.003_n7000_seed3001_coef_cosine_ff6",
}
D_U = 2 + 2 * N_T


def verdict(ok: bool, msg: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def make_models(d_u: int):
    """The two final architectures (samples input, 6 Fourier time features), fresh weights."""
    torch.manual_seed(SEED)
    return {"deeponet": DeepONet(d_u, p=64, width=127, n_fourier=6),
            "mlp": MLPBaseline(d_u, width=191, n_fourier=6)}


def splits(d, n_train=None, mode="samples"):
    sub = {**d, **{k: v[:n_train] for k, v in d.items() if k.startswith("train_")}} if n_train else d
    in_norm = InputNormaliser.fit(raw_inputs(sub, "train", mode))
    out_norm = ChannelNormaliser.fit(sub["train_traj"])
    return (make_split(sub, "train", in_norm, out_norm, T_FINAL, mode),
            make_split(d, "val", in_norm, out_norm, T_FINAL, mode), out_norm)


# ---------------------------------------------------------------------------------------------------
def check_metric(d):
    """Criterion: truth vs truth = 0 exactly; zeros vs truth = 1 exactly (definition of relative L2)."""
    std = d["train_traj"].std(axis=(0, 1))
    y = d["val_traj"]
    e0, e1 = rel_l2_per_sample(y, y, std), rel_l2_per_sample(np.zeros_like(y), y, std)
    return verdict(e0.max() == 0 and np.allclose(e1, 1.0, atol=1e-12),
                   f"truth vs truth max {e0.max():.1e}; zeros vs truth in [{e1.min():.6f}, {e1.max():.6f}]")


def check_float32(d):
    """Criterion: float32 rounding error < 1e-6, i.e. >= 3 orders below the best surrogate error (~1e-2)."""
    std = d["train_traj"].std(axis=(0, 1))
    y = d["val_traj"].astype(np.float64)
    e = rel_l2_per_sample(y.astype(np.float32).astype(np.float64), y, std)
    return verdict(e.max() < 1e-6, f"float32 round trip: median {np.median(e):.1e}, max {e.max():.1e}")


def check_substeps(d, n=40):
    """Criterion: median label error (4 vs 8 substeps, which bounds the 8-substep error from above
    up to a factor ~16/15) < 1e-3, i.e. well below the surrogate's ~5e-2 median."""
    std = d["train_traj"].std(axis=(0, 1))
    idx = np.random.default_rng(SEED).choice(len(d["train_traj"]), n, replace=False)
    y8 = d["train_traj"][idx]
    y4 = np.stack([simulate_smooth(d["train_params"][i], d["train_tau_coef"][i], substeps=4) for i in idx])
    regen = np.abs(np.stack([simulate_smooth(d["train_params"][i], d["train_tau_coef"][i]) for i in idx[:3]])
                   - y8[:3]).max()
    e = rel_l2_per_sample(y4, y8, std)
    ok = verdict(regen < 1e-10, f"stored labels regenerate bit-for-bit-ish (max diff {regen:.1e})")
    # RK4: e(h) - e(h/2) ~ 15/16 e(h)  ->  error of the 8-substep labels ~ e / 15
    return ok & verdict(np.median(e) < 1e-3,
                        f"4 vs 8 substeps over {n} train samples: median {np.median(e):.1e}, max {e.max():.1e}"
                        f"  -> estimated 8-substep label error ~ median {np.median(e) / 15:.1e}, max {e.max() / 15:.1e}"
                        "  (run scripts.truth_check for the DOP853 comparison per split)")


def check_energy(d, n=200):
    """Criterion: the energy-balance residual of TRUE trajectories is quadrature error, not physics:
    re-simulating the worst sample on a finer output grid cuts it by ~4x per halving (trapezoid, O(h^2)).
    On the 151-point grid this floor reaches ~1 % of the energy range for fast samples (|qdot| ~ 35 rad/s),
    so on predictions only the residual ABOVE the true trajectory's floor is meaningful (what final_eval does)."""
    params, torque, traj = d["val_params"][:n], d["val_torque"][:n], d["val_traj"][:n]
    rel = np.abs(energy_balance_residual(traj, params, torque, T_GRID)).max(axis=1) / energy_range(traj, params)
    print(f"    151-point floor over {n} val samples: median {np.median(rel):.1e}, p95 "
          f"{np.quantile(rel, 0.95):.1e}, max {rel.max():.1e}")
    i = int(np.argmax(rel))
    p = replace(ArmParams(), m_p=float(params[i, 0]), b=float(params[i, 1]))
    tau = smooth_torque_fn(d["val_tau_coef"][i])
    fine = np.linspace(0.0, T_FINAL, 150 * 16 + 1)
    X = rk4_integrate(lambda t, x: state_derivative(t, x, p, tau), X0, fine)
    row = []
    for k in (16, 8, 4, 2):  # 151, 301, 601, 1201 output points
        tg, Xk = fine[::k], X[::k][None]
        r = energy_balance_residual(Xk, params[i:i + 1], smooth_torque_samples(d["val_tau_coef"][i], tg)[None], tg)
        row.append(np.abs(r).max() / energy_range(Xk, params[i:i + 1])[0])
    ratios = np.array(row[:-1]) / np.array(row[1:])
    return verdict(np.all((ratios > 3.0) & (ratios < 5.5)),
                   f"worst sample (val {i}) on 151/301/601/1201 points: " + " ".join(f"{v:.1e}" for v in row)
                   + f"; ratios {' '.join(f'{v:.1f}' for v in ratios)} (expect ~4)")


def check_params(d):
    """Criterion: models compared head-to-head have parameter counts within 2 % of each other (W1)."""
    counts = {}
    for key, name in FINAL.items():
        model, *_ = load_checkpoint(f"checkpoints/{name}.pt")
        counts[key] = n_params(model)
        print(f"    {key:18s} {counts[key]:>9,d}  ({name})")
    lo, hi = min(counts.values()), max(counts.values())
    default = n_params(DeepONet(D_U)), n_params(MLPBaseline(D_U))
    print(f"    for reference, default constructors: DeepONet {default[0]:,}, MLPBaseline {default[1]:,}"
          "  (DESIGN_LOG W1 says 171,396 / 89,220)")
    return verdict((hi - lo) / hi < 0.02, f"final models within {(hi - lo) / hi:.2%} of each other")


def check_leak(d):
    """Criterion: every final checkpoint's stored normaliser equals the statistics of train_* (first
    n_train samples) to float32 precision. If test/val had been used, they would differ."""
    ok = True
    for key, name in FINAL.items():
        _, in_norm, out_norm, hist = load_checkpoint(f"checkpoints/{name}.pt")
        mode = "coef" if "coef" in name else "samples"
        n = int(name.split("_n")[1].split("_")[0])
        sub = {k: v[:n] for k, v in d.items() if k.startswith("train_")}
        ref_in, ref_out = InputNormaliser.fit(raw_inputs(sub, "train", mode)), ChannelNormaliser.fit(sub["train_traj"])
        # and the same stats computed WITH val added must differ, or this check could not detect a leak
        both = {f"train_{k[6:]}": np.concatenate([v, d[f"val_{k[6:]}"]]) for k, v in sub.items()}
        leaky = ChannelNormaliser.fit(both["train_traj"])
        same = (torch.allclose(in_norm.mean, ref_in.mean, rtol=1e-5, atol=1e-6)
                and torch.allclose(in_norm.std, ref_in.std, rtol=1e-5, atol=1e-6)
                and torch.allclose(out_norm.mean, ref_out.mean, rtol=1e-5, atol=1e-6)
                and torch.allclose(out_norm.std, ref_out.std, rtol=1e-5, atol=1e-6))
        detectable = not torch.allclose(out_norm.std, leaky.std, rtol=1e-5, atol=1e-6)
        ok &= verdict(same and detectable, f"{key}: normalisers = train-only statistics "
                                           f"(train+val would differ: {detectable})")
    return ok


def check_overfit(d, n=5, epochs=1500):
    """Criterion: train loss on 5 real samples < 0.02 (2 % relative L2) for both models. A model that
    cannot memorise 5 trajectories has a bug, not a hyperparameter problem."""
    sub = {**d, **{k: v[:n] for k, v in d.items() if k.startswith("train_")}}
    train, _, out_norm = splits(sub)
    ok = True
    for name, model in make_models(D_U).items():
        t0 = time.perf_counter()
        model, hist = fit(model, train, train, out_norm, epochs=epochs, lr=1e-3, batch_size=n, verbose=0,
                          schedule="cosine")
        final = evaluate_loss(model, *train, out_norm)
        ok &= verdict(final < 0.02, f"{name}: 5-sample loss {hist['train'][0]:.3f} -> {final:.4f} "
                                    f"in {epochs} epochs, {time.perf_counter() - t0:.0f} s")
    return ok


def check_shuffle(d, n_train=1000, epochs=30):
    """Criterion: with targets permuted across samples, best val loss stays within 10 % of the
    predict-the-train-mean baseline, while the same budget on true labels gets clearly below it.
    If shuffled labels beat the mean, information is leaking from somewhere other than the input."""
    train, val, out_norm = splits(d, n_train)
    u, t, y = train
    mean_pred = y.mean(dim=0, keepdim=True).expand(len(val[2]), -1, -1)
    baseline = relative_l2_loss(mean_pred, val[2], out_norm).item()
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SEED))
    ok = True
    for name in ("deeponet", "mlp"):
        res = {}
        for label, yy in (("shuffled", y[perm]), ("true", y)):
            model = make_models(D_U)[name]
            _, hist = fit(model, (u, t, yy), val, out_norm, epochs=epochs, lr=1e-3, verbose=0)
            res[label] = hist["best_val"]
        ok &= verdict(res["shuffled"] > 0.9 * baseline and res["true"] < 0.8 * baseline,
                      f"{name}: val loss  predict-mean {baseline:.3f} | shuffled labels {res['shuffled']:.3f} "
                      f"| true labels {res['true']:.3f}  ({n_train} train, {epochs} epochs)")
    return ok


def check_easy(n_train=200, n_val=50, epochs=300):
    """Criterion: zero torque, zero payload, b in [0.05, 0.5]: the arm just falls from horizontal and
    swings. A 1-parameter family; both models should reach median val error < 2 %."""
    rng = np.random.default_rng(SEED)
    b = rng.uniform(0.05, 0.5, n_train + n_val)
    params = np.column_stack([np.zeros_like(b), b])
    coef = np.zeros((len(b), 2, 2, 6))
    traj = np.stack([simulate_smooth(p, c) for p, c in zip(params, coef)])
    torque = np.stack([smooth_torque_samples(c) for c in coef])
    d = {"t_grid": T_GRID}
    for s, sl in (("train", slice(0, n_train)), ("val", slice(n_train, None))):
        d.update({f"{s}_params": params[sl], f"{s}_torque": torque[sl], f"{s}_traj": traj[sl]})
    train, val, out_norm = splits(d)
    std = traj[:n_train].std(axis=(0, 1))
    ok = True
    for name, model in make_models(D_U).items():
        model, _ = fit(model, train, val, out_norm, epochs=epochs, lr=1e-3, batch_size=50, verbose=0,
                       schedule="cosine")
        e = rel_l2_per_sample(predict(model, val[0], val[1], out_norm), traj[n_train:], std)
        ok &= verdict(np.median(e) < 0.02, f"{name}: zero-torque/zero-payload val error median "
                                           f"{np.median(e):.4f}, max {e.max():.4f}")
    return ok


def check_lr(d, n_train=2000, lrs=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """Not pass/fail: val loss after ONE epoch from the same initial weights, per learning rate.
    Read it as: too small -> barely moves; too large -> unstable / worse. A good range is just below
    the rate where the curve turns back up."""
    train, val, out_norm = splits(d, n_train)
    for name in ("deeponet", "mlp"):
        row = []
        for lr in lrs:
            model = make_models(D_U)[name]
            _, hist = fit(model, train, val, out_norm, epochs=1, lr=lr, verbose=0)
            row.append(hist["val"][0])
        best = lrs[int(np.nanargmin(row))]
        print(f"    {name:8s} " + "  ".join(f"{lr:g}:{v:.3f}" for lr, v in zip(lrs, row)) + f"   best 1-epoch lr {best:g}")
    return True


CHECKS = {"metric": check_metric, "float32": check_float32, "substeps": check_substeps,
          "energy": check_energy, "params": check_params, "leak": check_leak, "overfit": check_overfit,
          "shuffle": check_shuffle, "easy": lambda d: check_easy(), "lr": check_lr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CHECKS), default=list(CHECKS))
    args = ap.parse_args()
    keys = ("train_", "val_", "t_grid")
    d = {k: v for k, v in load(DATA_PATH_SMOOTH).items() if k.startswith(keys)}  # no test / OOD
    results = {}
    for name in args.only:
        print(f"\n== {name}: {CHECKS[name].__doc__.strip().splitlines()[0] if CHECKS[name].__doc__ else ''}")
        t0 = time.perf_counter()
        results[name] = CHECKS[name](d)
        print(f"  ({time.perf_counter() - t0:.0f} s)")
    print("\nsummary: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))


if __name__ == "__main__":
    main()
