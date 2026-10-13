"""Train (or reload) a model on the smooth dataset and save it, so it never has to be retrained.

    py -3.10 -m scripts.train_quick --model deeponet --epochs 200     # trains, saves, or reloads
    py -3.10 -m scripts.train_quick --model deeponet --epochs 200 --retrain   # force a fresh run

The checkpoint name encodes the settings, so different runs never overwrite each other:
checkpoints/<model>_p<p>_w<width>_ep<epochs>_lr<lr>_n<n_train>_seed<seed>.pt
(plus _coef, _cosine, _ff<K> (Fourier time features) and _cons<w> (qdot = dq/dt penalty) suffixes
when those options are used, so older names stay valid).
If that file exists, it is loaded and training is skipped unless --retrain is given.
Uses train for fitting and val for checkpoint selection only. Test and OOD are never loaded.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data import DATA_PATH_SMOOTH, N_T, T_FINAL, load
from src.model import n_params
from src.train import (COEF_MODES, ChannelNormaliser, InputNormaliser, build_model, fit,
                       load_checkpoint, make_split, raw_inputs, save_checkpoint)

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def model_config(args, d_u: int) -> dict:
    cfg = ({"d_u": d_u, "p": args.p, "width": args.width} if args.model == "deeponet"
           else {"d_u": d_u, "width": args.width})
    if args.fourier:
        cfg["n_fourier"] = args.fourier   # absent from older configs -> they load with the default 0
    return cfg


def matched_width(args, target: int) -> int:
    """Width (8..1024) whose model has the parameter count closest to ``target``."""
    d_u = 2 + (2 * 2 * COEF_MODES if args.input == "coef" else 2 * N_T)
    best = min(range(8, 1025), key=lambda w: abs(
        n_params(build_model(args.model, model_config(argparse.Namespace(**{**vars(args), "width": w}), d_u))) - target))
    return best


def subset_train(d: dict, n: int) -> dict:
    """The dataset dict with EVERY train_* array cut to its first n samples (val/test/OOD untouched)."""
    return {**d, **{k: v[:n] for k, v in d.items() if k.startswith("train_")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["deeponet", "mlp"], default="deeponet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=7000, help="use the first n training samples")
    ap.add_argument("--p", type=int, default=64, help="DeepONet number of basis functions")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3001)
    ap.add_argument("--input", choices=["samples", "coef"], default="samples",
                    help="branch input: 304 torque samples, or the 26 compact coefficients (baseline W6)")
    ap.add_argument("--schedule", choices=["none", "cosine"], default="none")
    ap.add_argument("--fourier", type=int, default=0,
                    help="number of Fourier time features k pi t (trunk of the DeepONet, input of the MLP); 0 = off")
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="weight of the qdot = dq/dt self-consistency loss term; 0 = off")
    ap.add_argument("--match-params", type=int, default=0,
                    help="pick the width whose parameter count is closest to this (fair comparison, W1)")
    ap.add_argument("--retrain", action="store_true", help="train again even if a checkpoint exists")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    if args.match_params:
        args.width = matched_width(args, args.match_params)

    name = (f"{args.model}_p{args.p}_w{args.width}_ep{args.epochs}_lr{args.lr:g}"
            f"_n{args.n_train}_seed{args.seed}" if args.model == "deeponet" else
            f"mlp_w{args.width}_ep{args.epochs}_lr{args.lr:g}_n{args.n_train}_seed{args.seed}")
    name += ("_coef" if args.input == "coef" else "") + ("_cosine" if args.schedule == "cosine" else "")
    name += (f"_ff{args.fourier}" if args.fourier else "") + (f"_cons{args.consistency:g}" if args.consistency else "")
    path = CKPT_DIR / f"{name}.pt"

    if path.exists() and not args.retrain:
        model, _, _, hist = load_checkpoint(path)
        print(f"loaded {path.name}: {n_params(model):,} params, best val {hist['best_val']:.4f} "
              f"@ epoch {hist['best_epoch']}  (pass --retrain to train again)")
        return

    d = load(DATA_PATH_SMOOTH)
    n = args.n_train
    sub = subset_train(d, n)
    in_norm = InputNormaliser.fit(raw_inputs(sub, "train", args.input))                     # train only
    out_norm = ChannelNormaliser.fit(sub["train_traj"])     # train only
    train = make_split(sub, "train", in_norm, out_norm, T_FINAL, args.input)
    val = make_split(d, "val", in_norm, out_norm, T_FINAL, args.input)

    d_u = train[0].shape[1]
    config = model_config(args, d_u)
    # Seed HERE, just before the weights are created: fit() seeds again, but by then the initial
    # weights already exist, so without this line "seed 3001" fixed only the batch order.
    # (Runs saved before 2026-10-13 ~15:30 have no "init_seeded" entry in hist: unseeded init.)
    torch.manual_seed(args.seed)
    model = build_model(args.model, config)
    print(f"{name}: {n_params(model):,} params, d_u = {d_u}, n_train = {n}")
    t0 = time.perf_counter()
    model, hist = fit(model, train, val, out_norm, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, seed=args.seed, verbose=1,
                      schedule=args.schedule, consistency_weight=args.consistency)
    hist.update(input_mode=args.input, n_params=n_params(model), train_seconds=time.perf_counter() - t0,
                batch_size=args.batch_size, schedule=args.schedule, init_seeded=True,
                n_fourier=args.fourier, consistency_weight=args.consistency)
    print(f"done in {time.perf_counter() - t0:.0f} s; best val {hist['best_val']:.4f} @ epoch {hist['best_epoch']}")
    if not args.no_save:
        save_checkpoint(path, model, args.model, config, in_norm, out_norm, hist)
        print("saved", path)


if __name__ == "__main__":
    main()
