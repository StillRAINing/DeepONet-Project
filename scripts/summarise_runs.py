"""Evaluate every checkpoint on the VALIDATION split and write one row per run to results/runs.csv.

    py -3.10 -m scripts.summarise_runs            # all checkpoints, prints a markdown table too

This is the experiment log: it keeps the numbers of every run (including the earlier constant-lr
ones) so later changes (cosine schedule, more data) can be compared like for like. It never
touches the test or OOD splits: those are for the final evaluation only.
"""
import csv
import re
from pathlib import Path

import numpy as np

from src.data import DATA_PATH_SMOOTH, T_FINAL, load
from src.evaluate import fraction_below, predict, qdot_consistency, rel_l2_per_channel, rel_l2_per_sample
from src.model import n_params
from src.train import load_checkpoint, make_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "runs.csv"
OUT_AGG = ROOT / "results" / "runs_agg.csv"
FIELDS = ["run", "model", "input", "width", "params", "lr", "schedule", "n_fourier", "cons_weight", "epochs",
          "n_train", "seed", "best_epoch", "train_at_best", "val_loss", "median", "p95", "max",
          "under_1pct", "under_5pct", "under_10pct", "q1", "q2", "q1dot", "q2dot", "consistency",
          "train_s", "init_seeded"]


def from_name(stem: str, key: str, cast=float):
    m = re.search(rf"_{key}([0-9.e-]+?)(?=_|$)", stem)
    return cast(m.group(1)) if m else None


def summarise(path: Path, d: dict) -> dict:
    model, in_norm, out_norm, h = load_checkpoint(path)
    mode = h.get("input_mode", "samples")
    u, t, _ = make_split(d, "val", in_norm, out_norm, T_FINAL, mode)
    true = d["val_traj"].astype(np.float64)
    pred = predict(model, u, t, out_norm)
    err = rel_l2_per_sample(pred, true, out_norm.std.numpy())
    ch = np.median(rel_l2_per_channel(pred, true), axis=0)
    res = qdot_consistency(model, u, t, out_norm)
    cons = np.sqrt((res ** 2).mean(axis=(1, 2))) / np.sqrt((true[..., 2:] ** 2).mean(axis=(1, 2)))
    frac = fraction_below(err)
    stem, b = path.stem, h["best_epoch"]
    return {
        "run": stem, "model": stem.split("_")[0], "input": mode,
        "width": from_name(stem, "w", int), "params": n_params(model),
        "lr": from_name(stem, "lr"), "schedule": h.get("schedule", "none"),
        "n_fourier": from_name(stem, "ff", int) or 0, "cons_weight": from_name(stem, "cons") or 0.0,
        "epochs": len(h["train"]),
        "n_train": from_name(stem, "n", int), "seed": from_name(stem, "seed", int),
        "best_epoch": b, "train_at_best": round(h["train"][b - 1], 4), "val_loss": round(h["best_val"], 4),
        "median": round(np.median(err), 4), "p95": round(np.percentile(err, 95), 4), "max": round(err.max(), 4),
        "under_1pct": round(frac[0.01], 4), "under_5pct": round(frac[0.05], 4), "under_10pct": round(frac[0.10], 4),
        "q1": round(ch[0], 4), "q2": round(ch[1], 4), "q1dot": round(ch[2], 4), "q2dot": round(ch[3], 4),
        "consistency": round(np.median(cons), 4),
        "train_s": round(h["train_seconds"]) if "train_seconds" in h else "",
        # False: weights were initialised before the seed was set, so the run is not exactly reproducible
        "init_seeded": bool(h.get("init_seeded", False)),
    }


AGG_KEY = ("model", "input", "params", "lr", "schedule", "n_fourier", "cons_weight", "n_train")
AGG_METRICS = ("val_loss", "median", "p95", "under_5pct", "under_10pct", "q2", "consistency")


def aggregate(rows: list) -> list:
    """One row per setting (everything but the seed): n_seeds and mean / std of each metric.

    std is the sample std (ddof 1) and is empty for a setting with a single seed. Reading it:
    a difference between two settings smaller than their stds is not yet a difference.
    """
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in AGG_KEY), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(AGG_KEY, key), n_seeds=len(rs), seeds=" ".join(str(r["seed"]) for r in rs))
        for m in AGG_METRICS:
            v = np.array([r[m] for r in rs], dtype=float)
            row[f"{m}_mean"] = round(v.mean(), 4)
            row[f"{m}_std"] = round(v.std(ddof=1), 4) if len(v) > 1 else ""
        out.append(row)
    return sorted(out, key=lambda r: r["val_loss_mean"])


def write_aggregate(rows: list) -> None:
    agg = aggregate(rows)
    fields = [*AGG_KEY, "n_seeds", "seeds", *[f"{m}_{s}" for m in AGG_METRICS for s in ("mean", "std")]]
    with open(OUT_AGG, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(agg)
    print("\nBy setting, best val loss first (mean +- std over seeds):")
    print("| model | input | lr | sched | ff | cons | n_train | seeds | val_loss | median | p95 | under_10pct |\n|" + "---|" * 12)
    for r in agg:
        cell = lambda m: (f"{r[m + '_mean']:.4f} +- {r[m + '_std']:.4f}" if r[m + "_std"] != "" else f"{r[m + '_mean']:.4f}")
        print(f"| {r['model']} | {r['input']} | {r['lr']:g} | {r['schedule']} | {r['n_fourier']} | {r['cons_weight']:g} "
              f"| {r['n_train']} | {r['n_seeds']} | {cell('val_loss')} | {cell('median')} | {cell('p95')} | {cell('under_10pct')} |")
    print(f"wrote {OUT_AGG}")


def main():
    d = load(DATA_PATH_SMOOTH)
    rows = [summarise(p, d) for p in sorted((ROOT / "checkpoints").glob("*.pt"))]
    rows.sort(key=lambda r: (r["model"], r["input"], r["schedule"], r["lr"], r["n_fourier"], r["cons_weight"],
                             r["n_train"], r["seed"]))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    cols = ["model", "input", "params", "lr", "schedule", "n_fourier", "cons_weight", "best_epoch", "train_at_best",
            "val_loss", "median", "p95", "under_5pct", "under_10pct", "q2", "consistency"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\nwrote {OUT} ({len(rows)} runs)")
    write_aggregate(rows)


if __name__ == "__main__":
    main()
