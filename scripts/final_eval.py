"""Selection-rule step 3 (docs/EXPERIMENTS.md): evaluate the 12 final checkpoints ONCE on test and OOD.

    py -3.10 -m scripts.final_eval            # the 12 rule-selected checkpoints
    py -3.10 -m scripts.final_eval --ff6      # E4 follow-up: the same settings with Fourier time features
    py -3.10 -m scripts.final_eval --ff6cons  # E4 follow-up: Fourier features + consistency penalty (samples input)

The four settings were fixed from validation results before this script was run, and each was trained
with seeds 3001, 1 and 2. Nothing may be re-tuned after reading this output; if a setting is changed
later, the test numbers below are no longer clean and the report must say so.
Writes results/final_eval.csv (one row per checkpoint and split) and results/final_eval.md (mean +- std
over the three seeds).
"""
import csv
from pathlib import Path

import numpy as np

from src.data import DATA_PATH_SMOOTH, T_FINAL, load
from src.evaluate import (energy_balance_summary, fraction_below, predict, qdot_consistency,
                          rel_l2_per_channel, rel_l2_per_sample)
from src.train import load_checkpoint, make_split

ROOT = Path(__file__).resolve().parent.parent
FINAL = {  # (model, input) -> checkpoint stem without the seed part; from EXPERIMENTS.md "Selected settings"
    ("mlp", "coef"): "mlp_w229_ep200_lr0.003_n7000_seed{s}_coef_cosine",
    ("mlp", "samples"): "mlp_w192_ep200_lr0.001_n7000_seed{s}_cosine",
    ("deeponet", "coef"): "deeponet_p64_w147_ep200_lr0.003_n7000_seed{s}_coef_cosine",
    ("deeponet", "samples"): "deeponet_p64_w128_ep200_lr0.001_n7000_seed{s}_cosine",
}
FINAL_FF6 = {  # same four settings retrained with --fourier 6 (E4 follow-up); chosen/compared on validation only
    ("mlp", "coef"): "mlp_w227_ep200_lr0.003_n7000_seed{s}_coef_cosine_ff6",
    ("mlp", "samples"): "mlp_w191_ep200_lr0.001_n7000_seed{s}_cosine_ff6",
    ("deeponet", "coef"): "deeponet_p64_w146_ep200_lr0.003_n7000_seed{s}_coef_cosine_ff6",
    ("deeponet", "samples"): "deeponet_p64_w127_ep200_lr0.001_n7000_seed{s}_cosine_ff6",
}
FINAL_FF6CONS = {  # ff6 + qdot = dq/dt penalty (weight 0.1); samples input only (E4 follow-up, validation-chosen)
    ("mlp", "samples"): "mlp_w191_ep200_lr0.001_n7000_seed{s}_cosine_ff6_cons0.1",
    ("deeponet", "samples"): "deeponet_p64_w127_ep200_lr0.001_n7000_seed{s}_cosine_ff6_cons0.1",
}
VARIANTS = {"": FINAL, "ff6": FINAL_FF6, "ff6cons": FINAL_FF6CONS}
SEEDS = (3001, 1, 2)
SPLITS = ("test", "ood", "ood_amp1", "ood_amp2", "ood_smooth")
METRICS = ("median", "p95", "max", "under_1pct", "under_5pct", "under_10pct",
           "q1", "q2", "q1dot", "q2dot", "consistency", "energy_excess_rel")


def evaluate(path, d, split):
    model, in_norm, out_norm, h = load_checkpoint(path)
    u, t, _ = make_split(d, split, in_norm, out_norm, T_FINAL, h.get("input_mode", "samples"))
    true = d[f"{split}_traj"].astype(np.float64)
    pred = predict(model, u, t, out_norm)
    err = rel_l2_per_sample(pred, true, out_norm.std.numpy())
    frac = fraction_below(err)
    ch = np.median(rel_l2_per_channel(pred, true), axis=0)
    res = qdot_consistency(model, u, t, out_norm)
    cons = np.sqrt((res ** 2).mean(axis=(1, 2))) / np.sqrt((true[..., 2:] ** 2).mean(axis=(1, 2)))
    row = {"median": np.median(err), "p95": np.percentile(err, 95), "max": err.max(),
           "under_1pct": frac[0.01], "under_5pct": frac[0.05], "under_10pct": frac[0.10],
           "q1": ch[0], "q2": ch[1], "q1dot": ch[2], "q2dot": ch[3], "consistency": np.median(cons)}
    if split == "test":  # energy balance is slow (a Python loop over every state); test only
        eb = energy_balance_summary(pred, true, d[f"{split}_params"], d[f"{split}_torque"], d["t_grid"])
        row["energy_excess_rel"] = np.median(eb["excess_rel"])
    else:
        row["energy_excess_rel"] = np.nan
    return row


def main(variant=""):
    """variant "" = the rule-selected models; "ff6" / "ff6cons" = E4 follow-ups, written to
    results/final_eval_<variant>.*; the original files are left untouched."""
    models = VARIANTS[variant]
    suffix = f"_{variant}" if variant else ""
    d = load(DATA_PATH_SMOOTH)
    rows = []
    for (model, inp), stem in models.items():
        for s in SEEDS:
            path = ROOT / "checkpoints" / f"{stem.format(s=s)}.pt"
            for split in SPLITS:
                r = evaluate(path, d, split)
                rows.append({"model": model, "input": inp, "seed": s, "split": split, "n": len(d[f"{split}_traj"]),
                             **{k: round(float(v), 4) for k, v in r.items()}})
                print(f"{model:8s} {inp:7s} seed {s:4d} {split:10s} median {r['median']:.4f}  p95 {r['p95']:.4f}")
    out = ROOT / "results" / f"final_eval{suffix}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    title = ("# Final evaluation: test and OOD (selection-rule step 3)" if not variant else
             f"# E4 follow-up `{variant}`: test and OOD (evaluated after the selection rule; not a re-selection)")
    lines = [title, "",
             "Generated once by `scripts/final_eval.py`. Mean +- sample std over seeds 3001, 1, 2. "
             "Headline error: per-sample relative L2, channels scaled by the train std. "
             "`q2`: median own-scale relative L2 of q2. `consistency`: median rms(dq/dt - qdot) / rms(true qdot). "
             "`energy`: median excess energy-balance residual over the RK4 floor, as a fraction of the "
             "trajectory's energy range (test only).", ""]
    for split in SPLITS:
        n = next(r["n"] for r in rows if r["split"] == split)
        lines += [f"## {split} (n = {n})", "",
                  "| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency"
                  + (" | energy |" if split == "test" else " |"),
                  "|---|---|---|---|---|---|---|---|---" + ("|---|" if split == "test" else "|")]
        for (model, inp) in models:
            sel = [r for r in rows if r["model"] == model and r["input"] == inp and r["split"] == split]
            def ms(k, pct=False):
                v = np.array([r[k] for r in sel]) * (100 if pct else 1)
                return f"{v.mean():.1f} +- {v.std(ddof=1):.1f} %" if pct else f"{v.mean():.4f} +- {v.std(ddof=1):.4f}"
            cells = [model, inp, ms("median"), ms("p95"), ms("max"), ms("under_5pct", True),
                     ms("under_10pct", True), ms("q2"), ms("consistency")] + ([ms("energy_excess_rel")] if split == "test" else [])
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    (ROOT / "results" / f"final_eval{suffix}.md").write_text("\n".join(lines), encoding="utf8")
    print("wrote", out, "and", out.with_suffix(".md"))


if __name__ == "__main__":
    import sys
    main("ff6cons" if "--ff6cons" in sys.argv else "ff6" if "--ff6" in sys.argv else "")
