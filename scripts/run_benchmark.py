"""Final speed / memory / FLOP benchmark of the four selected models (seed 3001).

    py -3.10 -m scripts.run_benchmark          # close other heavy programs first (idle machine)

For each selected checkpoint (same settings as ``scripts/final_eval.py``) it measures, on the smooth
family VALIDATION split, the cost of one trajectory through the surrogate, and compares it with
vectorised RK4 (the truth settings) and ``solve_ivp`` at several tolerances, including the tolerance
whose median error is closest to the surrogate's. It also reports parameters, FLOPs, memory, a
cProfile of the scalar solver and the break-even number of queries (data generation + training
against the per-query saving). Writes results/benchmark.md and results/benchmark.csv.
"""
import cProfile
import csv
import io
import platform
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from src import benchmark as bm
from src.data import DATA_PATH_SMOOTH, N_T, T_GRID, X0, load, simulate_smooth
from src.data import T_FINAL
from src.train import load_checkpoint, raw_inputs

from .final_eval import FINAL

ROOT = Path(__file__).resolve().parent.parent
SEED = 3001
N_BATCH = 64         # trajectories in the timed batch
N_TOTAL = 10_000     # size of the smooth dataset the models were trained on (data generation cost)
BATCHES = (1, 8, 64, 300)
RTOLS = (1e-1, 8e-2, 6e-2, 5e-2, 4e-2, 3e-2, 1e-2, 1e-3, 1e-4, 1e-6)  # dense near the surrogate's error (~0.1)


def flops_for(kind: str, cfg: dict, d_u: int, hist: dict) -> int:
    """FLOPs for one trajectory (151 query times) from the architecture in the checkpoint."""
    if kind == "deeponet":
        return bm.deeponet_flops(d_u, N_T, cfg["p"], cfg["width"], cfg.get("branch_depth", 3),
                                 cfg.get("trunk_depth", 3))
    return bm.mlp_baseline_flops(d_u, N_T, cfg["width"], cfg.get("depth", 4))


def batch_sweep(model, in_norm, out_norm, mode: str, val: dict) -> dict:
    """Seconds per trajectory vs batch size, full query path, surrogate against vectorised RK4.

    (``benchmark.batch_sweep`` is tied to the sine dataset and the sample-input mode, so it cannot
    time the compact-input models.)
    """
    t = torch.as_tensor(T_GRID / T_FINAL, dtype=torch.float32)
    out = {}
    for B in BATCHES:
        sub = {k: v[:B] for k, v in val.items()}

        def query():
            u = in_norm.encode(torch.as_tensor(raw_inputs(sub, "val", mode), dtype=torch.float32))
            with torch.no_grad():
                return out_norm.decode(model(u, t)).numpy()

        sur = bm.time_fn(query, 15)["median"] / B
        rk4 = bm.time_fn(lambda: bm.rk4_batched(sub["val_params"], sub["val_tau_coef"], T_GRID, X0), 5)["median"] / B
        out[B] = {"rk4_batched": rk4, "surrogate": sur, "speedup": rk4 / sur}
    return out


def cpu_load_pct() -> float:
    """System-wide CPU use over one second (needs psutil; nan if unavailable)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1.0)
    except ImportError:
        return float("nan")


def rss_mb() -> float:
    """Resident memory of this process in MB (needs psutil; nan if unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def main():
    load_before = cpu_load_pct()
    d = load(DATA_PATH_SMOOTH)
    val = {k: v for k, v in d.items() if k.startswith("val_")}
    params, coef = val["val_params"][:N_BATCH], val["val_tau_coef"][:N_BATCH]

    # Data-generation cost: scalar smooth RK4 (8 substeps, what `generate_smooth` runs per sample).
    t_one = bm.time_fn(lambda: [simulate_smooth(p, c) for p, c in zip(params[:4], coef[:4])], 3)["median"] / 4
    t_data = t_one * N_TOTAL
    prof = cProfile.Profile()
    prof.runcall(simulate_smooth, params[0], coef[0])
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(6)
    profile_txt = "\n".join(buf.getvalue().strip().splitlines()[-14:])

    rows, sweeps = [], {}
    for (kind, inp), stem in FINAL.items():
        path = ROOT / "checkpoints" / f"{stem.format(s=SEED)}.pt"
        _, _, _, hist = load_checkpoint(path)
        before = rss_mb()
        res = bm.run_checkpoint(path, n_samples=N_BATCH, repeat=7, rtols=RTOLS)
        model, in_norm, out_norm, _ = load_checkpoint(path)
        d_u = int(in_norm.mean.numel())
        cfg_stub = torch.load(path, map_location="cpu", weights_only=True)["config"]
        sur_err = res["surrogate_batch"]["median_err"]
        ivp = {k: v for k, v in res.items() if k.startswith("solve_ivp")}
        matched = min(ivp, key=lambda k: abs(np.log10(ivp[k]["median_err"]) - np.log10(sur_err)))
        t_train = float(hist["train_seconds"])
        row = {
            "model": kind, "input": inp, "params": int(hist["n_params"]), "d_u": d_u,
            "param_mb": hist["n_params"] * 4 / 1e6, "flops_per_traj": flops_for(kind, cfg_stub, d_u, hist),
            "train_s": t_train, "data_gen_s_est": t_data,
            "surrogate_ms_batch64": res["surrogate_batch"]["s_per_traj"] * 1e3,
            "surrogate_ms_single": res["surrogate_single"]["s_per_traj"] * 1e3,
            "median_err": sur_err, "rss_delta_mb": rss_mb() - before,
            "rk4_batched_ms": res["rk4_batched"]["s_per_traj"] * 1e3,
            "matched_solve_ivp": matched, "matched_ivp_err": ivp[matched]["median_err"],
            "matched_ivp_ms": ivp[matched]["s_per_traj"] * 1e3,
        }
        for ref, sec in (("rk4", res["rk4_batched"]["s_per_traj"]), ("ivp", ivp[matched]["s_per_traj"])):
            row[f"break_even_vs_{ref}"] = bm.break_even_queries(t_data, t_train, sec,
                                                                res["surrogate_batch"]["s_per_traj"])
        rows.append(row)
        sweeps[(kind, inp)] = batch_sweep(model, in_norm, out_norm, hist.get("input_mode", "samples"), val)
        sweeps[(kind, inp, "ivp")] = {k: (v["s_per_traj"] * 1e3, v["median_err"]) for k, v in ivp.items()}

    # numpy memory of the solvers (tracemalloc sees NumPy, not torch)
    mem_rk4 = bm.peak_memory_mb(lambda: bm.rk4_batched(params, coef, T_GRID, X0))
    mem_ivp = bm.peak_memory_mb(lambda: bm.solve_ivp_one(params[0], coef[0], T_GRID, X0, rtol=1e-3))

    # The solvers do not depend on the checkpoint: use the median over the four measurements.
    ivp_ms = {k: float(np.median([sweeps[key + ("ivp",)][k][0] for key in FINAL])) for k in sweeps[("mlp", "coef", "ivp")]}
    rk4_ms = float(np.median([r["rk4_batched_ms"] for r in rows]))
    for r in rows:
        r["rk4_batched_ms"] = rk4_ms
        r["matched_ivp_ms"] = ivp_ms[r["matched_solve_ivp"]]
        for ref, sec in (("rk4", rk4_ms / 1e3), ("ivp", ivp_ms[r["matched_solve_ivp"]] / 1e3)):
            r[f"break_even_vs_{ref}"] = bm.break_even_queries(r["data_gen_s_est"], r["train_s"], sec,
                                                              r["surrogate_ms_batch64"] / 1e3)
    for key in FINAL:
        sweeps[key + ("ivp",)] = {k: (ivp_ms[k], v[1]) for k, v in sweeps[key + ("ivp",)].items()}
    load_after = cpu_load_pct()

    keys = list(rows[0])
    with open(ROOT / "results" / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, sweeps, t_one, t_data, profile_txt, mem_rk4, mem_ivp, load_before, load_after)


def write_md(rows, sweeps, t_one, t_data, profile_txt, mem_rk4, mem_ivp, load_before, load_after):
    contended = not (max(load_before, load_after) < 25)  # also True if psutil is missing (nan)
    banner = (["> **CONTENDED RUN: do not quote these times.** System CPU use was "
               f"{load_before:.0f}% before and {load_after:.0f}% after the measurements (other jobs were running). "
               "Errors, FLOPs and parameter counts are unaffected; ms and break-even figures are only rough. "
               "Rerun `py -3.10 -m scripts.run_benchmark` on an idle machine.", ""] if contended else
              [f"System CPU use outside this process: {load_before:.0f}% before, {load_after:.0f}% after (idle run).", ""])
    L = ["# Benchmark: speed, memory and FLOPs (smooth family, validation split, seed 3001)", "",
         f"Generated by `scripts/run_benchmark.py` on {time.strftime('%Y-%m-%d %H:%M')}, "
         f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}, "
         f"PyTorch {torch.__version__} (CPU), NumPy {np.__version__}, {torch.get_num_threads()} torch threads. "
         "Wall clock, median of 7 repeats after one warm-up (batch sweep: 15 / 5). Solver timings are the median over the four checkpoint runs.", "",
         "Surrogate time covers the whole query path: build the raw input, z-score it, run the network, "
         "decode to raw units. Error is the headline metric (median per-sample relative L2, channels "
         "scaled by the train std) against the stored RK4 truth.", ""] + banner
    L += ["## Per model", "",
          "| model | input | params | param MB | FLOPs / traj | ms / traj (batch 64) | ms (single query) | median err |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['params']:,} | {r['param_mb']:.2f} | {r['flops_per_traj']:.2e} | "
                 f"{r['surrogate_ms_batch64']:.3f} | {r['surrogate_ms_single']:.2f} | {r['median_err']:.3f} |")
    L += ["", "FLOPs count multiply-adds as 2 and ignore activations; the DeepONet figure is the un-cached cost "
          "(the trunk does not depend on the sample and could be cached on a fixed time grid).", "",
          "## Solvers (batch of 64 unless stated)", "",
          "| method | ms / traj | median err | peak NumPy heap |", "|---|---|---|---|",
          f"| vectorised RK4, 8 substeps (the truth) | {rows[0]['rk4_batched_ms']:.2f} | 0 (by definition) | {mem_rk4 * 1e3:.0f} kB (batch 64) |",
          f"| scalar RK4 (`simulate_smooth`, what generated the data) | {t_one * 1e3:.1f} | 0 | n/a |"]
    ivp = sweeps[("deeponet", "samples", "ivp")]
    for k, (ms, err) in ivp.items():
        L.append(f"| `solve_ivp` RK45, {k.replace('solve_ivp_rtol', 'rtol ')} | {ms:.2f} | {err:.2e} | "
                 f"{mem_ivp * 1e3:.0f} kB (rtol 1e-3, one traj) |" if k.endswith("0.001") else
                 f"| `solve_ivp` RK45, {k.replace('solve_ivp_rtol', 'rtol ')} | {ms:.2f} | {err:.2e} | |")
    L += ["", "## Speed-up depends on the batch size", "",
          "Seconds per trajectory against vectorised RK4 (same 8-substep settings as the truth). NumPy RK4 "
          "amortises its Python overhead over the batch, so the ratio shrinks as the batch grows.", "",
          "| model | input | " + " | ".join(f"B={b}" for b in BATCHES) + " |", "|---|---|" + "---|" * len(BATCHES)]
    for (kind, inp), sw in ((k, v) for k, v in sweeps.items() if len(k) == 2):
        L.append(f"| {kind} | {inp} | " + " | ".join(f"{sw[b]['speedup']:.0f}x" for b in BATCHES) + " |")
    L += ["", "## Break-even: how many queries until the surrogate pays for itself", "",
          f"Cost to build the surrogate = data generation (estimated {t_data:.0f} s = {t_one * 1e3:.1f} ms x 10,000 scalar RK4 "
          "trajectories, serial) + training (`train_seconds` from the checkpoint; measured while other runs "
          "shared the CPU for some models, so treat as an upper bound). "
          "N* = (data + training) / (solver time per query - surrogate time per query), batch-64 surrogate time. "
          "The matched `solve_ivp` is the tolerance whose median error is closest to the surrogate's; "
          "that is the fair comparison (weak point 3); its error is shown so you can see how close the match is.", "",
          "| model | input | train s | matched `solve_ivp` (err) | ms/traj | N* vs matched solve_ivp | N* vs vectorised RK4 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['train_s']:.0f} | {r['matched_solve_ivp'].replace('solve_ivp_rtol', 'rtol ')} "
                 f"({r['matched_ivp_err']:.2f}) | {r['matched_ivp_ms']:.2f} | {r['break_even_vs_ivp']:,.0f} | "
                 f"{r['break_even_vs_rk4']:,.0f} |")
    L += ["", "## Profile of the scalar solver (cProfile, one `simulate_smooth` call)", "", "```", profile_txt, "```", "",
          "Caveats: one machine; single seed; the memory column sees NumPy only (PyTorch allocations are not "
          "traced, parameter memory is `n_params x 4 bytes`); `solve_ivp` timed on 8 trajectories; data-generation "
          "time is an estimate from the serial scalar solver, not a measured end-to-end generation run."]
    (ROOT / "results" / "benchmark.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
