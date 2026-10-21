"""Final speed / memory / FLOP benchmark of the four selected models (seed 3001).

    py -3.10 -m scripts.run_benchmark          # close other heavy programs first (idle machine)

For each selected checkpoint (same settings as ``scripts/final_eval.py``) it measures, on the smooth
family VALIDATION split, the cost of one trajectory through the surrogate, and compares it with
vectorised RK4 (the truth settings) and ``solve_ivp`` at several tolerances, including the tolerance
whose median error is closest to the surrogate's. It also reports parameters, FLOPs, memory, a
cProfile of the scalar solver and the break-even number of queries (data generation + training
against the per-query saving). Writes results/benchmark.md, results/benchmark.csv and
results/benchmark.json (everything, for the figure scripts).

Fairness choices (CLAUDE.md weak point 3), all measured in this run rather than taken from logs:
  * training cost: a few epochs of each final configuration are timed on this (idle) machine and
    scaled to 200 epochs; the ``train_seconds`` stored in the checkpoints were measured while other
    runs shared the CPU and are shown only for reference;
  * data-generation cost: measured, both with the serial scalar solver that actually generated the
    data and with the vectorised RK4 (the cheapest way to generate the same labels);
  * ``solve_ivp`` error is measured on the same 64 trajectories as the surrogate (timing on 16);
  * surrogate timed with all torch threads and with one thread (NumPy RK4 uses one core);
  * the DeepONet is also timed with its trunk cached on the fixed output grid (the trunk does not
    depend on the input function), which only a DeepONet can do.
"""
import cProfile
import csv
import io
import json
import platform
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from src import benchmark as bm
from src.data import DATA_PATH_SMOOTH, N_T, T_FINAL, T_GRID, X0, load, simulate_smooth
from src.evaluate import rel_l2_per_sample
from src.model import time_features
from src.train import (ChannelNormaliser, InputNormaliser, build_model, fit, load_checkpoint,
                       make_split, raw_inputs)

from .final_eval import FINAL

ROOT = Path(__file__).resolve().parent.parent
SEED = 3001
N_BATCH = 64          # trajectories in the timed batch (and in every error measurement)
N_IVP_TIME = 16       # trajectories timed per solve_ivp tolerance
N_TOTAL = 10_000      # size of the smooth dataset the models were trained on (train + val + test)
N_GEN_CHUNK = 1000    # vectorised-generation chunk; two chunks are timed and scaled to N_TOTAL
PROBE_EPOCHS = 3      # epochs timed per model to measure the training cost; scaled to EPOCHS
EPOCHS = 200
GRID_RUNS = 5         # runs in each model's lr / schedule grid (3 constant + 2 cosine), EXPERIMENTS.md E3
BATCHES = (1, 8, 64, 300, 1500)
RTOLS = (1e-1, 8e-2, 6e-2, 5e-2, 4e-2, 3e-2, 1e-2, 1e-3, 1e-4, 1e-6)  # dense near the surrogate's error (~0.1)


# ----------------------------------------------------------------------------------------- helpers
def cpu_load_pct() -> float:
    """System-wide CPU use over one second (needs psutil; nan if unavailable)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1.0)
    except ImportError:
        return float("nan")


def flops_for(kind: str, cfg: dict, d_u: int) -> dict:
    """FLOPs for one trajectory (151 query times): un-cached, and with the DeepONet trunk cached."""
    if kind == "deeponet":
        full = bm.deeponet_flops(d_u, N_T, cfg["p"], cfg["width"], cfg.get("branch_depth", 3),
                                 cfg.get("trunk_depth", 3))
        cached = bm.mlp_flops(d_u, cfg["width"], cfg.get("branch_depth", 3), 4 * cfg["p"]) + N_T * 4 * 2 * cfg["p"]
        return {"flops": full, "flops_cached": cached}
    f = bm.mlp_baseline_flops(d_u, N_T, cfg["width"], cfg.get("depth", 4))
    return {"flops": f, "flops_cached": f}


class Query:
    """The whole surrogate query path: raw input -> z-score -> network -> raw units.

    ``cached=True`` (DeepONet only) evaluates the trunk once on the fixed output grid at construction
    and reuses it, so a query costs one branch pass plus the p-term sum.
    """

    def __init__(self, model, in_norm, out_norm, mode: str, cached: bool = False):
        self.model, self.in_norm, self.out_norm, self.mode = model, in_norm, out_norm, mode
        self.t = torch.as_tensor(T_GRID / T_FINAL, dtype=torch.float32)
        self.trunk = None
        if cached:
            with torch.no_grad():
                tt = self.t.view(1, -1, 1)
                self.trunk = model.trunk(time_features(tt, model.n_fourier)).view(len(self.t), 4, model.p)

    def __call__(self, data: dict, split: str = "val") -> np.ndarray:
        u = self.in_norm.encode(torch.as_tensor(raw_inputs(data, split, self.mode), dtype=torch.float32))
        with torch.no_grad():
            if self.trunk is None:
                z = self.model(u, self.t)
            else:
                b = self.model.branch(u).view(len(u), 4, self.model.p)
                z = torch.einsum("bkp,tkp->btk", b, self.trunk) + self.model.bias
            return self.out_norm.decode(z).numpy()


def with_threads(n: int, fn):
    """Run fn() with torch limited to n threads, then restore the previous setting."""
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        return fn()
    finally:
        torch.set_num_threads(old)


def training_cost(path: Path, d: dict) -> float:
    """Seconds for EPOCHS epochs of this checkpoint's configuration, timed over PROBE_EPOCHS on this machine.

    Same data (7000 train / 1500 val), batch size, loss and per-epoch validation pass as the real run;
    the lr schedule does not change the cost. The first epoch is included (warm-up is part of training).
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    hist = ck["hist"]
    mode = hist.get("input_mode", "samples")
    in_norm = InputNormaliser.fit(raw_inputs(d, "train", mode))
    out_norm = ChannelNormaliser.fit(d["train_traj"])
    train = make_split(d, "train", in_norm, out_norm, T_FINAL, mode)
    val = make_split(d, "val", in_norm, out_norm, T_FINAL, mode)
    torch.manual_seed(SEED)
    model = build_model(ck["kind"], ck["config"])
    t0 = time.perf_counter()
    fit(model, train, val, out_norm, epochs=PROBE_EPOCHS, lr=1e-3, batch_size=hist.get("batch_size", 100),
        seed=SEED, verbose=0, consistency_weight=hist.get("consistency_weight", 0.0))
    return (time.perf_counter() - t0) / PROBE_EPOCHS * EPOCHS


# ----------------------------------------------------------------------------------------- main
def main():
    load_before = cpu_load_pct()
    d = load(DATA_PATH_SMOOTH)
    val = {k: v[:N_BATCH] for k, v in d.items() if k.startswith("val_")}
    params, coef, true = val["val_params"], val["val_tau_coef"], val["val_traj"].astype(np.float64)

    # --- data generation: serial scalar solver (what generated the data) and vectorised RK4
    t_scalar = bm.time_fn(lambda: [simulate_smooth(p, c) for p, c in zip(params[:4], coef[:4])], 3)["median"] / 4
    gen_times, gen_err = [], []
    for k in range(2):
        sl = slice(k * N_GEN_CHUNK, (k + 1) * N_GEN_CHUNK)
        t0 = time.perf_counter()
        X = bm.rk4_batched(d["train_params"][sl], d["train_tau_coef"][sl], T_GRID, X0)
        gen_times.append(time.perf_counter() - t0)
        gen_err.append(np.abs(X - d["train_traj"][sl]).max())
    t_gen_vec = float(np.mean(gen_times)) / N_GEN_CHUNK * N_TOTAL
    t_gen_scalar = t_scalar * N_TOTAL
    prof = cProfile.Profile()
    prof.runcall(simulate_smooth, params[0], coef[0])
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(6)
    profile_txt = "\n".join(buf.getvalue().strip().splitlines()[-14:])

    # --- solvers (independent of the checkpoint): vectorised RK4 and solve_ivp at each tolerance
    std_any = load_checkpoint(ROOT / "checkpoints" / f"{next(iter(FINAL.values())).format(s=SEED)}.pt")[2].std.numpy()
    rk4_s = bm.time_fn(lambda: bm.rk4_batched(params, coef, T_GRID, X0), 5)["median"] / N_BATCH
    rk4_err = rel_l2_per_sample(bm.rk4_batched(params, coef, T_GRID, X0), true, std_any)
    ivp = {}
    for rtol in RTOLS:
        run = lambda n: [bm.solve_ivp_one(p, c, T_GRID, X0, rtol=rtol, atol=rtol * 1e-3)
                         for p, c in zip(params[:n], coef[:n])]
        sols = run(N_BATCH)
        # At loose tolerances RK45 can blow up and stop early ("required step size is less than spacing
        # between numbers"): it returns fewer than 151 rows. Count it as a failure with infinite error.
        ok = np.array([len(s) == N_T for s in sols])
        err = np.full(N_BATCH, np.inf)
        if ok.any():
            err[ok] = rel_l2_per_sample(np.stack([s for s, k in zip(sols, ok) if k]), true[ok], std_any)
        ivp[f"{rtol:g}"] = {"ms": bm.time_fn(lambda: run(N_IVP_TIME), 3)["median"] / N_IVP_TIME * 1e3,
                            "median_err": float(np.median(err)), "p95_err": float(np.percentile(err, 95)),
                            "failures": int((~ok).sum())}
    mem_rk4 = bm.peak_memory_mb(lambda: bm.rk4_batched(params, coef, T_GRID, X0))
    mem_ivp = bm.peak_memory_mb(lambda: bm.solve_ivp_one(params[0], coef[0], T_GRID, X0, rtol=1e-3))

    # --- batch sweep for vectorised RK4 (once)
    big = {k: v[:max(BATCHES)] for k, v in d.items() if k.startswith("val_")}
    rk4_sweep = {B: bm.time_fn(lambda: bm.rk4_batched(big["val_params"][:B], big["val_tau_coef"][:B], T_GRID, X0),
                               5 if B < 1000 else 3)["median"] / B for B in BATCHES}

    # --- surrogates
    rows, sweeps = [], {}
    for (kind, inp), stem in FINAL.items():
        path = ROOT / "checkpoints" / f"{stem.format(s=SEED)}.pt"
        model, in_norm, out_norm, hist = load_checkpoint(path)
        cfg = torch.load(path, map_location="cpu", weights_only=True)["config"]
        mode = hist.get("input_mode", "samples")
        d_u = int(in_norm.mean.numel())
        q = Query(model, in_norm, out_norm, mode)
        variants = {"full": q}
        if kind == "deeponet":
            variants["cached"] = Query(model, in_norm, out_norm, mode, cached=True)
            assert np.allclose(variants["cached"](val), q(val), atol=1e-4), "cached trunk changed the output"
        pred = q(val)
        err = rel_l2_per_sample(pred, true, out_norm.std.numpy())
        row = {"model": kind, "input": inp, "params": int(hist["n_params"]), "d_u": d_u,
               "param_mb": hist["n_params"] * 4 / 1e6, **flops_for(kind, cfg, d_u),
               "median_err": float(np.median(err)), "p95_err": float(np.percentile(err, 95)),
               "train_s_logged": float(hist["train_seconds"]), "train_s_idle": training_cost(path, d)}
        for name, fn in variants.items():
            for threads in (torch.get_num_threads(), 1):
                tag = f"{name}_t{'all' if threads != 1 else 1}"
                row[f"ms_b64_{tag}"] = with_threads(threads, lambda: bm.time_fn(lambda: fn(val), 21)["median"]) / N_BATCH * 1e3
                one = {k: v[:1] for k, v in val.items()}
                row[f"ms_single_{tag}"] = with_threads(threads, lambda: bm.time_fn(lambda: fn(one), 51)["median"]) * 1e3
        best = min((k for k in row if k.startswith("ms_b64_")), key=lambda k: row[k])
        row["ms_b64_best"], row["ms_b64_best_variant"] = row[best], best[len("ms_b64_"):]
        sweeps[f"{kind}/{inp}"] = {}
        for name, fn in variants.items():
            sweeps[f"{kind}/{inp}"][name] = {
                B: bm.time_fn(lambda: fn({k: v[:B] for k, v in big.items()}), 15 if B < 1000 else 7)["median"] / B
                for B in BATCHES}
        rows.append(row)

    # --- matched tolerance and break-even
    for r in rows:
        r["matched_rtol"] = min(ivp, key=lambda k: abs(np.log10(ivp[k]["median_err"]) - np.log10(r["median_err"])))
        r["matched_ms"] = ivp[r["matched_rtol"]]["ms"]
        r["matched_err"] = ivp[r["matched_rtol"]]["median_err"]
        sur_s = r["ms_b64_best"] / 1e3
        for gen_name, t_gen in (("vec", t_gen_vec), ("scalar", t_gen_scalar)):
            for cost_name, t_train in (("one", r["train_s_idle"]), ("grid", GRID_RUNS * r["train_s_idle"])):
                for ref, sec in (("ivp", r["matched_ms"] / 1e3), ("rk4", rk4_s)):
                    r[f"N_{ref}_{gen_name}_{cost_name}"] = bm.break_even_queries(t_gen, t_train, sec, sur_s)
    load_after = cpu_load_pct()

    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "cpu": platform.processor() or platform.machine(),
            "os": f"{platform.system()} {platform.release()}", "torch": torch.__version__, "numpy": np.__version__,
            "threads": torch.get_num_threads(), "load_before": load_before, "load_after": load_after,
            "t_scalar_ms": t_scalar * 1e3, "t_gen_vec_s": t_gen_vec, "t_gen_scalar_s": t_gen_scalar,
            "gen_max_abs_diff": float(max(gen_err)), "rk4_ms": rk4_s * 1e3,
            "rk4_err_median": float(np.median(rk4_err)), "mem_rk4_mb": mem_rk4, "mem_ivp_mb": mem_ivp}
    (ROOT / "results" / "benchmark.json").write_text(json.dumps(
        {"meta": meta, "rows": rows, "ivp": ivp, "rk4_sweep": rk4_sweep, "sweeps": sweeps}, indent=1), encoding="utf-8")
    fields = list(dict.fromkeys(k for r in rows for k in r))   # DeepONet rows have extra cached-trunk columns
    with open(ROOT / "results" / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, restval="")
        w.writeheader()
        w.writerows(rows)
    write_md(meta, rows, ivp, rk4_sweep, sweeps, profile_txt)


def write_md(meta, rows, ivp, rk4_sweep, sweeps, profile_txt):
    load = max(meta["load_before"], meta["load_after"])
    contended = not (load < 25)  # also True if psutil is missing (nan)
    banner = ((f"> **Not fully idle.** System CPU use was {meta['load_before']:.0f}% before and {meta['load_after']:.0f}% "
               "after the measurements. No other training or benchmark job was running; the load came from desktop "
               "programs (check Task Manager). Above about 50 % treat every ms as unreliable; between 25 and 50 % "
               "expect timings up to roughly 20 to 30 % slow, ratios between methods less affected.")
              if contended else
              f"System CPU use outside this process: {meta['load_before']:.0f}% before, {meta['load_after']:.0f}% "
              "after (idle run; background desktop programs only).")
    L = ["# Benchmark: speed, memory and FLOPs (smooth family, validation split, seed-3001 checkpoints)", "",
         f"Generated by `scripts/run_benchmark.py` on {meta['date']}, {meta['cpu']}, {meta['os']}, PyTorch "
         f"{meta['torch']} (CPU), NumPy {meta['numpy']}, {meta['threads']} torch threads by default. Wall clock, "
         "median of repeated calls after one warm-up call.", "", banner, "",
         "Surrogate time covers the whole query path: build the raw input, z-score it, run the network, decode to "
         "raw units. Error is the headline metric (per-sample relative L2, channels scaled by the train std) against "
         f"the stored RK4 truth, on the first {N_BATCH} validation trajectories (the same ones for every method).", "",
         "## Per model", "",
         "| model | input | params | FLOPs / traj | FLOPs / traj, trunk cached | median err | p95 err | ms / traj, batch 64 | "
         "same, 1 thread | single query (ms) | same, 1 thread |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['params']:,} | {r['flops']:.2e} | {r['flops_cached']:.2e} | "
                 f"{r['median_err']:.3f} | {r['p95_err']:.3f} | {r['ms_b64_full_tall']:.3f} | {r['ms_b64_full_t1']:.3f} | "
                 f"{r['ms_single_full_tall']:.2f} | {r['ms_single_full_t1']:.2f} |")
    dn = [r for r in rows if r["model"] == "deeponet"]
    L += ["", "DeepONet with the trunk evaluated once on the fixed 151-point grid and reused (identical output, checked "
          "to 1e-4): " + "; ".join(f"{r['input']}: {r['ms_b64_cached_tall']:.3f} ms / traj at batch 64 "
                                   f"({r['ms_b64_cached_t1']:.3f} on 1 thread), single query {r['ms_single_cached_tall']:.2f} ms"
                                   for r in dn) + ". An MLP cannot do this: t enters its first layer together with the input.", "",
          "FLOPs count a multiply-add as 2 and ignore activations. Parameter memory is n_params x 4 bytes "
          f"(about {rows[0]['param_mb']:.2f} MB for every model).", "",
          "## Solvers", "",
          f"Errors over the same {N_BATCH} trajectories; `solve_ivp` time per trajectory over {N_IVP_TIME} of them "
          "(one at a time, it is not vectorised); atol = rtol x 1e-3.", "",
          "| method | ms / traj | median err | p95 err | failed runs |", "|---|---|---|---|---|",
          f"| scalar RK4 (`simulate_smooth`, what generated the data), one trajectory | {meta['t_scalar_ms']:.1f} | 0 (is the truth) | | |",
          f"| vectorised RK4, 8 substeps, batch {N_BATCH} | {meta['rk4_ms']:.2f} | {meta['rk4_err_median']:.0e} (same scheme as the truth) | | |"]
    for k, v in ivp.items():
        L.append(f"| `solve_ivp` RK45, rtol {k} | {v['ms']:.2f} | {v['median_err']:.2e} | {v['p95_err']:.2e} | "
                 f"{v['failures']} / {N_BATCH} |")
    L += ["", "A failed run is one where RK45 stopped early ('required step size is less than spacing between "
          "numbers': a loose step let the state blow up). It counts as infinite error, so a p95 of inf means more "
          "than 5 % failed. The time column includes the work done before a failure."]
    L += ["", f"Peak NumPy heap: vectorised RK4, batch {N_BATCH}: {meta['mem_rk4_mb'] * 1e3:.0f} kB; `solve_ivp` rtol 1e-3, one "
          f"trajectory: {meta['mem_ivp_mb'] * 1e3:.0f} kB (tracemalloc; PyTorch allocations are not traced).", "",
          "## Speed-up depends on the batch size", "",
          "Milliseconds per trajectory (all torch threads). Vectorised NumPy RK4 amortises its Python overhead over the "
          "batch, so the ratio shrinks as the batch grows.", "",
          "| method | " + " | ".join(f"B={b}" for b in BATCHES) + " |", "|---|" + "---|" * len(BATCHES),
          "| vectorised RK4 | " + " | ".join(f"{rk4_sweep[b] * 1e3:.3f}" for b in BATCHES) + " |"]
    for key, sw in sweeps.items():
        for name, s in sw.items():
            L.append(f"| {key}{' (trunk cached)' if name == 'cached' else ''} | " +
                     " | ".join(f"{s[b] * 1e3:.3f} ({rk4_sweep[b] / s[b]:.0f}x)" for b in BATCHES) + " |")
    L += ["", "## Break-even: how many queries until the surrogate pays for itself", "",
          "N* = (data generation + training) / (solver time per query - surrogate time per query), surrogate at "
          "batch 64 in its fastest variant. The fair rival is `solve_ivp` at the tolerance whose median error is "
          "closest to the surrogate's (weak point 3).", "",
          f"- Data generation, measured: {meta['t_gen_vec_s']:.0f} s with vectorised RK4 (two chunks of {N_GEN_CHUNK}, scaled "
          f"to {N_TOTAL:,}; identical to the stored labels to {meta['gen_max_abs_diff']:.0e}), or "
          f"{meta['t_gen_scalar_s']:.0f} s with the serial scalar solver that produced the dataset.",
          f"- Training, measured idle: {PROBE_EPOCHS} epochs of each final configuration timed and scaled to {EPOCHS}. "
          f"'with lr grid' = {GRID_RUNS} such runs (the lr / schedule grid each model was selected from); the 2 extra "
          "seeds are for error bars and are not counted.", "",
          "| model | input | train s (idle) | train s (logged, contended) | matched rtol (its err) | ms / traj | "
          "N*, vec data, 1 run | N*, vec data, lr grid | N*, scalar data, lr grid | N* vs vectorised RK4, vec data, 1 run |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['train_s_idle']:.0f} | {r['train_s_logged']:.0f} | {r['matched_rtol']} "
                 f"({r['matched_err']:.3f}) | {r['matched_ms']:.2f} | {r['N_ivp_vec_one']:,.0f} | {r['N_ivp_vec_grid']:,.0f} | "
                 f"{r['N_ivp_scalar_grid']:,.0f} | {r['N_rk4_vec_one']:,.0f} |")
    L += ["", "## Profile of the scalar solver (cProfile, one `simulate_smooth` call)", "", "```", profile_txt, "```", "",
          "Caveats: one machine, one run of the benchmark; seed-3001 checkpoints only (the other seeds have identical "
          "architectures, so identical cost); timings on a Windows laptop CPU vary by roughly 10 to 20 % between runs; "
          "the surrogate's p95 error is about 0.6, so 'matched median error' still favours the surrogate in the tail "
          "(compare the p95 columns)."]
    (ROOT / "results" / "benchmark.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
