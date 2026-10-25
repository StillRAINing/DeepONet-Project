"""Report figures (Figures 2 to 8), drawn at the size they are printed.

    py -3.10 -m scripts.report_figures          # about 1 minute, no model is loaded

The analysis scripts (``solver_figures``, ``final_analysis``) draw wide working figures that become
unreadable when shrunk onto an A4 page. This script redraws the figures used in the body of the report
at 6.5 in wide (the text width) with 8 pt text, so nothing is rescaled when they are inserted. It reads only
saved results, so it never touches model selection:

  data/dataset_smooth.npz            labels, parameters, torques (solver checks are recomputed with src.solvers)
  results/final_eval*.csv            test / OOD metrics, three seeds
  results/final_persample.npz        per-sample errors of the seed-3001 checkpoints (written by final_analysis)
  results/benchmark.json             timings (written by run_benchmark)

Writes results/report/fig2_convergence.png ... fig8_breakeven.png.
"""
import csv
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

try:                                     # the plotting below never calls torch; allow running without it
    import torch  # noqa: F401
except ImportError:
    stub = types.ModuleType("torch")
    stub.no_grad = lambda: (lambda f: f)
    sys.modules["torch"] = stub

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.integrate import solve_ivp

from src.data import DATA_PATH_SMOOTH, T_FINAL, T_GRID, X0, load, smooth_torque_fn
from src.dynamics import ArmParams, energy, state_derivative
from src.solvers import rk4_integrate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "report"
W = 6.5                                    # text width of the report, inches
H_DATA = T_FINAL / ((len(T_GRID) - 1) * 8)  # 2.5 ms, the step that made the labels

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "savefig.dpi": 300, "mathtext.default": "regular",
})
# one colour per architecture (blue MLP, orange DeepONet), darker shade and filled/hatched = compact input
MODELS = {
    ("mlp", "coef"): ("MLP, compact", "#0072B2", "o"),
    ("mlp", "samples"): ("MLP, samples", "#56B4E9", "s"),
    ("deeponet", "coef"): ("DeepONet, compact", "#D55E00", "D"),
    ("deeponet", "samples"): ("DeepONet, samples", "#E69F00", "^"),
}
SOLVER = "#555555"
RK4 = "#009E73"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def seeds(rows, key, model, inp, split):
    v = np.array([float(r[key]) for r in rows if r["model"] == model and r["input"] == inp and r["split"] == split])
    return v.mean(), v.std(ddof=1)


def plain_log(ax, axis, ticks):
    """Log axis with readable plain-number tick labels (0.05, 0.1, 0.2 ...) instead of only 10^k."""
    a = getattr(ax, f"{axis}axis")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", OUT / name)


def arm_rhs(params, coef):
    p = replace(ArmParams(), m_p=float(params[0]), b=float(params[1]))
    tau = smooth_torque_fn(coef)
    return (lambda t, x: state_derivative(t, x, p, tau)), p, tau


# ----------------------------------------------------------------------------------- Figure 2: RK4 order
def fig2(d):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    ns = 10 * 2 ** np.arange(0, 8)
    hs = 2.0 / ns
    err = [abs(rk4_integrate(lambda t, y: -y, np.array([1.0]), np.linspace(0, 2, n + 1))[-1, 0] - np.exp(-2)) for n in ns]
    ax[0].loglog(hs, err, "o-", color="#0072B2", label="RK4")
    ax[0].loglog(hs, err[0] * (hs / hs[0]) ** 4, ":", color=SOLVER, label="slope 4")
    ax[0].set(xlabel="step h", ylabel=r"error at t = 2", title="(a) dy/dt = −y (exact solution known)")
    ax[0].legend(loc="upper left")

    peak = np.abs(d["test_traj"][..., :2]).max(axis=(1, 2))
    calm, wild = int(np.argsort(peak)[len(peak) // 2]), int(np.argmax(peak))
    arm_n = 75 * 2 ** np.arange(0, 8)
    arm_h = T_FINAL / arm_n
    for name, i, c in (("calm", calm, "#D55E00"), ("most violent", wild, "#009E73")):
        f, _, _ = arm_rhs(d["test_params"][i], d["test_tau_coef"][i])
        e = []
        for n in arm_n:
            t = np.linspace(0, T_FINAL, n + 1)
            s = n // 75
            ref = solve_ivp(f, (0, T_FINAL), X0, method="DOP853", t_eval=t[::s], rtol=1e-12, atol=1e-12).y.T
            e.append(np.abs(rk4_integrate(f, X0, t)[::s] - ref).max())
        ax[1].loglog(arm_h * 1e3, e, "o-", color=c, label=f"{name} sample (peak |q| {peak[i]:.0f} rad)")
        if name == "calm":
            e_calm = e
    ax[1].loglog(arm_h * 1e3, e_calm[2] * (arm_h / arm_h[2]) ** 4, ":", color=SOLVER, label="slope 4")
    ax[1].axvline(H_DATA * 1e3, color="#888888", lw=0.8, ls="--")
    ax[1].set_ylim(3e-11, 1e5)
    ax[1].annotate("dataset step 2.5 ms", (H_DATA * 1e3, 1e-10), xytext=(4, 0), textcoords="offset points",
                   fontsize=7, color="#555555")
    ax[1].set(xlabel="step h [ms]", ylabel="max state error vs DOP853", title="(b) arm, reference DOP853 (rtol 1e-12)")
    ax[1].legend(loc="upper left", fontsize=6.5)
    save(fig, "fig2_convergence.png")


# ----------------------------------------------------------------------------------- Figure 3: energy
def fig3(d):
    fig = plt.figure(figsize=(W, 2.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=0.12, wspace=0.32)
    a = fig.add_subplot(gs[:, 0])
    p = replace(ArmParams(), m_p=0.5, b=0.0)
    f = lambda t, x: state_derivative(t, x, p, lambda s: np.zeros(2))
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for k, h in enumerate((0.02, 0.01, 0.005, H_DATA)):
        t = np.linspace(0, 10.0, int(round(10.0 / h)) + 1)
        X = rk4_integrate(f, X0, t)
        E = np.array([energy(x, p) for x in X])
        dE = np.maximum.accumulate(np.abs(E - E[0]))     # running maximum: the drift so far
        a.semilogy(t[1:], np.maximum(dE[1:], 1e-16), color=cols[k], label=f"h = {h * 1e3:g} ms")
    a.set_ylim(1e-9, 3)
    a.set(xlabel="t [s]", ylabel=r"max $|E(t) - E(0)|$ so far [J]", title=r"(a) conservation: $b = 0$, $\tau = 0$")
    a.legend(loc="lower right", ncol=2)

    i = int(np.argsort(np.abs(d["test_traj"][..., :2]).max(axis=(1, 2)))[len(d["test_traj"]) // 2])
    f, pb, tau = arm_rhs(d["test_params"][i], d["test_tau_coef"][i])
    t = np.linspace(0, T_FINAL, int(round(T_FINAL / H_DATA)) + 1)
    X = rk4_integrate(f, X0, t)
    E = np.array([energy(x, pb) for x in X])
    tq = np.array([tau(s) for s in t])
    power = (tq * X[:, 2:]).sum(1) - pb.b * (X[:, 2:] ** 2).sum(1)
    Wk = np.concatenate([[0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    top = fig.add_subplot(gs[0, 1])
    bot = fig.add_subplot(gs[1, 1], sharex=top)
    top.plot(t, E - E[0], color="#0072B2", lw=2.2, label=r"$E(t) - E(0)$")
    top.plot(t, Wk, color="#D55E00", lw=1.2, ls="--", label=r"work in, $\int (\tau\cdot\dot q - b|\dot q|^2)\,dt$")
    top.set(ylabel="energy [J]", title=r"(b) balance with torque and friction (test sample)")
    top.legend(loc="upper right", fontsize=6.5)
    plt.setp(top.get_xticklabels(), visible=False)
    bot.semilogy(t[1:], np.maximum(np.abs((E - E[0]) - Wk)[1:], 1e-12), color="#009E73", lw=1)
    bot.set(xlabel="t [s]", ylabel="|difference| [J]", ylim=(1e-6, 1e-1))
    bot.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-1]))
    save(fig, "fig3_energy.png")


# ----------------------------------------------------------------------------------- Figure 4: overview
SPLITS = [("test", "test\n(in range)"), ("ood", "heavier\nm_p, b"), ("ood_amp1", "torque\n6–9 N m"),
          ("ood_amp2", "torque\n9–12 N m"), ("ood_smooth", "rougher\ntorque")]


def fig4(fe):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4))
    x = np.arange(len(SPLITS))
    off = np.linspace(-0.24, 0.24, len(MODELS))
    for k, ((m, inp), (label, c, mk)) in enumerate(MODELS.items()):
        for a, key in zip(ax, ("median", "p95")):
            mu = [seeds(fe, key, m, inp, s)[0] for s, _ in SPLITS]
            sd = [seeds(fe, key, m, inp, s)[1] for s, _ in SPLITS]
            a.errorbar(x + off[k], mu, yerr=sd, fmt=mk, color=c, ms=4.5, capsize=2, lw=1, label=label)
    for a, key, ticks in ((ax[0], "median", [0.05, 0.1, 0.2, 0.5, 1]), (ax[1], "p95", [0.1, 0.2, 0.5, 1])):
        a.set_yscale("log")
        plain_log(a, "y", ticks)
        a.set_xticks(x, [n for _, n in SPLITS], fontsize=7)
        a.grid(axis="x", visible=False)
        a.axvspan(-0.5, 0.5, color="#009E73", alpha=0.07, lw=0)
        a.set_xlim(-0.5, len(SPLITS) - 0.5)
    ax[0].axhline(0.10, color="#888888", lw=0.8, ls="--")
    ax[0].text(4.45, 0.104, "criterion 0.10", ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax[1].axhline(0.20, color="#888888", lw=0.8, ls="--")
    ax[1].text(4.45, 0.208, "criterion 0.20", ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax[0].set(ylabel="relative L2 error", title="(a) median error")
    ax[1].set(title="(b) 95th-percentile error")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig4_overview.png")


# ----------------------------------------------------------------------------------- Figure 5: consistency
def fig5(fe, fe_cons):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})
    rows = []
    for (m, inp), (label, c, mk) in MODELS.items():
        rows.append((label, c, mk, fe, m, inp, False))
    for (m, inp) in (("mlp", "samples"), ("deeponet", "samples")):
        label, c, mk = MODELS[(m, inp)]
        rows.append((label + " + penalty", c, mk, fe_cons, m, inp, True))
    for y, (label, c, mk, src, m, inp, pen) in enumerate(rows):
        mu, sd = seeds(src, "consistency", m, inp, "test")
        ax[0].errorbar(mu, y, xerr=sd, fmt=mk, color=c, mfc="white" if pen else c, mew=1.3, ms=5, capsize=2)
        ax[0].text((mu + sd) * 1.2, y, f"{mu:.3f}", va="center", fontsize=7)
        em, _ = seeds(src, "energy_excess_rel", m, inp, "test")
        ex, _ = seeds(src, "median", m, inp, "test")
        ax[1].plot(ex, em, mk, color=c, mfc="white" if pen else c, mew=1.3, ms=6)
    ax[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    ax[0].invert_yaxis()
    ax[0].set_xscale("log")
    plain_log(ax[0], "x", [0.01, 0.03, 0.1, 0.3])
    ax[0].set_xlim(0.005, 0.6)
    ax[0].grid(axis="y", visible=False)
    ax[0].set(xlabel=r"rms($dq/dt - \dot q$) / rms(true $\dot q$)", title=r"(a) are the $q$ and $\dot q$ outputs consistent?")
    ax[1].set(xlabel="test median error (accuracy)", ylabel="excess energy-balance residual",
              title="(b) energy balance follows accuracy")
    ax[1].legend(handles=[Line2D([], [], marker="o", ls="none", color="#0072B2", label="MLP"),
                          Line2D([], [], marker="D", ls="none", color="#D55E00", label="DeepONet"),
                          Line2D([], [], marker="o", ls="none", color="#555555", mfc="white", mew=1.3,
                                 label="with consistency penalty")], loc="upper left")
    save(fig, "fig5_consistency.png")


# ----------------------------------------------------------------------------------- Figure 6: out of range
def fig6(d, ps, fe):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.3})
    peak = np.concatenate([np.abs(d[f"{s}_torque"]).max(axis=(1, 2)) for s in ("test", "ood_amp1", "ood_amp2")])
    edges = np.arange(0, 13, 1.0)
    cen = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(peak, edges) - 1
    for (m, inp), (label, c, mk) in MODELS.items():
        e = np.concatenate([ps[f"{m}__{inp}|{s}|err"] for s in ("test", "ood_amp1", "ood_amp2")])
        med = [np.median(e[idx == k]) if (idx == k).sum() >= 5 else np.nan for k in range(len(cen))]
        ax[0].plot(cen, med, marker=mk, color=c, ms=4, label=label)
    ax[0].axvspan(0, 6, color="#009E73", alpha=0.08, lw=0)
    ax[0].axvline(6, color="#009E73", lw=0.8)
    ax[0].text(3, 0.75, "training range", ha="center", fontsize=7.5, color="#006b4f")
    ax[0].axhline(0.43, color="#888888", lw=0.8, ls="--")
    ax[0].text(11.9, 0.39, "predict-the-mean\nbaseline (0.43)", ha="right", va="top", fontsize=6.5, color="#555555")
    ax[0].set_yscale("log")
    plain_log(ax[0], "y", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax[0].set_ylim(0.007, 1.3)
    ax[0].set_xlim(0, 12)
    ax[0].set(xlabel=r"peak $|\tau|$ of the torque input [N m]", ylabel="median relative L2 error",
              title="(a) error against torque amplitude")
    ax[0].legend(loc="lower right", ncol=1)

    for k, fam in enumerate(("mlp", "deeponet")):
        for j, inp in enumerate(("coef", "samples")):
            _, c, _ = MODELS[(fam, inp)]
            mu_s, sd_s = seeds(fe, "median", fam, inp, "ood_smooth")
            mu_t, _ = seeds(fe, "median", fam, inp, "test")
            pos = k + (j - 0.5) * 0.38
            ax[1].bar(pos, mu_s, 0.36, yerr=sd_s, color=c, capsize=2, edgecolor="#333333", lw=0.5,
                      hatch="////" if inp == "coef" else "", error_kw={"lw": 0.8})
            ax[1].plot([pos - 0.18, pos + 0.18], [mu_t, mu_t], color="#222222", lw=1.6, ls=":")
            ax[1].text(pos, mu_s + sd_s + 0.006, f"{mu_s:.2f}", ha="center", fontsize=7)
    ax[1].set_xticks([0, 1], ["MLP", "DeepONet"])
    ax[1].grid(axis="x", visible=False)
    ax[1].set_ylim(0, 0.34)
    ax[1].set(ylabel="median relative L2 error", title="(b) rougher torque (13 modes)")
    ax[1].legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="#333333", hatch="////", label="compact input"),
                          Patch(facecolor="#bbbbbb", edgecolor="#333333", label="sampled input"),
                          Line2D([], [], color="#222222", ls=":", lw=1.6, label="same model, in-range test")],
                 loc="upper center", ncol=1, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig6_ood.png")


# ----------------------------------------------------------------------------------- Figures 7 and 8: speed
def fig7_8():
    B = json.loads((RES / "benchmark.json").read_text(encoding="utf-8"))
    meta, rows, ivp = B["meta"], B["rows"], B["ivp"]
    rk4_big = B["rk4_sweep"]["1500"] * 1e3

    # Figure 7: the same numbers as Table 5 (batch 64, default threads)
    fig, ax = plt.subplots(figsize=(W, 2.55))
    ok = {k: v for k, v in ivp.items() if np.isfinite(v["median_err"])}
    ax.plot([v["ms"] for v in ok.values()], [v["median_err"] for v in ok.values()], "o-", color=SOLVER, ms=3.5,
            label="solve_ivp RK45, one at a time")
    for k, v in ok.items():
        if float(k) in (0.1, 0.03, 0.001, 1e-4, 1e-6):
            ax.annotate(f"rtol {k}", (v["ms"], v["median_err"]), fontsize=6.5, xytext=(5, -2), textcoords="offset points")
    for r in rows:
        label, c, mk = MODELS[(r["model"], r["input"])]
        ax.plot(r["ms_b64_full_tall"], r["median_err"], mk, color=c, ms=6, label=label)
        if r["model"] == "deeponet":
            ax.plot(r["ms_b64_cached_tall"], r["median_err"], mk, color=c, ms=6, mfc="white", mew=1.3)
            ax.annotate("", (r["ms_b64_cached_tall"] * 1.15, r["median_err"]), (r["ms_b64_full_tall"] / 1.15, r["median_err"]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
    ax.text(0.019, 0.042, "hollow: trunk cached", fontsize=6.5, color="#D55E00")
    for x, txt, ha, f in ((meta["rk4_ms"], "batched RK4,\nbatch 64 (exact)", "left", 1.08),
                          (rk4_big, "batched RK4,\nbatch 1500 (exact)", "right", 0.92)):
        ax.axvline(x, color=RK4, lw=1, ls="--")
        ax.text(x * f, 2e-6, txt, ha=ha, fontsize=6.5, color="#00684a")
    ax.text(0.275, 0.045, "two MLPs\n(overlap)", ha="center", va="top", fontsize=6.5, color="#0072B2")
    ax.annotate("better", xy=(0.02, 1.5e-6), xytext=(0.06, 1.2e-5), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax, "x", [0.01, 0.1, 1, 10, 30])
    ax.set_xlim(0.01, 50)
    ax.set_ylim(3e-7, 1.5)
    ax.set(xlabel="milliseconds per trajectory (surrogates at batch 64)",
           ylabel="median relative L2 error")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.72), fontsize=6.5)
    save(fig, "fig7_pareto.png")

    # Figure 8: break-even, compact-input models, data + one training run
    fig, ax = plt.subplots(figsize=(W, 2.35))
    N = np.logspace(3, 8, 300)
    msi = ivp["0.04"]["ms"]   # fastest of the matched tolerances (0.03 to 0.05)
    ax.loglog(N, N * msi / 1e3, color=SOLVER, ls="--", label=f"solve_ivp at matched error ({msi:.1f} ms)")
    ax.loglog(N, N * rk4_big / 1e3, color=RK4, ls="-.", label=f"batched RK4, batch 1500 ({rk4_big:.2f} ms)")
    out = {}
    for r in rows:
        if r["input"] != "coef":
            continue
        label, c, mk = MODELS[(r["model"], r["input"])]
        key = f"{r['model']}/{r['input']}"
        sur = min(B["sweeps"][key][v]["1500"] for v in B["sweeps"][key]) * 1e3     # fastest variant at batch 1500
        fixed = meta["t_gen_vec_s"] + r["train_s_idle"]
        ax.loglog(N, fixed + N * sur / 1e3, color=c, lw=2, label=f"{label}: data + training + queries")
        for rival, ms, va in ((r["matched_ms"], "solve_ivp", "bottom"), (rk4_big, "RK4", "top")):
            # vs solve_ivp: the benchmark's own N* (its matched tolerance, surrogate at batch 64)
            n = r["N_ivp_vec_one"] if ms == "solve_ivp" else fixed / ((rival - sur) / 1e3)
            out[(label, ms)] = n
            ax.plot(n, fixed + n * sur / 1e3, mk, color=c, ms=6, mec="white", mew=0.6)
            txt = f"{n / 1e3:.0f}k" if n < 1e6 else f"{n / 1e6:.1f}M"
            ax.annotate(txt, (n, fixed + n * sur / 1e3), xytext=(-5, 5) if va == "bottom" else (6, -6),
                        textcoords="offset points", ha="right" if va == "bottom" else "left", va=va, fontsize=7, color=c, fontweight="bold")
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(5, 3e5)
    ax.set(xlabel="number of trajectories queried", ylabel="total wall-clock time [s]")
    ax.legend(loc="upper left")
    save(fig, "fig8_breakeven.png")
    for k, v in out.items():
        print("break-even", k, f"{v:,.0f}")


def main():
    d = load(DATA_PATH_SMOOTH)
    fe = read_csv(RES / "final_eval.csv")
    fe_cons = read_csv(RES / "final_eval_ff6cons.csv")
    ps = dict(np.load(RES / "final_persample.npz"))
    fig2(d)
    fig3(d)
    fig4(fe)
    fig5(fe, fe_cons)
    fig6(d, ps, fe)
    fig7_8()


if __name__ == "__main__":
    main()
