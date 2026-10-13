"""Headless smoke test for the evaluation figures in scripts/plot_eval.py (synthetic data, no checkpoint)."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("torch")


def _fake_result(rng, true, label, noise):
    from src.evaluate import error_vs_time, rel_l2_per_sample
    pred = true + noise * rng.standard_normal(true.shape)
    std = true.std(axis=(0, 1))
    return {"label": label, "pred": pred, "std": std,
            "err": rel_l2_per_sample(pred, true, std),
            "err_time": error_vs_time(pred, true, std),
            "err_time_ch": [error_vs_time(pred[..., [j]], true[..., [j]]) for j in range(4)]}


def test_all_four_figures_are_written(tmp_path):
    from scripts.plot_eval import fig_cases, fig_cdf, fig_inputs, fig_regime, fig_time
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 31)
    # calm and spinning samples so that several regime bins are populated
    amp = np.repeat([1.0, 5.0, 10.0, 20.0], 6)[:, None, None]
    true = amp * np.sin(t[None, :, None] * np.arange(1, 5)) + 0.1
    results = [_fake_result(rng, true, "model A", 0.05), _fake_result(rng, true, "model B", 0.2)]
    outs = [tmp_path / f"{n}.png" for n in ("cases", "time", "cdf", "regime", "inputs")]
    fig_cases(results[0], true, t, outs[0], "val")
    fig_time(results, t, outs[1], "val")
    fig_cdf(results, outs[2], "val")
    fig_regime(results, true, outs[3], "val")
    params, torque = rng.random((len(true), 2)), rng.standard_normal((len(true), 2, len(t)))
    fig_inputs(results[0], params, torque, outs[4], "val")
    assert all(p.exists() and p.stat().st_size > 10_000 for p in outs)


def test_test_split_needs_final_flag():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "scripts.plot_eval", "nothing.pt", "--split", "test"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--final" in r.stderr
