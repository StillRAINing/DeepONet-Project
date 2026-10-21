"""Headless smoke test for the pygame visualiser (skipped when pygame is not installed)."""
import os

import pytest

pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def test_app_runs_a_few_frames_and_rk45_matches_rk4():
    import numpy as np

    from src.viz import App

    app = App()
    app.show_rk45 = True
    app.solve()
    assert app.traj.shape[1] == 4 and np.isfinite(app.traj).all()
    # RK45 overlay should sit on the RK4 curve for the default (moderate) settings
    assert np.max(np.abs(app.rk45 - app.traj)) < 1e-2 * max(1.0, np.max(np.abs(app.traj)))
    app.run(max_frames=3)


def test_smooth_family_app_scales_the_stored_torque_shape_and_overlays_surrogates():
    import numpy as np

    from src import viz
    from src.data import PEAK_GRID, smooth_torque_samples

    viz.SURROGATES["perfect"] = lambda params, coef: viz.simulate_smooth(params, coef)
    try:
        app = viz.App(family="smooth")
        assert len(app.sliders) == 4  # m_p, b and one peak slider per joint
        params, coef = app.inputs()
        peak = np.abs(smooth_torque_samples(coef, PEAK_GRID)).max(axis=1)
        assert coef.shape[:2] == (2, 2) and np.allclose(peak, [app.sliders[2].val, app.sliders[3].val])
        assert np.allclose(app.surr["perfect"], app.traj) and app.rel_l2(app.surr["perfect"]) == 0.0
        app.run(max_frames=3)
    finally:
        viz.SURROGATES.clear()


def test_demo_registers_all_four_selected_models_and_they_track_rk4():
    import numpy as np

    from src import demo, viz

    files = demo.checkpoints("base")
    assert sorted(files) == ["DeepONet coef", "DeepONet samples", "MLP coef", "MLP samples"]
    if not all((demo.CKPT_DIR / f).exists() for f in files.values()):
        pytest.skip("checkpoints not available")
    demo.register_surrogates(files)
    try:
        app = viz.App(family="smooth")
        for name, tr in app.surr.items():
            assert tr.shape == app.traj.shape and np.isfinite(tr).all(), name
            assert app.rel_l2(tr) < 0.5, name  # loose: catches a wrong input path, not accuracy
        app.run(max_frames=3)
    finally:
        viz.SURROGATES.clear()


def test_demo_rejects_unknown_model_key():
    from src import demo

    with pytest.raises(ValueError, match="unknown models"):
        demo.checkpoints("ff6cons", ["deeponet-coef"])  # ff6cons has samples-input models only


def test_unknown_family_is_rejected():
    from src.viz import App

    with pytest.raises(ValueError, match="family"):
        App(family="bogus")
