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
