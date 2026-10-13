"""Live demo: RK4 ground truth next to the trained DeepONet and MLP in the pygame viewer.

Run with ``python -m src.demo`` from the repo root. It needs ``data/dataset_smooth.npz`` and the
two checkpoints below. Keys are those of ``src.viz`` (N loads a random test sample, O a random
out-of-range one, sliders change m_p, b and the peak torque, drag on a plot to scrub).

The two networks are the "compact coefficient" models picked by the selection rule fixed in
``docs/EXPERIMENTS.md`` (lowest validation loss, seed 3001). Both take the same 2 + 24 numbers as
input, so the comparison is fair. Note that the MLP has the lower test error (``results/final_eval.md``).
"""
import argparse
from pathlib import Path

import numpy as np

from . import viz
from .surrogate import Surrogate

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINTS = {
    "DeepONet": "deeponet_p64_w147_ep200_lr0.003_n7000_seed3001_coef_cosine.pt",
    "MLP": "mlp_w229_ep200_lr0.003_n7000_seed3001_coef_cosine.pt",
}


def register_surrogates(ckpt_dir: Path = CKPT_DIR) -> None:
    """Load each checkpoint and register it in ``viz.SURROGATES`` as ``f(params, coef) -> (N_T, 4)``."""
    for name, file in CHECKPOINTS.items():
        sur = Surrogate.from_checkpoint(ckpt_dir / file)
        viz.SURROGATES[name] = lambda params, coef, sur=sur: sur.predict(params, coef[None])[0]


def main(argv=None):
    """Open the viewer with the DeepONet and MLP registered (``--frames N --screenshot PATH`` for a headless check)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--screenshot", type=Path, default=None, help="save the last frame here (needs --frames)")
    args = ap.parse_args(argv)
    register_surrogates()
    viz.App(family="smooth").run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
