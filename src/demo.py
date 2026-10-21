"""Live demo: RK4 ground truth next to the trained DeepONets and MLPs in the pygame viewer.

Run with ``python -m src.demo`` from the repo root. It needs ``data/dataset_smooth.npz`` and the
checkpoints listed in ``scripts/final_eval.py``. Keys are those of ``src.viz`` (N loads a random test
sample, O a random out-of-range one, sliders change m_p, b and the peak torque, drag on a plot to scrub).

The default ``--variant base`` shows the four models picked by the selection rule fixed in
``docs/EXPERIMENTS.md`` (DeepONet / MLP x coef / samples input, lowest validation loss, seed 3001).
``--variant ff6`` / ``ff6cons`` show the E4 follow-ups (Fourier time features, + consistency penalty);
those were evaluated after selection and are not a re-selection. ``--models`` picks a subset,
e.g. ``--models deeponet-coef,mlp-coef``. Test errors per setting are in ``results/final_eval*.md``.
"""
import argparse
from pathlib import Path

from scripts.final_eval import VARIANTS

from . import viz
from .data import smooth_torque_samples
from .surrogate import Surrogate

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SEED = 3001
NAMES = {"deeponet": "DeepONet", "mlp": "MLP"}


def checkpoints(variant: str = "base", models=None, seed: int = SEED) -> dict:
    """``{display name: checkpoint file}`` for one variant of ``scripts.final_eval``, optionally a subset.

    ``models`` is an iterable of ``"<model>-<input>"`` keys such as ``"deeponet-coef"``."""
    table = VARIANTS["" if variant == "base" else variant]
    keys = {f"{m}-{i}": (m, i) for m, i in table}
    if models is not None:
        unknown = set(models) - set(keys)
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)} for variant {variant!r}; choose from {sorted(keys)}")
        keys = {k: keys[k] for k in models}
    return {f"{NAMES[m]} {i}": f"{table[m, i].format(s=seed)}.pt" for m, i in keys.values()}


def register_surrogates(files: dict, ckpt_dir: Path = CKPT_DIR) -> None:
    """Load each checkpoint and register it in ``viz.SURROGATES`` as ``f(params, coef) -> (N_T, 4)``.

    The viewer always hands over Fourier coefficients (2, 2, K); samples-input models get the torque
    sampled on the 151-point grid instead, which is exactly what they were trained on."""
    for name, file in files.items():
        sur = Surrogate.from_checkpoint(ckpt_dir / file)
        if sur.input_mode == "samples":
            f = lambda params, coef, sur=sur: sur.predict(params, smooth_torque_samples(coef)[None])[0]
        else:
            f = lambda params, coef, sur=sur: sur.predict(params, coef[None])[0]
        viz.SURROGATES[name] = f


def main(argv=None):
    """Open the viewer with the chosen surrogates registered (``--frames N --screenshot PATH`` for a headless check)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=["base", "ff6", "ff6cons"], default="base",
                    help="base = selection-rule models; ff6 / ff6cons = E4 follow-ups")
    ap.add_argument("--models", type=lambda s: s.split(","), default=None,
                    help="comma-separated subset, e.g. deeponet-coef,mlp-samples (default: all of the variant)")
    ap.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--screenshot", type=Path, default=None, help="save the last frame here (needs --frames)")
    args = ap.parse_args(argv)
    register_surrogates(checkpoints(args.variant, args.models))
    viz.App(family="smooth").run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
