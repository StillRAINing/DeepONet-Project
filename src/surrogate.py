"""User-facing entry point: a trained surrogate that validates its inputs and works in raw units.

    sur = Surrogate.from_checkpoint("checkpoints/<name>.pt")
    traj = sur.predict(params, torque)      # (B, n_t, 4) = [q1, q2, q1dot, q2dot], rad and rad/s

Everything the network needs (input z-scoring, time rescaling to [0, 1], decoding to raw units) is
done here, and every input goes through ``validate`` first, so a bad call fails with a clear
message instead of a shape error deep in PyTorch or a silent NaN. Inputs that are valid but outside
the trained range emit ``validate.OutOfRangeWarning`` (or raise with ``strict=True``).
"""
import numpy as np
import torch

from .data import SMOOTH_RANGES, T_GRID, T_FINAL, smooth_torque_samples
from .train import COEF_MODES, load_checkpoint
from .validate import validate_params, validate_time, validate_torque


class Surrogate:
    """A trained DeepONet or MLP together with its normalisers.

    Args:
        model: module with ``model(u, t) -> (B, n_t, 4)``, in eval mode.
        in_norm: ``train.InputNormaliser`` fitted on the training split.
        out_norm: ``train.ChannelNormaliser`` fitted on the training split.
        input_mode: ``"samples"`` (torque sampled on the 151-point grid) or ``"coef"`` (the 2 x 2 x K
            Fourier coefficients of the smooth torque family).
    """

    def __init__(self, model, in_norm, out_norm, input_mode: str = "samples"):
        if input_mode not in ("samples", "coef"):
            raise ValueError(f"input_mode must be 'samples' or 'coef'; got {input_mode!r}")
        self.model, self.in_norm, self.out_norm, self.input_mode = model, in_norm, out_norm, input_mode

    @classmethod
    def from_checkpoint(cls, path) -> "Surrogate":
        """Load a checkpoint written by ``train``. Raises ``FileNotFoundError`` if it is missing."""
        model, in_norm, out_norm, hist = load_checkpoint(path)
        return cls(model, in_norm, out_norm, hist.get("input_mode", "samples"))

    def _branch_input(self, params: np.ndarray, torque_or_coef, strict: bool) -> np.ndarray:
        """Validate and assemble the raw branch input (B, d_u)."""
        if self.input_mode == "samples":
            torque = validate_torque(torque_or_coef, strict=strict)
            # branch input = [m_p, b, 151 samples of tau1, 151 samples of tau2]
            return np.concatenate([params, torque.reshape(len(torque), -1)], axis=1)
        coef = np.asarray(torque_or_coef, dtype=np.float64)
        if coef.ndim == 3:
            coef = coef[None]
        if coef.ndim != 4 or coef.shape[1:3] != (2, 2) or coef.shape[3] > COEF_MODES:
            raise ValueError(f"coef must have shape (B, 2, 2, K) with K <= {COEF_MODES}; got {coef.shape}")
        if not np.all(np.isfinite(coef)):
            raise ValueError("coef contains NaN or infinite values")
        validate_torque(np.stack([smooth_torque_samples(c) for c in coef]), strict=strict)  # peak check
        # zero-pad to the 13 modes the compact models were trained with
        coef = np.pad(coef, [(0, 0), (0, 0), (0, 0), (0, COEF_MODES - coef.shape[3])])
        return np.concatenate([params, coef.reshape(len(coef), -1)], axis=1)

    @torch.no_grad()
    def predict(self, params, torque_or_coef, t=None, *, strict: bool = False) -> np.ndarray:
        """Predict trajectories in raw units.

        Args:
            params: ``[m_p, b]`` (2,) or (B, 2); kg and N m s/rad.
            torque_or_coef: torque samples (B, 2, 151) in N m for ``input_mode="samples"``, or
                smooth-family coefficients (B, 2, 2, K) for ``"coef"``.
            t: query times in seconds, 1-D. Default: the 151-point grid on [0, 3] s.
            strict: raise ``ValueError`` instead of warning when an input is out of range.

        Returns:
            (B, n_t, 4) array, channels ``[q1, q2, q1dot, q2dot]`` in rad and rad/s.

        Raises:
            ValueError: wrong shape, NaN/infinite, negative m_p or b, or (strict) out of range.
        """
        # 1. validate every input before the model is called
        p = validate_params(params, ranges=SMOOTH_RANGES, strict=strict)
        t_s = T_GRID if t is None else validate_time(t, strict=strict)
        raw = self._branch_input(p, torque_or_coef, strict)
        if len(raw) != len(p):
            raise ValueError(f"params has {len(p)} rows but the torque input has {len(raw)}")
        # 2. z-score the inputs with the training statistics; time goes in as t / T
        u = self.in_norm.encode(torch.as_tensor(raw, dtype=torch.float32))
        # 3. run the network and undo the output scaling to get rad and rad/s
        out = self.out_norm.decode(self.model(u, torch.as_tensor(t_s / T_FINAL, dtype=torch.float32)))
        return out.numpy().astype(np.float64)
