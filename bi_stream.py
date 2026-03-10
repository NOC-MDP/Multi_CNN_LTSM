from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from bi_model import BifurcationRegressor
# ──────────────────────────────────────────────────────────────────────────────
# Streaming inference — rolling window alerting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BifurcationAlert:
    """Fired when the model believes a bifurcation is imminent or occurring."""
    absolute_timestep: int          # current position in the stream
    p_bifurcation: float            # model confidence [0,1]
    predicted_bifurc_t: int         # predicted absolute timestep of bifurcation
    steps_until_bifurc: int         # predicted_bifurc_t - absolute_timestep
                                    #   negative = already occurred in this window


class StreamingBifurcationDetector:
    """
    Wraps a trained BifurcationRegressor for real-time rolling-window inference.

    Usage
    ─────
        detector = StreamingBifurcationDetector(model, window_len=128, stride=16)
        for new_slice in data_stream:           # new_slice: (P, D, stride)
            alerts = detector.push(new_slice)
            for alert in alerts:
                handle_alert(alert)

    Hysteresis
    ──────────
    An alert fires when p_bifurc crosses `threshold` from below.
    The detector then enters a cooldown of `cooldown_steps` timesteps
    before it can fire again — preventing a cascade of duplicate alerts
    from overlapping windows that all detect the same bifurcation.

    Exponential smoothing
    ─────────────────────
    Raw per-window probabilities are noisy.  We apply an EMA (alpha controls
    responsiveness vs. smoothness) before threshold comparison.
    """

    def __init__(
        self,
        model: BifurcationRegressor,
        window_len: int = 128,
        stride: int = 16,
        threshold: float = 0.6,
        cooldown_steps: int = 64,
        ema_alpha: float = 0.4,
        device: Optional[torch.device] = None,
        normalise: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.window_len = window_len
        self.stride = stride
        self.threshold = threshold
        self.cooldown_steps = cooldown_steps
        self.alpha = ema_alpha
        self.device = device or torch.device("cpu")
        self.normalise = normalise

        # Rolling buffer: deque of (P, D, 1) slices
        self._buffer: deque = deque()
        self._buffer_len = 0        # total timesteps currently buffered
        self._absolute_t = 0        # absolute timestep of the newest sample

        # State
        self._ema_p = 0.0
        self._cooldown_remaining = 0
        self._prev_smoothed = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def push(self, new_data: np.ndarray) -> list[BifurcationAlert]:
        """
        Ingest new_data of shape (num_params, depth, stride_len) and
        return any alerts triggered by this update.

        new_data may contain fewer than `stride` timesteps (e.g. final chunk).
        """
        assert new_data.ndim == 3, "new_data must be (P, D, T_chunk)"
        chunk_len = new_data.shape[2]

        self._buffer.append(new_data)
        self._buffer_len += chunk_len
        self._absolute_t += chunk_len

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= chunk_len

        # Only run inference once we have a full window
        if self._buffer_len < self.window_len:
            return []

        # Assemble window from buffer (most recent window_len timesteps)
        window = self._assemble_window()
        p_raw, t_norm = self._infer(window)

        # EMA smoothing
        self._ema_p = self.alpha * p_raw + (1 - self.alpha) * self._ema_p

        alerts = []

        # Threshold crossing with hysteresis
        crossed = (self._prev_smoothed < self.threshold <= self._ema_p)
        if crossed and self._cooldown_remaining <= 0:
            t_window_start = self._absolute_t - self.window_len

            # Convert normalised t_norm back to absolute timestep
            predicted_abs = int(t_window_start + t_norm * self.window_len)
            steps_until  = predicted_abs - self._absolute_t

            alerts.append(BifurcationAlert(
                absolute_timestep    = self._absolute_t,
                p_bifurcation        = self._ema_p,
                predicted_bifurc_t   = predicted_abs,
                steps_until_bifurc   = steps_until,
            ))
            self._cooldown_remaining = self.cooldown_steps

        self._prev_smoothed = self._ema_p
        return alerts

    def reset(self):
        """Clear buffer and state (e.g. between independent recordings)."""
        self._buffer.clear()
        self._buffer_len = 0
        self._absolute_t = 0
        self._ema_p = 0.0
        self._cooldown_remaining = 0
        self._prev_smoothed = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _assemble_window(self) -> np.ndarray:
        """Concatenate buffer chunks and take the last window_len timesteps."""
        full = np.concatenate(list(self._buffer), axis=2)   # (P, D, total_t)
        window = full[:, :, -self.window_len:]               # (P, D, window_len)

        # Trim stale chunks from the front of the buffer to save memory
        keep = self.window_len + self.stride
        if full.shape[2] > keep:
            # Drop chunks until we keep only what's needed
            trimmed = 0
            while self._buffer and trimmed + self._buffer[0].shape[2] <= full.shape[2] - keep:
                trimmed += self._buffer[0].shape[2]
                self._buffer.popleft()
            self._buffer_len = sum(c.shape[2] for c in self._buffer)

        return window

    @torch.no_grad()
    def _infer(self, window: np.ndarray) -> tuple[float, float]:
        x = torch.from_numpy(window).float().unsqueeze(0)   # (1, P, D, T)

        if self.normalise:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std  = x.std(dim=(1, 2, 3),  keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std

        x = x.to(self.device)
        p_pred, t_pred = self.model(x)

        return p_pred.item(), t_pred.item()
