"""v63 action low-pass — the torch-side cascade (isaaclab-free on purpose).

Single source of truth for the TRAINING-side filter (AmpEnv._action_lpf
delegates here). The DEPLOY-side twin lives inline in
sim2sim/mujoco_rollout.py (numpy; it must stay importable with only
PYTHONPATH=pylibs so it cannot import robolab) — the v63 dry-run drives
BOTH formulas on identical signals and asserts numerical parity plus the
analytic -3 dB/section attenuation, so the two cannot drift.

Rationale (user suggestion #1, contract rev4): at 100 Hz the policy can
synthesize action content up to 50 Hz that the ~50 Hz demos (human
bandwidth ~< 10 Hz) never contain. The filter caps the policy's effective
action bandwidth at the demo band while the PD servo keeps its 100 Hz
correction bandwidth ("frequency-hierarchy alignment": 伪造高频细节的
能力被没收, PD 伺服的 100 Hz 好处保留).
"""
from __future__ import annotations

import math

import torch


def lpf_alpha(hz: float, dt: float) -> float:
    """Single-section smoothing factor of a discrete 1st-order low-pass
    with cutoff ``hz`` at sample period ``dt`` (impulse-invariant mapping
    of exp(-t/tau), tau = 1/(2*pi*hz))."""
    return 1.0 - math.exp(-2.0 * math.pi * hz * dt)


class TorchActionLPF:
    """Cascaded first-order sections applied to raw policy actions.

    y_i[t] = alpha * y_{i-1}[t] + (1 - alpha) * y_i[t-1], y_0 = x.
    order=2 -> 12 dB/oct rolloff above the cutoff; DC gain exactly 1.
    State survives env resets by design (action sequences are continuous
    across them; the state time constant at 10 Hz is ~16 ms).
    """

    def __init__(self, hz: float, order: int, dt: float):
        self.hz = float(hz)
        self.order = max(1, int(order))
        self.dt = float(dt)
        self.alpha = lpf_alpha(hz, dt) if hz > 0 else None

    @property
    def enabled(self) -> bool:
        return self.alpha is not None

    def reset_with(self, action: torch.Tensor) -> None:
        self._state = [action.clone() for _ in range(self.order)]

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return action
        if not hasattr(self, "_state"):
            self.reset_with(action)
        out = action
        for i in range(self.order):
            self._state[i].mul_(1.0 - self.alpha).add_(out * self.alpha)
            out = self._state[i]
        return out
