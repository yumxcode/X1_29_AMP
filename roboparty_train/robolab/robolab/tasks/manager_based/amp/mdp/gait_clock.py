# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
"""v71: EXOGENOUS GAIT CLOCK — the rhythm command channel (isaaclab-free).

18 falsified rounds (V64-V70, see acceptance/X1_GAIT_SPEC_V71.md §6) proved
that within the AMP+RL stack no REWARD-SIDE anchoring flips the 2.2 Hz
fast-step attractor: whenever the policy chooses "when to step" endogenously,
micro-stepping stays the easy balance optimum. This module makes the gait
timing EXOGENOUS — a clock the policy OBSERVES and a per-foot target profile
the task scores against:

    T*(v) = clamp(1.46 - 0.36 v, 0.70, 1.75)   single-foot cycle [s]
    d*(v) = clip(0.68 - 0.07 v, 0.55, 0.70)    duty factor (stance fraction)
    h*(v) = clip(0.025 + 0.025 v, 0.025, 0.075) swing-apex foot height [m]

    phase:   phi_L = (t / T*(v)) % 1 ;  phi_R = (phi_L + 0.5) % 1
             stance window of foot i:  phi_i in [0, d*) ;  swing: [d*, 1)
    height:  z*(phi_i) = h* * sin(pi * (phi_i - d*) / (1 - d*))  on swing,
             0 on stance  (C0-continuous bell, apex mid-swing)

t is EPISODE time (episode_length_buf * step_dt) — pure function, no state:
reset-stable, exactly reproducible by the sim2sim/deploy twin
(sim2sim/mujoco_rollout.py builds the same channels from its loop counter).

X1 physical calibration (FK-verified 2026-10-17): L = 0.70 m effective leg;
Froude-scaled natural cadence ~1.0 Hz/foot; T* fit on the retargeted demo
library (36_01 1.03 s @1.0 m/s ...); h* inside the demo clearance band
(22-63 mm) with margin over the 25 mm drag gate.
"""
from __future__ import annotations

import torch

# --- schedule constants (single source of truth; mirrored in sim2sim) -----
T_STAR_A, T_STAR_B = 1.46, 0.36
T_STAR_LO, T_STAR_HI = 0.70, 1.75
D_A, D_B = 0.68, 0.07
D_LO, D_HI = 0.55, 0.70
H_A, H_B = 0.025, 0.025
H_LO, H_HI = 0.025, 0.075
# contact-timing hinge windows (fractions INSIDE each desired phase)
SWING_MID_LO, SWING_MID_HI = 0.2, 0.8   # of the swing span
STANCE_MID_LO, STANCE_MID_HI = 0.15, 0.85  # of the stance span


def t_star(v: torch.Tensor) -> torch.Tensor:
    """Single-foot gait-cycle period target at planar command speed v."""
    return (T_STAR_A - T_STAR_B * v).clamp(min=T_STAR_LO, max=T_STAR_HI)


def duty_star(v: torch.Tensor) -> torch.Tensor:
    """Stance duty-factor target at speed v."""
    return (D_A - D_B * v).clamp(min=D_LO, max=D_HI)


def apex_star(v: torch.Tensor) -> torch.Tensor:
    """Swing-apex foot-height target [m] at speed v."""
    return (H_A + H_B * v).clamp(min=H_LO, max=H_HI)


def phases(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-foot clock phases, shape (N, 2) = [phi_L, phi_R].

    Left foot's cycle starts (touchdown) at episode t=0; right is anti-phase
    by construction (+0.5). Pure function of episode time + command speed.
    """
    phi_l = torch.remainder(t / t_star(v), 1.0)
    phi_r = torch.remainder(phi_l + 0.5, 1.0)
    return torch.stack([phi_l, phi_r], dim=-1)


def clock_obs(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Policy observation channels, exact layout (N, 4):
    [sin(2pi phi_L), sin(2pi phi_R), cos(2pi phi_L), cos(2pi phi_R)]
    — the sim2sim/deploy twin must reproduce THIS order.

    The clock runs at ALL command speeds (at v=0, T*=1.46 s); the clock
    REWARDS are gated off below their own min_cmd_speed — the policy learns
    to ignore the rotating clock when the command says stand.
    """
    phi = phases(t, v)
    return torch.cat([torch.sin(2 * torch.pi * phi),
                      torch.cos(2 * torch.pi * phi)], dim=-1)


def target_height(phi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Desired lowest-sole-point height z*(phi_i) per foot, shape (N, 2).

    0 through the stance window; raised C0 bell peaking at h*(v) mid-swing.
    """
    d = duty_star(v).unsqueeze(-1)
    h = apex_star(v).unsqueeze(-1)
    frac = (phi - d) / (1.0 - d).clamp(min=1e-6)          # <0 stance, [0,1) swing
    bell = torch.sin(torch.pi * frac.clamp(0.0, 1.0))
    return torch.where(frac > 0.0, h * bell, torch.zeros_like(bell))


def _window_mask(phi: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (phi >= lo) & (phi < hi)


def swing_mid_mask(phi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """True where the foot SHOULD be mid-swing (contact here = violation)."""
    d = duty_star(v).unsqueeze(-1)
    span = (1.0 - d).clamp(min=1e-6)
    return _window_mask(phi, d + SWING_MID_LO * span, d + SWING_MID_HI * span)


def stance_mid_mask(phi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """True where the foot SHOULD be mid-stance (airborne here = violation)."""
    d = duty_star(v).unsqueeze(-1)
    return _window_mask(phi, STANCE_MID_LO * d, STANCE_MID_HI * d)
