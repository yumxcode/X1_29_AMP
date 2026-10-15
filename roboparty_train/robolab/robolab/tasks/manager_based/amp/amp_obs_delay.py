# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""v70d: per-env stochastic OBSERVATION pipeline delay ring.

Isaaclab-free on purpose (the amp_action_lpf.py pattern) so the local
dry-run/unit test drives the REAL ring. AmpEnv._obs_delay delegates here.

The training-side twin of sim2sim mujoco_rollout --latency-steps: the
policy obs group is replaced by the obs from N control steps ago, with a
per-env random delay drawn each step (bias toward 0). Ring entries are
flushed for just-reset envs so fresh episodes start un-delayed.
"""
from __future__ import annotations

import torch


class ObsDelayRing:
    def __init__(self, max_delay: int, p_zero: float = 0.7):
        assert max_delay >= 1
        self.max_delay = max_delay
        self.p = [p_zero] + [(1.0 - p_zero) / max_delay] * max_delay
        self.ring: list[torch.Tensor] | None = None

    def __call__(self, obs: torch.Tensor, reset_buf: torch.Tensor | None) -> torch.Tensor:
        if self.ring is None:
            self.ring = [obs.clone() for _ in range(self.max_delay)]
        if reset_buf is not None and reset_buf.any():
            for d in range(self.max_delay):
                self.ring[d][reset_buf] = obs[reset_buf]
        p = torch.tensor(self.p, device=obs.device)
        delays = torch.multinomial(p, obs.shape[0], replacement=True)
        out = obs.clone()
        for d in range(1, self.max_delay + 1):
            m = delays == d
            if m.any():
                out[m] = self.ring[d - 1][m]
        # update ring AFTER selection: head = most recent past obs
        self.ring.insert(0, obs.clone())
        self.ring.pop()
        return out
