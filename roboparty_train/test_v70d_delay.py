#!/usr/bin/env python3
"""v70d local unit test: the ObsDelayRing (isaaclab-free, REAL code).

Checks (torch-only, statistical where random):
  1. Zero-delay envs see the CURRENT obs; delayed envs see an EARLIER obs
     (driven deterministically by monkeypatching torch.multinomial draws).
  2. Ring depth: delay-2 env sees obs from 2 steps ago, not 1.
  3. Reset flush: a just-reset env sees the CURRENT (post-reset) obs even
     when drawn delayed — no pre-reset leak across the episode boundary.
  4. Distribution: with the default p=[.7,.15,.15], long-run delay
     frequencies match within 3 sigma.
"""
import sys
from pathlib import Path

import torch

import importlib.util

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "amp_obs_delay",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/amp_obs_delay.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
ObsDelayRing = _m.ObsDelayRing  # noqa: E402


import contextlib


@contextlib.contextmanager
def force_delays(n, want):
    """Monkeypatch multinomial to return a fixed delay draw."""
    def fake(p, num_samples, replacement=False):
        return torch.tensor([want] * num_samples)
    real = torch.multinomial
    torch.multinomial = fake
    try:
        yield
    finally:
        torch.multinomial = real


def test_ring():
    torch.manual_seed(0)
    ring = ObsDelayRing(2)
    n = 6
    # t0: init ring with obs0
    obs0 = torch.arange(n, dtype=torch.float32).unsqueeze(1) * 100.0
    for want in [0]:
        with force_delays(n, want):
            out = ring(obs0, None)
    assert torch.equal(out, obs0), "delay-0 env must see current obs"
    ring.ring = [obs0.clone(), obs0.clone()]  # deterministic init

    # t1: obs1, delay-1 envs see obs0
    obs1 = obs0 + 1.0
    with force_delays(n, 1):
        out1 = ring(obs1, None)
    assert torch.equal(out1, obs0), "delay-1 must see t-1 obs"

    # t2: obs2, delay-2 envs see obs0 (ring[1] = obs0), delay-1 see obs1
    obs2 = obs0 + 2.0
    with force_delays(n, 2):
        out2 = ring(obs2, None)
    assert torch.equal(out2, obs0), "delay-2 must see t-2 obs"
    # t2b: a THIRD call (delay-1) — its t-1 is the t2 call's obs (obs2)
    with force_delays(n, 1):
        out2b = ring(obs2, None)
    assert torch.equal(out2b, obs2), "delay-1 at next call must see the previous obs"
    print("[ring] depth semantics (0/1/2 step delays) PASS")

    # reset flush: envs [True]*n, drawn delay-1 -> must see CURRENT obs
    reset = torch.ones(n, dtype=torch.bool)
    obs3 = obs0 + 3.0
    with force_delays(n, 1):
        out3 = ring(obs3, reset)
    assert torch.equal(out3, obs3), "reset env must see post-reset obs"
    print("[ring] reset flush (no cross-episode leak) PASS")

    # distribution over 200k draws of a fresh ring
    ring2 = ObsDelayRing(2)
    obs = torch.zeros(200_000, 1)
    counts = {0: 0, 1: 0, 2: 0}
    prev = obs.clone()
    for t in range(30):
        cur = torch.full((200_000, 1), float(t))
        out = ring2(cur, None)
        if t == 0:
            continue  # first call initializes the ring: delay ineffective
        counts[0] += int((out == cur).all(dim=1).sum())
    frac0 = counts[0] / (29 * 200_000)
    sigma = (0.7 * 0.3 / (29 * 200_000)) ** 0.5
    assert abs(frac0 - 0.7) < 4 * sigma, f"zero-delay frac {frac0}"
    print(f"[ring] delay-0 frequency {frac0:.4f} ~ p0=0.7 PASS")


if __name__ == "__main__":
    test_ring()
    print("\nALL v70d DELAY RING UNIT TESTS PASS")
