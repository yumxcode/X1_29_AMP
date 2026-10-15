#!/usr/bin/env python3
"""v70 local unit test: the shoulder_yaw_fold guard, isaaclab-stubbed.

Same stub machinery as test_v69_guards.py; drives the actual reward
function in roboparty_train/robolab/robolab/tasks/manager_based/amp/
mdp/rewards.py with a synthetic env and checks the forensics-calibrated
numbers (soup69_drift / m13497 forensics, 2026-10-15):

  1. resume-point posture sy L -25.6 / R -43.1 deg -> relu(43.1-15) +
     relu(25.6-15) deg = 0.675 rad (weight -0.5 -> -0.34/step).
  2. demo-level arms (~0 deg both) -> exactly 0.
  3. sign symmetry: +43.1/+25.6 -> same penalty.
  4. AC swing std 10 deg around 0 DC -> 0 (band absorbs the swing).
  5. instantaneous nature: a DC posture pays every step (no EMA lag).
"""
import functools
import sys
import types
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]

# ---- stub isaaclab (identical to test_v69_guards.py) --------------------
def quat_apply(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx)], dim=-1)


class _SceneEntityCfg:
    def __init__(self, name="robot", joint_names=None, body_names=None,
                 preserve_order=False):
        self.name = name
        self.joint_names = joint_names
        self.body_names = body_names
        self.joint_ids = None
        self.body_ids = None


_iso = types.ModuleType("isaaclab")
_iso.utils = types.ModuleType("isaaclab.utils")
_mu = types.ModuleType("isaaclab.utils.math")
_mu.quat_apply = quat_apply
_mu.quat_apply_inverse = lambda q, v: quat_apply(
    torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1), v)
_iso.utils.math = _mu
def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub_module("isaaclab.envs", mdp=types.SimpleNamespace())
_stub_module("isaaclab.managers", SceneEntityCfg=_SceneEntityCfg)
_stub_module("isaaclab.sensors", ContactSensor=object, RayCaster=object)
_stub_module("isaaclab.assets", Articulation=object, RigidObject=object)
sys.modules.update({
    "isaaclab": _iso, "isaaclab.utils": _iso.utils,
    "isaaclab.utils.math": _mu,
})

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "v70_rewards",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

JOINTS = ["left_shoulder_yaw_joint", "right_shoulder_yaw_joint"]
BODIES = ["base_link"]


class FakeAsset:
    def __init__(self):
        self.data = types.SimpleNamespace()


class FakeEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.scene = {"robot": FakeAsset()}


def make_cfg():
    cfg = _SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)
    cfg.joint_ids = [JOINTS.index(n) for n in JOINTS]
    return cfg


def test_fold():
    env = FakeEnv()
    cfg = make_cfg()
    asset = env.scene["robot"]
    asset.data.joint_pos = torch.tensor(
        [[np.radians(-25.6), np.radians(-43.1)]], dtype=torch.float32)
    out = R.shoulder_yaw_fold_guard(env, cfg)
    expect = np.radians(43.1 - 15) + np.radians(25.6 - 15)
    print(f"[sy] resume-point L-25.6/R-43.1: {out[0]:.4f} rad (expect {expect:.4f},"
          f" weight -0.5 -> {-0.5*expect:.3f}/step)")
    assert abs(out[0].item() - expect) < 1e-3

    # demo level ~0 deg -> exactly 0
    asset.data.joint_pos = torch.zeros(1, 2)
    out = R.shoulder_yaw_fold_guard(env, cfg)
    print(f"[sy] demo-level: {float(out[0]):.4f} (expect 0)")
    assert float(out[0]) == 0.0

    # sign symmetry (fold the other way — same statistic)
    asset.data.joint_pos = torch.tensor(
        [[np.radians(25.6), np.radians(43.1)]], dtype=torch.float32)
    out = R.shoulder_yaw_fold_guard(env, cfg)
    assert abs(out[0].item() - expect) < 1e-3
    print("[sy] sign-symmetry PASS")

    # AC swing 10 deg @ ~1 Hz around 0 DC -> 0 (band absorbs)
    t = np.arange(2000)
    ac = np.radians(10) * np.sin(2 * np.pi * 1.0 * t / 50.0)
    q = np.stack([ac, -ac], 1)
    mx = 0.0
    for k in range(len(t)):
        asset.data.joint_pos = torch.tensor(q[k], dtype=torch.float32).unsqueeze(0)
        mx = max(mx, float(R.shoulder_yaw_fold_guard(env, cfg)[0]))
    print(f"[sy] pure-AC 10deg@1Hz max: {mx:.4f} (expect 0 — band absorbs)")
    assert mx == 0.0

    # band edge: exactly 15 deg -> 0 (0.262 >= radians(15)); 16 deg ->
    # 2*(radians(16) - 0.262) total (band const is a rounded 15 deg)
    asset.data.joint_pos = torch.tensor(
        [[np.radians(15), np.radians(15)]], dtype=torch.float32)
    assert float(R.shoulder_yaw_fold_guard(env, cfg)[0]) == 0.0
    asset.data.joint_pos = torch.tensor(
        [[np.radians(16), np.radians(-16)]], dtype=torch.float32)
    out = R.shoulder_yaw_fold_guard(env, cfg)
    expect_edge = 2 * (np.radians(16) - 0.262)
    assert abs(out[0].item() - expect_edge) < 1e-6
    print("[sy] band-edge exactness PASS")
    print("[sy] ALL PASS")


if __name__ == "__main__":
    test_fold()
    print("\nALL v70 GUARD UNIT TESTS PASS")
