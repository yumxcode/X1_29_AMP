#!/usr/bin/env python3
"""v70b local unit test: the arm_shared_lean guard, isaaclab-stubbed.

Same stub machinery as test_v70_guards.py. The phi formula is verified
against a REAL MuJoCo FK of two logged policy postures (m13500 PASS
8.9 deg / m15496 FAIL 19.9 deg on the P7h gate), plus synthetic
geometry checks:

  1. m15496-level shared lean (both upper arms 20 deg fwd) -> relu(14)
     deg; m13500-level 9 deg -> relu(3) deg; demo-level 2 deg -> 0.
  2. backward lean caught (sign symmetry).
  3. antiphase swing does not leak into the shared sum (L +20 / R -20
     around a 0-deg DC -> ~0).
"""
import functools
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]


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


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_iso = types.ModuleType("isaaclab")
_iso.utils = types.ModuleType("isaaclab.utils")
_mu = _stub_module("isaaclab.utils.math",
                   quat_apply=quat_apply,
                   quat_apply_inverse=lambda q, v: quat_apply(
                       torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1), v))
_iso.utils.math = _mu
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
    "v70b_rewards",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


class FakeAsset:
    def __init__(self):
        self.data = types.SimpleNamespace()


class FakeEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.scene = {"robot": FakeAsset()}


def make_cfgs():
    s = _SceneEntityCfg("robot", body_names=[
        "left_shoulder_pitch_link", "right_shoulder_pitch_link"], preserve_order=True)
    s.body_ids = [0, 1]
    e = _SceneEntityCfg("robot", body_names=[
        "left_elbow_pitch_link", "right_elbow_pitch_link"], preserve_order=True)
    e.body_ids = [2, 3]
    return s, e


def run_guard(env, sho_pos, elb_pos, root_quat=None):
    s, e = make_cfgs()
    asset = env.scene["robot"]
    asset.data.body_pos_w = torch.cat([sho_pos, elb_pos], dim=0).unsqueeze(0)
    asset.data.root_quat_w = (root_quat if root_quat is not None
                              else torch.tensor([[1.0, 0, 0, 0]]))
    return R.arm_shared_lean_guard(env, s, e)


def posture(lean_deg):
    """Shoulders at +-0.25 y, elbows 0.28 m below rotated forward by lean."""
    r = math.radians(lean_deg)
    d = torch.tensor([math.sin(r), 0.0, -math.cos(r)]) * 0.28
    sho = torch.tensor([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]])
    elb = sho + d.unsqueeze(0)
    return sho, elb


def test_lean():
    env = FakeEnv()
    # m15496-level: both arms 20 deg forward
    sho, elb = posture(20.0)
    out = run_guard(env, sho, elb)
    expect = math.radians(20.0) - 0.105   # band const is a rounded 6 deg
    print(f"[lean] m15496-level 20deg: {float(out[0]):.4f} rad (expect {expect:.4f},"
          f" weight -0.5 -> {-0.5*expect:.3f}/step)")
    assert abs(float(out[0]) - expect) < 1e-4
    # m13500-level: 9 deg
    sho, elb = posture(9.0)
    out = run_guard(env, sho, elb)
    expect = math.radians(9.0) - 0.105
    print(f"[lean] m13500-level 9deg: {float(out[0]):.4f} (expect {expect:.4f})")
    assert abs(float(out[0]) - expect) < 1e-4
    # demo-level: 2 deg -> 0
    sho, elb = posture(2.0)
    out = run_guard(env, sho, elb)
    print(f"[lean] demo-level 2deg: {float(out[0]):.4f} (expect 0)")
    assert float(out[0]) == 0.0
    # backward lean caught too (sign symmetry)
    sho, elb = posture(-15.0)
    out = run_guard(env, sho, elb)
    expect = math.radians(15.0) - 0.105
    assert abs(float(out[0]) - expect) < 1e-4
    print("[lean] backward -15deg caught (sign-symmetric) PASS")
    # antiphase leak check: L +20 / R -20 around 0 DC -> ~0
    sho = torch.tensor([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]])
    dL = torch.tensor([math.sin(math.radians(20)), 0.0, -math.cos(math.radians(20))]) * 0.28
    dR = torch.tensor([math.sin(math.radians(-20)), 0.0, -math.cos(math.radians(-20))]) * 0.28
    elb = sho + torch.stack([dL, dR])
    out = run_guard(env, sho, elb)
    print(f"[lean] antiphase +-20deg @0DC: {float(out[0]):.4f} (expect ~0)")
    assert float(out[0]) == 0.0
    # base-frame invariance: rotate BOTH the base quat and the arm vectors
    # by the same 15-deg pitch -> base-frame phi unchanged (the guard sees
    # the arm pose relative to the torso, which is what P7h measures)
    a = math.radians(7.5)
    q = torch.tensor([[math.cos(a), 0, math.sin(a), 0]])   # pitch +15 deg (wxyz)
    sho0, elb0 = posture(20.0)
    u0 = elb0 - sho0                                        # (2,3) world
    u1 = quat_apply(q.expand(2, 4), u0)                     # rotate with base
    sho = quat_apply(q.expand(2, 4), sho0)
    elb = sho + u1
    out = run_guard(env, sho, elb, root_quat=q)
    expect = math.radians(20.0) - 0.105
    print(f"[lean] pitched-base consistency: {float(out[0]):.4f} (expect {expect:.4f})")
    assert abs(float(out[0]) - expect) < 1e-3
    print("[lean] ALL PASS")


if __name__ == "__main__":
    test_lean()
    print("\nALL v70b GUARD UNIT TESTS PASS")
