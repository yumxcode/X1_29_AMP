#!/usr/bin/env python3
"""v69 local unit test: the three guard rewards, isaaclab-stubbed.

Stubs isaaclab modules, implements REAL quat math in the math_utils stub
(torch), drives the actual reward functions in
roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py with
a synthetic env, and checks the forensics-calibrated numbers:

  1. foot_yaw_neutral: 20/11 deg toe-out -> excess (14+5) deg; demo 2/2 deg
     -> 0; turn cmd gate; stand gate.
  2. hip_roll_abduction_guard: +13 deg outward DC -> relu(13-4)*2 deg;
     inward (-) also caught; AC not penalized (EMA).
  3. lateral_sway_guard: 1 Hz pump passes the high-pass; DC not penalized.
"""
import functools
import sys
import types
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]

# ---- stub isaaclab ------------------------------------------------------
def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def quat_apply(q, v):
    """q (...,4) wxyz, v (...,3) -> rotated v. Real math."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # t = 2 * cross(q_vec, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx)], dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1), v)


class _SceneEntityCfg:
    def __init__(self, name="robot", joint_names=None, body_names=None,
                 preserve_order=False):
        self.name = name
        self.joint_names = joint_names
        self.body_names = body_names
        self.joint_ids = None
        self.body_ids = None


def _resolve_ids(terms, joint_names_all, body_names_all):
    for cfg in terms:
        if cfg.joint_names:
            cfg.joint_ids = [joint_names_all.index(n) for n in cfg.joint_names]
        if cfg.body_names:
            cfg.body_ids = [body_names_all.index(n) for n in cfg.body_names]


_iso = types.ModuleType("isaaclab")
_mu = _stub_module("isaaclab.utils.math",
                   quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse)
_iso = types.ModuleType("isaaclab")
_iso.utils = types.ModuleType("isaaclab.utils")
_iso.utils.math = _mu
sys.modules["isaaclab.utils"] = _iso.utils
sys.modules["isaaclab.utils.math"] = _mu
_envs = _stub_module("isaaclab.envs", mdp=types.SimpleNamespace())
_man = _stub_module("isaaclab.managers", SceneEntityCfg=_SceneEntityCfg)
_sens = _stub_module("isaaclab.sensors", ContactSensor=object, RayCaster=object)
_as = _stub_module("isaaclab.assets", Articulation=object, RigidObject=object)
_mu = _stub_module("isaaclab.utils.math",
                   quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse)
sys.modules["isaaclab"] = _iso
sys.modules["isaaclab.envs"] = _envs
sys.modules["isaaclab.managers"] = _man
sys.modules["isaaclab.sensors"] = _sens
sys.modules["isaaclab.assets"] = _as
sys.modules["isaaclab.utils"] = _iso.utils
sys.modules["isaaclab.utils.math"] = _mu

# load rewards.py directly by path (the package __init__ chain pulls
# isaaclab_tasks, unavailable locally)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "v69_rewards",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

DEV = torch.device("cpu")
JOINTS = ["lumbar_roll_joint", "lumbar_yaw_joint",
          "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_pitch_joint"]
BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class FakeAsset:
    def __init__(self):
        self.data = types.SimpleNamespace()


class FakeCmd:
    def __init__(self, cmd):
        self.cmd = torch.tensor([cmd], dtype=torch.float32)   # (1,3)

    def get_command(self, name):
        return self.cmd


class FakeEnv:
    def __init__(self):
        self.device = DEV
        self.scene = {"robot": FakeAsset()}
        self.command_manager = FakeCmd([1.0, 0.0, 0.0])
        self._counter = 0

    @property
    def the_id(self):
        return id(self)


def yaw_quat(deg):
    r = np.radians(deg)
    return torch.tensor([np.cos(r / 2), 0, 0, np.sin(r / 2)], dtype=torch.float32)


def qmul(a, b):
    """Hamilton product, wxyz."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def foot_quat(yaw_deg):
    """Ankle body orientation = mount R_y(90deg) (local z -> world x, the
    X1 foot fore-aft axis) then world-z yaw by yaw_deg."""
    qy90 = torch.tensor([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                        dtype=torch.float32)
    qz = yaw_quat(yaw_deg)
    return qmul(qz, qy90)


def test_foot_yaw():
    env = FakeEnv()
    cfg = _SceneEntityCfg("robot", body_names=BODIES, preserve_order=True)
    _resolve_ids([cfg], JOINTS, BODIES)
    asset = env.scene["robot"]
    asset.data.body_quat_w = torch.stack([foot_quat(20.0), foot_quat(-11.0)]).unsqueeze(0)
    asset.data.root_quat_w = torch.tensor([[1.0, 0, 0, 0]])
    out = R.foot_yaw_neutral(env, cfg)
    expect = np.radians(14) + np.radians(5)
    print(f"[foot_yaw] m11498-level: {out[0]:.4f} rad (expect {expect:.4f})")
    assert abs(out[0].item() - expect) < 1e-3
    # demo-level feet (2 deg) -> 0
    asset.data.body_quat_w = torch.stack([foot_quat(2.0), foot_quat(2.0)]).unsqueeze(0)
    out = R.foot_yaw_neutral(env, cfg)
    print(f"[foot_yaw] demo-level: {out[0]:.4f} (expect 0)")
    assert float(out[0]) == 0.0
    # turn gate
    asset.data.body_quat_w = torch.stack([foot_quat(20.0), foot_quat(20.0)]).unsqueeze(0)
    env.command_manager = FakeCmd([1.0, 0.0, 0.8])
    out = R.foot_yaw_neutral(env, cfg)
    assert float(out[0]) == 0.0
    # stand gate
    env.command_manager = FakeCmd([0.0, 0.0, 0.0])
    out = R.foot_yaw_neutral(env, cfg)
    assert float(out[0]) == 0.0
    # base-yaw invariance: base yawed +30 deg, feet +20 deg rel base
    env.command_manager = FakeCmd([1.0, 0.0, 0.0])
    asset.data.root_quat_w = yaw_quat(30.0).unsqueeze(0)
    asset.data.body_quat_w = torch.stack([foot_quat(50.0), foot_quat(19.0)]).unsqueeze(0)
    out = R.foot_yaw_neutral(env, cfg)
    assert abs(out[0].item() - expect) < 1e-3
    print("[foot_yaw] base-yaw invariance + gates PASS")


def test_abduction():
    env = FakeEnv()
    cfg = _SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"],
                          preserve_order=True)
    _resolve_ids([cfg], JOINTS, BODIES)
    asset = env.scene["robot"]
    R._EMA_STATE.clear()
    # policy: left -13 deg (outward), right +13 deg (outward)
    q = torch.tensor([[-np.radians(13), np.radians(13)]] * 512, dtype=torch.float32)
    asset.data.joint_pos = torch.zeros(512, len(JOINTS))
    asset.data.joint_pos[:, cfg.joint_ids] = q
    out = torch.zeros(512)
    for _ in range(512):   # let EMA converge
        out = R.hip_roll_abduction_guard(env, cfg)
    expect = 2 * (np.radians(13) - 0.07)
    print(f"[abd] 13-deg outward DC: {out[0]:.4f} rad (expect {expect:.4f})")
    assert abs(out[0].item() - expect) < 1e-3
    # demo level ~0.5 deg -> 0 (EMA tau 8 s: run long enough to converge)
    asset.data.joint_pos[:, cfg.joint_ids] = torch.tensor(
        [[np.radians(0.3), np.radians(-0.3)]] * 512, dtype=torch.float32)
    for _ in range(10000):
        out = R.hip_roll_abduction_guard(env, cfg)
    print(f"[abd] demo level: {out[0]:.4f} (expect 0)")
    assert float(out[0]) < 1e-3
    # AC swings NOT penalized: +/-8 deg at 2 Hz, zero DC (TIME SERIES:
    # one reward call per control step, batch=1)
    R._EMA_STATE.clear()
    N2 = 3000
    t = np.arange(N2)
    ac = 8 * np.sin(2 * np.pi * 2 * t / 50.0)  # 2 Hz, 50 Hz ctrl
    q2 = np.radians(np.stack([-ac, ac], 1))    # outward oscillation
    asset.data.joint_pos = torch.zeros(1, len(JOINTS), dtype=torch.float32)
    out = None
    for k in range(N2):
        asset.data.joint_pos[:, cfg.joint_ids] = torch.tensor(q2[k], dtype=torch.float32)
        out = R.hip_roll_abduction_guard(env, cfg)
    print(f"[abd] pure-AC 8deg@2Hz: {out[0]:.4f} (expect ~0, EMA attenuated)")
    assert float(out[0]) < 0.02
    print("[abd] PASS")


def test_sway():
    env = FakeEnv()
    lum = _SceneEntityCfg("robot", joint_names=["lumbar_roll_joint", "lumbar_yaw_joint"])
    hr = _SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"])
    _resolve_ids([lum, hr], JOINTS, BODIES)
    asset = env.scene["robot"]
    R._EMA_STATE.clear()
    N = 3000
    t = np.arange(N)
    lr = np.radians(13.5) * np.sin(2 * np.pi * 2 * t / 50.0)   # ptp 27 deg @2 Hz
    ly = np.radians(26.0) * np.sin(2 * np.pi * 2 * t / 50.0 + 0.3)
    hrl = np.radians(7.5) * np.sin(2 * np.pi * 2 * t / 50.0)
    hrr = np.radians(7.5) * np.sin(2 * np.pi * 2 * t / 50.0 + np.pi)
    jp = np.zeros((N, len(JOINTS)))
    jp[:, lum.joint_ids[0]] = lr + 0.05     # + DC 3 deg (must NOT pay)
    jp[:, lum.joint_ids[1]] = ly - 0.10     # - DC 6 deg (must NOT pay)
    jp[:, hr.joint_ids[0]] = hrl
    jp[:, hr.joint_ids[1]] = hrr
    asset.data.joint_pos = torch.zeros(1, len(JOINTS), dtype=torch.float32)
    outs = []
    for k in range(N):
        asset.data.joint_pos[:] = torch.tensor(jp[k], dtype=torch.float32)
        outs.append(float(R.lateral_sway_guard(env, lum, hr)[0]))
    outs = np.array(outs)
    tail = outs[-500:]
    exp_mean = 0.637 * (np.radians(13.5) + np.radians(26.0) + np.radians(7.5) * 2)
    print(f"[sway] m11498-level pump: mean {tail.mean():.3f} rad "
          f"(analytic ~{exp_mean:.3f}; DC ignored)")
    assert 0.7 * exp_mean < tail.mean() < 1.15 * exp_mean
    # still mode (all zero) -> ~0
    for _ in range(500):
        asset.data.joint_pos[:] = 0.0
        out = R.lateral_sway_guard(env, lum, hr)
    print(f"[sway] still: {float(out[0]):.4f} (expect ~0)")
    assert float(out[0]) < 0.02
    print("[sway] PASS")


if __name__ == "__main__":
    test_foot_yaw()
    test_abduction()
    test_sway()
    print("\nALL v69 GUARD UNIT TESTS PASS")
