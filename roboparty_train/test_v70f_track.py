#!/usr/bin/env python3
"""v70f local unit test: demo_joint_tracking_exp, isaaclab-stubbed.

Same stub machinery as test_v70_guards.py; drives the REAL reward in
.../amp/mdp/rewards.py with a synthetic env (stashed tracking refs +
fake animation term + fake asset). Checks:
  1. No stash -> zeros (recipes without the event unaffected).
  2. In-sync (q == demo at t_k) -> 1.0 gated on; stand/turn gates off.
  3. Time base: t_k = t0 + episode_length_buf*step_dt, loop-wrapped.
  4. Out-of-sync |q-demo|=0.7 rad total -> exp(-0.7/0.35)=0.135.
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
    "v70f_rewards",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

N = 3
T0 = torch.tensor([0.5, 0.0, 1.0])
DUR = torch.tensor([2.0, 1.0, 3.0])


class FakeDataTerm:
    def get_motion_state(self, mids, mtimes):
        # demo joint value = 0.1 * motion_id + 0.2 * t (per joint)
        t = mtimes.unsqueeze(-1)
        dof = 0.1 * mids.unsqueeze(-1).float() + 0.2 * t
        return {"dof_pos": dof.expand(-1, 29).clone()}


class FakeAnimTerm:
    def __init__(self):
        self.motion_ids = torch.tensor([3, 7, 11])
        self.motion_data_term = FakeDataTerm()


class FakeAnim:
    def __init__(self):
        self._terms = {"anim": FakeAnimTerm()}


class FakeAsset:
    def __init__(self):
        self.device = torch.device("cpu")
        self.data = types.SimpleNamespace(joint_pos=torch.zeros(1, 29))


class FakeCmd:
    def __init__(self, cmd):
        self.cmd = cmd

    def get_command(self, name):
        return self.cmd


class FakeEnv:
    def __init__(self, cmds, joint_pos):
        self.device = torch.device("cpu")
        self.num_envs = N
        self.step_dt = 0.02
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.scene = {"robot": FakeAsset()}
        self.scene["robot"].data.joint_pos = joint_pos
        self.animation_manager = FakeAnim()
        self.command_manager = FakeCmd(torch.tensor(cmds, dtype=torch.float32))
        self._demo_track_mid = torch.tensor([3, 7, 11])
        self._demo_track_t0 = T0
        self._demo_track_dur = DUR
        self._demo_track_ok = torch.tensor([True, True, False])


def run(env, cfg):
    return R.demo_joint_tracking_exp(env, cfg)


def main():
    cfg = _SceneEntityCfg("robot", joint_names=["hipL", "hipR", "kneeL", "kneeR"])
    cfg.joint_ids = [10, 11, 12, 13]

    # 1. no stash -> zeros
    env0 = FakeEnv([[1, 0, 0]] * N, torch.zeros(1, 29))
    env0._demo_track_ok = torch.zeros(N, dtype=torch.bool)
    out = run(env0, cfg)
    assert float(out.abs().sum()) == 0.0
    print("[track] no-stash short-circuit PASS")

    # 2. in-sync: q at tracked joints == demo(t_k) value
    env = FakeEnv([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], torch.zeros(1, 29))
    env.episode_length_buf[:] = 25  # t_k = t0 + 0.5
    # env0: mid 3, t=1.0 -> demo val 0.3+0.2=0.5
    # env1: mid 7, t=(0+0.5)%1.0=0.5 -> 0.7+0.1=0.8
    q = torch.zeros(1, 29)
    q[0, 10:14] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    q[0, 14:18] = torch.tensor([0.8, 0.8, 0.8, 0.8])  # not tracked for env2
    env.scene["robot"].data.joint_pos = q.expand(N, 29).clone()
    env.scene["robot"].data.joint_pos[1, 10:14] = 0.8  # env1 in sync
    out = run(env, cfg)
    assert abs(float(out[0]) - 1.0) < 1e-5, f"env0 in-sync {out[0]}"
    assert abs(float(out[1]) - 1.0) < 1e-5, f"env1 in-sync(loop wrap) {out[1]}"
    assert float(out[2]) == 0.0, "env2 untracked -> 0"
    print("[track] in-sync=1.0 (incl. loop wrap) + untracked=0 PASS")

    # 3. gates: stand (speed 0) and turn (yaw 0.8) pay 0 even in sync
    env.command_manager = FakeCmd(torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0.8], [1.0, 0, 0]], dtype=torch.float32))
    out = run(env, cfg)
    assert float(out[0]) == 0.0 and float(out[1]) == 0.0
    print("[track] stand/turn gates PASS")

    # 4. out-of-sync magnitude: |q-demo| = 0.7 total -> exp(-2) = 0.1353
    q2 = torch.zeros(N, 29)
    q2[:, 10:14] = 0.5          # all four tracked joints in sync (env0 demo)
    q2[:, 10] = 0.5 + 0.7       # 0.7 rad off on one joint only
    env.scene["robot"].data.joint_pos = q2
    env.command_manager = FakeCmd(torch.tensor(
        [[1.0, 0, 0]] * N, dtype=torch.float32))
    out = run(env, cfg)
    expect = math.exp(-0.7 / 0.35)
    assert abs(float(out[0]) - expect) < 1e-4, f"{out[0]} vs {expect}"
    print(f"[track] error kernel {float(out[0]):.4f} == exp(-2) PASS")


if __name__ == "__main__":
    main()
    print("\nALL v70f TRACKING UNIT TESTS PASS")
