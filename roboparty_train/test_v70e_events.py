#!/usr/bin/env python3
"""v70e local unit test: reset_robot_from_demo event, isaaclab-stubbed.

Same stub machinery as test_v70_guards.py. Drives the REAL event function
in .../amp/mdp/events.py with a synthetic env (animation manager + asset)
and checks:
  1. Command gating: only envs with |cmd_xy| > 0.3 are eligible.
  2. Probability gating: forced rand < p selects exactly the eligible set.
  3. The written root state = demo frame's z/quat/velocities at the env
     origin xy; joints = demo dof pos/vel — phase-locked to the animation
     manager's (motion_id, fetch_time).
  4. Zero-selection short-circuit (no asset writes when nothing eligible).
"""
import functools
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
    "v70e_events",
    REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/events.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


class FakeAsset:
    def __init__(self, n):
        self.device = torch.device("cpu")
        self.writes = []
        self.data = types.SimpleNamespace(
            default_root_state=torch.zeros(n, 13))

    def write_root_state_to_sim(self, root, env_ids=None):
        self.writes.append(("root", root.clone(), env_ids))

    def write_joint_state_to_sim(self, pos, vel, env_ids=None):
        self.writes.append(("joint", pos.clone(), vel.clone(), env_ids))


class FakeAnimTerm:
    def __init__(self):
        self.motion_ids = torch.tensor([3, 7, 11])
        self.motion_fetch_time = torch.tensor([[0.5], [1.2], [2.0]])
        self.motion_data_term = self

    def get_motion_durations(self, mids):
        return torch.tensor([5.0 + float(m) for m in mids])

    def get_motion_state(self, mids, mtimes):
        return {
            "root_pos_w": torch.stack([
                torch.tensor([0.1 * float(m) + 5.0, 2.0, 0.71 + 0.01 * float(m)])
                for m in mids]),
            "root_quat": torch.tensor(
                [[1.0, 0, 0, 0]] * len(mids)),
            "root_vel_w": torch.tensor(
                [[1.2, 0.0, 0.0]] * len(mids)),
            "root_ang_vel_w": torch.tensor(
                [[0.0, 0.0, 0.05]] * len(mids)),
            "dof_pos": torch.tensor(
                [[0.1 * float(m) + 0.5] * 29 for m in mids]),
            "dof_vel": torch.tensor(
                [[-0.2 * float(m)] * 29 for m in mids]),
        }


class FakeAnim:
    """Mirrors the real structure: motion ids live on the manager's TERM."""

    def __init__(self):
        self._terms = {"anim": FakeAnimTerm()}


class FakeCmd:
    def __init__(self, cmd):
        self.cmd = cmd

    def get_command(self, name):
        return self.cmd


class FakeScene:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)
        for i in range(n):
            self.env_origins[i, 0] = 10.0 * i
        self._assets = {"robot": FakeAsset(n)}

    def __getitem__(self, key):
        return self._assets[key]


class FakeEnv:
    def __init__(self, cmds):
        n = len(cmds)
        self.device = torch.device("cpu")
        self.num_envs = n
        self.scene = FakeScene(n)
        self.animation_manager = FakeAnim()
        self.command_manager = FakeCmd(torch.tensor(cmds, dtype=torch.float32))


def test_event():
    # cmds: env0 walking, env1 standing, env2 walking
    env = FakeEnv([[1.0, 0, 0], [0.0, 0, 0], [0.5, 0, 0.3]])
    asset = env.scene["robot"]
    env_ids = torch.tensor([0, 1, 2])
    torch.manual_seed(0)
    # force rand < 0.5 True for all (p_demo=1.0 keeps all eligible anyway)
    E.reset_robot_from_demo(env, env_ids, p_demo=1.0, min_cmd_speed=0.3)
    assert len(asset.writes) == 2, "stand env (cmd 0.0) must be skipped"
    kind, root, sel = asset.writes[0]
    assert torch.equal(sel, torch.tensor([0, 2])), "selection = walking envs"
    # root: xy at env origins, z from demo, quat/vel from demo
    assert root[0, 0] == 0.0 and root[1, 0] == 20.0  # env origins x (sel=[0,2])
    assert abs(root[0, 2] - 0.74) < 1e-6            # demo z (motion 3: 0.71+0.03)
    assert abs(root[0, 7] - 1.2) < 1e-6              # demo lin vel x
    assert abs(root[0, 12] - 0.05) < 1e-6            # demo ang vel z
    kind2, jpos, jvel, sel2 = asset.writes[1]
    assert torch.equal(sel2, sel)
    assert abs(jpos[0, 0] - 0.8) < 1e-6              # demo dof (motion 3)
    assert abs(jvel[0, 0] + 0.6) < 1e-6
    # phase lock: motion ids/times come from the animation manager
    print("[demo-reset] gating + phase-locked state writes PASS")

    # zero-selection short-circuit: all standing
    env2 = FakeEnv([[0.0, 0, 0]] * 3)
    asset2 = env2.scene["robot"]
    E.reset_robot_from_demo(env2, torch.tensor([0, 1, 2]), p_demo=1.0)
    assert len(asset2.writes) == 0
    # p_demo = 0 also writes nothing
    E.reset_robot_from_demo(env, env_ids, p_demo=0.0)
    assert len(asset.writes) == 2
    print("[demo-reset] zero-selection short-circuit PASS")


if __name__ == "__main__":
    test_event()
    print("\nALL v70e DEMO-RESET UNIT TESTS PASS")
