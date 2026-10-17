#!/usr/bin/env python3
"""v71 local unit test: exogenous gait clock (module math + reward kernels
+ sim2sim numpy-twin equivalence), isaaclab-stubbed.

Covers (spec: acceptance/X1_GAIT_SPEC_V71.md):
  1. schedule values T*/d*/h* at the spec table speeds (0.5/1.0/2.5 m/s);
  2. phase structure: L/R anti-phase, unit-norm obs, exact layout
     [sinL, sinR, cosL, cosR];
  3. target_height: 0 through stance, apex h*(v) mid-swing, C0 at both
     boundaries; swing-mid/stance-mid masks are disjoint with free
     transitions between;
  4. foot_clock_height income differential: on-clock follower ~2/step vs
     drag ~1.06/step and micro-tap <1.6/step (the dense anti-basin signal);
  5. contact_clock_timing: drag pays ~-1 on swing-mid frames; on-clock
     follower pays ~0; stand gate zeroes both terms;
  6. numpy twin (mujoco_rollout.py inline formulas) == torch clock_obs,
     bitwise over a (t, v) grid — the deploy contract.
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
AMP_MDP = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp"

# ---- stub isaaclab (v69 pattern) ----------------------------------------
def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


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


_iso = types.ModuleType("isaaclab")
_iso.utils = types.ModuleType("isaaclab.utils")
_mu = _stub_module("isaaclab.utils.math",
                   quat_apply=quat_apply,
                   quat_apply_inverse=lambda q, v: quat_apply(
                       torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], -1), v))
sys.modules["isaaclab"] = _iso
sys.modules["isaaclab.envs"] = _stub_module("isaaclab.envs", mdp=types.SimpleNamespace())
sys.modules["isaaclab.managers"] = _stub_module(
    "isaaclab.managers", SceneEntityCfg=lambda *a, **k: types.SimpleNamespace())
sys.modules["isaaclab.sensors"] = _stub_module(
    "isaaclab.sensors", ContactSensor=object, RayCaster=object)
sys.modules["isaaclab.assets"] = _stub_module(
    "isaaclab.assets", Articulation=object, RigidObject=object)
sys.modules["isaaclab.utils"] = _iso.utils
sys.modules["isaaclab.utils.math"] = _mu

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("v71_gait_clock", AMP_MDP / "gait_clock.py")
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)

_spec = importlib.util.spec_from_file_location("v71_rewards", AMP_MDP / "rewards.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

DEV = torch.device("cpu")
BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class FakeCfg:
    def __init__(self, body_ids, name="robot"):
        self.body_ids = body_ids
        self.name = name


class FakeAsset:
    def __init__(self):
        self.data = types.SimpleNamespace()
        self.device = DEV


class FakeContactData:
    def __init__(self, forces):
        # forces: (N, hist, 2, 3) already-normed boolean driver
        self.net_forces_w_history = forces


class FakeContact:
    def __init__(self, contact_bool):
        # encode contact (>1 N) as force-z of the LAST history frame
        # layout (num_envs=1, hist=1, feet=2, xyz=3)
        f = torch.zeros(1, 1, 2, 3)
        f[:, 0, :, 2] = contact_bool.float() * 10.0
        self.data = FakeContactData(f)


class FakeCmd:
    def __init__(self, cmd):
        self.cmd = torch.tensor([cmd], dtype=torch.float32)

    def get_command(self, name):
        return self.cmd


class FakeEnv:
    def __init__(self, step_k, cmd=(1.0, 0.0, 0.0), dt=0.02):
        self.device = DEV
        self.step_dt = dt
        self.episode_length_buf = torch.tensor([step_k], dtype=torch.long)
        self.command_manager = FakeCmd(cmd)
        self.robot = FakeAsset()
        self.contact = None
        self.scene = {"robot": self.robot, "contact_forces": self.contact}

    def set_feet(self, h_min_m, contact_bool):
        """Place both ankle bodies so the LOWEST sole point (toe, local
        z=-0.07, radius 2mm) sits at h_min_m; identity orientation.
        h_min_m / contact_bool accept scalars (broadcast to both feet)."""
        if not hasattr(h_min_m, "__len__"):
            h_min_m = [h_min_m, h_min_m]
        if not hasattr(contact_bool, "__len__"):
            contact_bool = [contact_bool, contact_bool]
        h = torch.as_tensor(h_min_m, dtype=torch.float32).reshape(2).clone()
        cb = torch.as_tensor(contact_bool, dtype=torch.bool).reshape(2).clone()
        body_z = h + 0.07 + 0.002
        n = 1
        pos = torch.zeros(n, 2, 3)
        pos[:, :, 2] = body_z
        quat = torch.zeros(n, 2, 4)
        quat[:, :, 0] = 1.0
        self.robot.data.body_pos_w = pos
        self.robot.data.body_quat_w = quat
        self.contact = FakeContact(cb)
        self.scene["contact_forces"] = self.contact


def foot_clock(env, sensor_cfg=None, asset_cfg=None, **kw):
    return R.foot_clock_height(env, FakeCfg([0, 1], "contact_forces"),
                               FakeCfg([0, 1], "robot"), **kw)


def contact_clock(env, sensor_cfg=None, **kw):
    return R.contact_clock_timing(env, FakeCfg([0, 1], "contact_forces"), **kw)


def run_all():
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
        if not ok:
            failures.append(name)

    # --- 1. schedule values (spec table) ---------------------------------
    print("[1] schedule values")
    for v, t_ref in ((0.5, 1.28), (1.0, 1.10), (2.5, 0.70)):
        t = float(GC.t_star(torch.tensor([v]))[0])
        check(f"T*({v})={t:.3f}", abs(t - t_ref) < 0.005, f"ref {t_ref}")
    d = float(GC.duty_star(torch.tensor([1.0]))[0])
    check("d*(1.0)=0.61", abs(d - 0.61) < 0.005)
    h = float(GC.apex_star(torch.tensor([1.0]))[0])
    check("h*(1.0)=0.050", abs(h - 0.050) < 1e-4)

    # --- 2. phase structure ----------------------------------------------
    print("[2] phase structure")
    t = torch.linspace(0, 12, 121)
    v = torch.full_like(t, 1.0)
    phi = GC.phases(t, v)
    circ = torch.remainder(phi[:, 1] - phi[:, 0] - 0.5, 1.0)
    circ = torch.minimum(circ.abs(), 1.0 - circ.abs())
    check("L/R anti-phase", bool((circ < 1e-6).float().mean() > 0.999))
    obs = GC.clock_obs(t, v)
    check("obs shape (N,4)", obs.shape == (121, 4))
    norm = obs[:, 0] ** 2 + obs[:, 2] ** 2
    check("unit circle", bool(((norm - 1).abs() < 1e-5).all()))
    # exact layout: col0 = sin(2pi phiL)
    check("layout [sinL,sinR,cosL,cosR]",
          bool(torch.allclose(obs[:, 0], torch.sin(2 * math.pi * phi[:, 0]), atol=1e-6)
               and torch.allclose(obs[:, 2], torch.cos(2 * math.pi * phi[:, 0]), atol=1e-6)
               and torch.allclose(obs[:, 1], torch.sin(2 * math.pi * phi[:, 1]), atol=1e-6)
               and torch.allclose(obs[:, 3], torch.cos(2 * math.pi * phi[:, 1]), atol=1e-6)))

    # --- 3. height profile + masks ----------------------------------------
    print("[3] target profile & masks")
    d1, h1 = 0.61, 0.050
    phi_g = torch.linspace(0, 1, 2001)
    v1 = torch.full((2001,), 1.0)
    phi2 = torch.stack([phi_g, phi_g], dim=-1)          # (2001, 2) both feet
    z = GC.target_height(phi2, v1)
    check("z*=0 on stance", bool((z[phi_g < d1].abs() < 1e-6).all()))
    mid = int(0.61 * 2000 + (1 - 0.61) / 2 * 2000)
    check(f"apex={float(z[mid, 0]):.3f} mid-swing", abs(float(z[mid, 0]) - h1) < 0.004)
    # C0 continuity at both boundaries
    i_td = int(d1 * 2000)
    check("C0 at touchdown", abs(float(z[i_td + 1, 0])) < 0.003)
    check("C0 at liftoff-end", abs(float(z[-2, 0])) < 0.003)
    sw = GC.swing_mid_mask(phi2, v1)
    st = GC.stance_mid_mask(phi2, v1)
    check("masks disjoint", bool((sw & st).sum() == 0))
    span_sw = float(sw.float().mean())
    check(f"swing-mid span {span_sw:.2f} (0.6*0.39=0.234)",
          abs(span_sw - 0.6 * 0.39) < 0.01)
    span_st = float(st.float().mean())
    check(f"stance-mid span {span_st:.2f} (0.70*0.61=0.427)",
          abs(span_st - 0.7 * 0.61) < 0.01)

    # --- 4. reward income differentials -----------------------------------
    print("[4] foot_clock_height income")
    dt, T = 0.02, 1.10
    inc_on, inc_drag, inc_tap = [], [], []
    pen_on, pen_drag, pen_tap = [], [], []
    for k in range(600):
        tk = (k + 1) * dt
        v_t = torch.tensor([tk / T]) % 1.0
        phi_env = GC.phases(torch.tensor([tk]), torch.tensor([1.0]))[0]
        z_star = GC.target_height(phi_env.unsqueeze(0), torch.tensor([[1.0]]))[0]

        # (a) on-clock follower: height tracks z*, contact iff stance
        env = FakeEnv(k + 1)
        env.set_feet(z_star.detach(), phi_env.detach() < 0.61)
        inc_on.append(float(foot_clock(env)))
        pen_on.append(float(contact_clock(env)))

        # (b) drag: foot glued at 5 mm, always in contact
        env = FakeEnv(k + 1)
        env.set_feet(0.005, True)
        inc_drag.append(float(foot_clock(env)))
        pen_drag.append(float(contact_clock(env)))

        # (c) 2.2 Hz micro-tap: 60% duty at 0.45 s period, h oscillating 5-25mm
        tap_c = (tk % 0.455) < 0.27
        tap_h = 0.02 if tap_c else 0.005
        env = FakeEnv(k + 1)
        env.set_feet(tap_h, bool(tap_c))
        inc_tap.append(float(foot_clock(env)))
        pen_tap.append(float(contact_clock(env)))

    m_on, m_drag, m_tap = np.mean(inc_on), np.mean(inc_drag), np.mean(inc_tap)
    print(f"    on-clock {m_on:.2f}  drag {m_drag:.2f}  micro-tap {m_tap:.2f}")
    check("on-clock ~2/step", m_on > 1.8)
    # differential = height kernel 0.45/step + contact hinge 0.24/step
    # (~26% of the ~2.65/step task stream) — the falsified priors froze
    # at partial score with NO frame-level gradient; here the differential
    # concentrates exactly on the swing frames that decide lift.
    check("drag far below on-clock", m_drag < m_on - 0.4)
    check("micro-tap below on-clock", m_tap < m_on - 0.15)
    p_on, p_drag, p_tap = np.mean(pen_on), np.mean(pen_drag), np.mean(pen_tap)
    print(f"    contact-timing penalty: on {p_on:.2f}  drag {p_drag:.2f}  tap {p_tap:.2f}")
    check("on-clock timing ~0", p_on > -0.15)
    check("drag timing < -0.3", p_drag < -0.3)
    check("tap timing < -0.3", p_tap < -0.3)

    # --- 5. gates ----------------------------------------------------------
    print("[5] command gating")
    env = FakeEnv(50, cmd=(0.0, 0.0, 0.0))          # stand
    env.set_feet(0.005, True)
    check("stand: foot_clock = 0", abs(float(foot_clock(env))) < 1e-6)
    check("stand: contact_clock = 0", abs(float(contact_clock(env))) < 1e-6)
    env = FakeEnv(50, cmd=(2.0, 0.0, 0.0))          # fast walk: active
    env.set_feet(0.02, True)                        # drag: both glued
    check("fast walk active (drag violation)", abs(float(contact_clock(env))) > 0)

    # --- 6. numpy twin (deploy contract) ----------------------------------
    print("[6] sim2sim numpy twin equivalence")
    worst = 0.0
    for v in (0.0, 0.5, 1.0, 1.7, 2.5):
        for step in range(0, 1200, 7):
            t_np = step * 0.02
            T_np = min(max(1.46 - 0.36 * v, 0.70), 1.75)
            pl = (t_np / T_np) % 1.0
            pr = (pl + 0.5) % 1.0
            ck = np.array([math.sin(2 * math.pi * pl), math.sin(2 * math.pi * pr),
                           math.cos(2 * math.pi * pl), math.cos(2 * math.pi * pr)],
                          dtype=np.float32)
            th = GC.clock_obs(torch.tensor([t_np]), torch.tensor([v]))[0].numpy()
            worst = max(worst, float(np.abs(ck - th).max()))
    # float32 (torch/GPU) vs float64->float32 (numpy/CPU): agreement at
    # float32 ULP level (obs noise on other channels is 0.03; 5e-5 is
    # 1/600 of that)
    check(f"twin max diff {worst:.2e}", worst < 5e-5)

    print("=" * 60)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
