#!/usr/bin/env python3
"""v95 local unit test: arm_distal_amp_guard (amplitude math + DC freedom +
demo zero-cost on real x1_lab_v72 streams + x7398 defect cost + EMA key
isolation + channel/band wiring), isaaclab-stubbed (v71/v94 pattern).

Covers:
  1. exact amplitude math: 1 Hz sinusoid of amplitude A -> steady-state
     RMS A/sqrt(2); excess = relu(RMS - band) per channel, summed;
  2. DC FREEDOM: a constant 56 deg elbow offset pays 0 (only the SWING is
     taxed — the complement of arm_dc_band_guard, which anchors DC only);
  3. demo streams (x1_lab_v72 PRIME + 35-family sample, looped to EMA
     steady state) -> ~0 total (max demo RMS under every band);
  4. x7398 walk10/walk05 rollouts (the defect) -> ~0.4-0.8 rad (the
     designed dose response at w=-0.5);
  5. shoP exclusion: shoulder_pitch not among the 12 channels;
  6. EMA key isolation: running arm_dc_band_guard on the same env before/
     after leaves arm_distal_amp_guard results unchanged (arqdc/arqms
     vs armdc namespaces).
"""
import functools
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
AMP_MDP = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp"
MDIR = REPO / "roboparty_train/robolab/data/motions/x1_lab_v72"


# ---- stub isaaclab (v71/v94 pattern) -------------------------------------
def quat_apply(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * vz),
        vz + w * tz + (x * ty - y * tx)], dim=-1)


_iso = types.ModuleType("isaaclab")
_iso.utils = types.ModuleType("isaaclab.utils")
_mu = types.ModuleType("isaaclab.utils.math")
_mu.quat_apply = quat_apply
_iso.utils.math = _mu
sys.modules["isaaclab"] = _iso
sys.modules["isaaclab.utils"] = _iso.utils
sys.modules["isaaclab.utils.math"] = _mu
sys.modules["isaaclab.envs"] = _stub = types.ModuleType("isaaclab.envs")
_stub.mdp = types.SimpleNamespace()
sys.modules["isaaclab.managers"] = types.ModuleType("isaaclab.managers")


class _SceneEntityCfg:
    def __init__(self, *a, **k):
        self.a, self.k = a, k
        self.name = a[0] if a else k.get("name", "robot")


sys.modules["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfg
_sensors = types.ModuleType("isaaclab.sensors")
_sensors.ContactSensor = object
_sensors.RayCaster = object
sys.modules["isaaclab.sensors"] = _sensors
_assets = types.ModuleType("isaaclab.assets")
_assets.Articulation = object
_assets.RigidObject = object
sys.modules["isaaclab.assets"] = _assets

import importlib.util as ilu
_spec = ilu.spec_from_file_location("amp_rewards_r", AMP_MDP / "rewards.py")
_r = ilu.module_from_spec(_spec)
sys.modules["amp_rewards_r"] = _r
_spec.loader.exec_module(_r)

# the wired v95 channel spec (must mirror x1_amp_env_cfg.py arm_distal_amp)
CHANNELS = ["left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_yaw_joint",
            "left_wrist_pitch_joint", "left_wrist_roll_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint", "right_elbow_yaw_joint",
            "right_wrist_pitch_joint", "right_wrist_roll_joint"]
BANDS = [0.14, 0.05, 0.05, 0.05, 0.16, 0.13] * 2
LAB_DOF_NAMES = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "right_ankle_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]
# rollout npz hinge order (from mujoco_rollout meta hinge_names)
HINGE_NAMES = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
    "left_elbow_yaw_joint", "left_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_pitch_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint"]


class _AssetData:
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene(dict):
    pass


class _Env:
    _next_id = [0]

    def __init__(self, q_stream):
        _Env._next_id[0] += 1
        self._id = _Env._next_id[0]
        self.scene = _Scene(robot=_Asset(_AssetData(None)))
        self._q_stream = q_stream
        self._t = -1

    def step_q(self):
        self._t += 1
        self.scene["robot"].data.joint_pos = torch.tensor(
            self._q_stream[self._t % len(self._q_stream)],
            dtype=torch.float32).unsqueeze(0)
        cfg = _SceneEntityCfg("robot")
        cfg.joint_ids = list(range(self._q_stream.shape[1]))
        return cfg


def run_stream(q_stream, loops=1):
    """Replay q_stream (T,12) rad through the guard; return per-channel
    steady-state excess list (last frame) and total."""
    env = _Env(q_stream)
    r = None
    per = None
    for _ in range(loops * len(q_stream)):
        cfg = env.step_q()
        r = _r.arm_distal_amp_guard(env, asset_cfg=cfg, bands=BANDS)
    per = []
    with torch.no_grad():
        for k, b in enumerate(BANDS):
            q = torch.tensor(q_stream[:, k], dtype=torch.float32)
            hp = q - torch.tensor(_ema_series(q_stream[:, k], 0.0025))
            ms = _ema_series((hp ** 2).numpy(), 0.02)
            per.append(max(0.0, float(ms[-1] ** 0.5) - b))
    return r.item(), per


def _ema_series(x, alpha):
    out, y = [], x[0]
    for v in x:
        y = y * (1 - alpha) + v * alpha
        out.append(y)
    return np.array(out)


FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


print("== 1. amplitude math: 1 Hz sinusoid -> RMS A/sqrt(2), excess summed ==")
fs = 50.0
T = int(20 * fs)  # 20 s
t = np.arange(T) / fs
# channel 1 (LshoY, band 0.05): A=20 deg; channel 2 (LelbP): A=6 deg (<band?)
q = np.zeros((T, 12))
A1 = np.radians(20.0)
A2 = np.radians(6.0)
q[:, 1] = A1 * np.sin(2 * np.pi * 1.0 * t)
q[:, 2] = A2 * np.sin(2 * np.pi * 1.0 * t)
total, per = run_stream(q, loops=1)
exp1 = A1 / np.sqrt(2) - 0.05
exp2 = max(0.0, A2 / np.sqrt(2) - 0.05)
check("sin A=20deg shoY excess", abs(per[1] - exp1) < 0.01,
      f"got {per[1]:.4f} expect {exp1:.4f} rad")
check("sin A=6deg elbP excess", abs(per[2] - exp2) < 0.01,
      f"got {per[2]:.4f} expect {exp2:.4f} rad")
check("quiet channels pay 0", all(p == 0 for i, p in enumerate(per)
                                  if i not in (1, 2)))
check("total == sum", abs(total - sum(per)) < 0.02,
      f"guard {total:.4f} vs sum {sum(per):.4f}")

print("== 2. DC FREEDOM: constant 56 deg elbow pays 0 ==")
q = np.zeros((T, 12))
q[:, 2] = np.radians(56.0)   # LelbP DC 56 deg
q[:, 8] = np.radians(50.0)   # RelbP DC 50
q[:, 0] = np.radians(-30.0)  # LshoR DC -30
total, per = run_stream(q, loops=1)
check("DC-only stream pays ~0", total < 1e-4, f"got {total:.6f}")

print("== 3. demo streams pay ~0 (x1_lab_v72) ==")
cols = [LAB_DOF_NAMES.index(n) for n in CHANNELS]
for clip in ("07_01", "08_03", "38_02", "103_07", "35_02", "0005_normal_walk1",
             "82_12", "f08_01"):
    d = pickle.load(open(MDIR / f"{clip}.pkl", "rb"))
    qd = d["dof_pos"][:, cols]   # already rad
    total, per = run_stream(qd, loops=8)
    worst = max(per)
    check(f"demo {clip} pays ~0", total < 0.035,
          f"total {total:.4f} rad, worst ch {worst:.4f}")

print("== 4. x7398 defect cost (walk10/walk05) ==")
for sc in ("walk10", "walk05"):
    z = np.load(REPO / f"acceptance/v83_eval/x7398_{sc}.npz")
    qh = z["q"]
    m = {n: i for i, n in enumerate(HINGE_NAMES)}
    qd = qh[:, [m[n] for n in CHANNELS]]   # already rad
    total, per = run_stream(qd, loops=8)
    idx = np.argsort(per)[::-1][:3]
    top = ", ".join(f"{CHANNELS[i].replace('_joint','')} {per[i]:.3f}"
                    for i in idx)
    check(f"x7398 {sc} cost in [0.9, 2.0]", 0.9 < total < 2.0,
          f"total {total:.3f} rad @w-0.5 = {0.5*total:.3f}/step; top: {top}")

print("== 5. shoP exclusion ==")
check("shoulder_pitch NOT in channels",
      not any("shoulder_pitch" in c for c in CHANNELS))
check("12 channels, 12 bands", len(CHANNELS) == 12 and len(BANDS) == 12)

print("== 6. EMA key isolation (arm_dc_band on same env first) ==")
q = np.zeros((T, 12))
q[:, 1] = A1 * np.sin(2 * np.pi * 1.0 * t)
env = _Env(q)
vals = []
for _ in range(2 * T):
    cfg = env.step_q()
    _r.arm_dc_band_guard(
        env, asset_cfg=cfg,
        bands=[(0.0, 0.4363), (0.0, 0.3491), (0.0, 0.2618), (0.3491, 0.6981)] * 2,
        alpha=0.0025)
    vals.append(_r.arm_distal_amp_guard(env, asset_cfg=cfg, bands=BANDS).item())
ref_total, _ = run_stream(q, loops=1)
check("identical value with armdc co-resident",
      abs(vals[-1] - ref_total) < 0.01,
      f"co-resident {vals[-1]:.4f} vs alone {ref_total:.4f}")

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("ALL PASS")
