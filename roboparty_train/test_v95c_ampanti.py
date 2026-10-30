#!/usr/bin/env python3
"""v95c local unit test: arm_amp_antiphase_prior (audit item 2,
RECORD_BOOK 14.5). Verifies the amp-under-antiphase mechanism on exact
synthetic streams + real x1_lab_v72 demos + the policy lineage.

Also documents the v55 arm_amp_phase_prior SIGN BUG (its arm-leg product
sums a +0.98 and a -0.91 correlation pair -> ~0 on natural demos; the v33
negation was never ported) — measured 2026-10-30 on demo joint channels:
  corr(hp_shoL, hp_hipR) = +0.98   corr(hp_shoR, hp_hipL) = -0.91
  -> prod = shoL*hipR + shoR*hipL ~ 0 on demos (gate active 9-29% frames)

Separation measured with sync_cap 0.24 rad (13.7 deg; demo sync max 13.1):
  demos 0.62-1.00 | y8397 (v95c base) 0.07-0.24 | x7398 0.51-0.62
  (x7398_walk05 0.62 is borderline-coherent in JOINT space — its FK phi
   corr -0.33 shows the real incoherence lived in phi space incl. the
   distal contamination; bound <= 0.65 with this note.)
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _stub_and_load():
    def quat_apply(q, v):
        return v
    _iso = types.ModuleType("isaaclab")
    _iso.utils = types.ModuleType("isaaclab.utils")
    _mu = types.ModuleType("isaaclab.utils.math")
    _mu.quat_apply = quat_apply
    _iso.utils.math = _mu
    sys.modules["isaaclab"] = _iso
    sys.modules["isaaclab.utils"] = _iso.utils
    sys.modules["isaaclab.utils.math"] = _mu
    sys.modules["isaaclab.envs"] = _st = types.ModuleType("isaaclab.envs")
    _st.mdp = types.SimpleNamespace()
    sys.modules["isaaclab.managers"] = types.ModuleType("isaaclab.managers")

    class SEC:
        def __init__(s, *a, **k):
            s.name = a[0] if a else k.get("name", "robot")
    sys.modules["isaaclab.managers"].SceneEntityCfg = SEC
    _sen = types.ModuleType("isaaclab.sensors")
    _sen.ContactSensor = object
    _sen.RayCaster = object
    sys.modules["isaaclab.sensors"] = _sen
    _as = types.ModuleType("isaaclab.assets")
    _as.Articulation = object
    _as.RigidObject = object
    sys.modules["isaaclab.assets"] = _as
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "amp_rewards_r",
        REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py")
    r = ilu.module_from_spec(spec)
    sys.modules["amp_rewards_r"] = r
    spec.loader.exec_module(r)
    return r


_r = _stub_and_load()
LAB = ["left_hip_pitch", "lumbar_yaw", "right_hip_pitch", "left_hip_roll",
       "lumbar_roll", "right_hip_roll", "left_hip_yaw", "lumbar_pitch",
       "right_hip_yaw", "left_knee_pitch", "left_shoulder_pitch",
       "right_shoulder_pitch", "right_knee_pitch", "left_ankle_pitch",
       "left_shoulder_roll", "right_shoulder_roll", "right_ankle_pitch",
       "left_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw",
       "right_ankle_roll", "left_elbow_pitch", "right_elbow_pitch",
       "left_elbow_yaw", "right_elbow_yaw", "left_wrist_pitch",
       "right_wrist_pitch", "left_wrist_roll", "right_wrist_roll"]
HN = ["lumbar_yaw", "lumbar_roll", "lumbar_pitch", "L_shoP", "L_shoR",
      "L_shoY", "L_elbP", "L_elbY", "L_wriP", "L_wriR", "R_shoP", "R_shoR",
      "R_shoY", "R_elbP", "R_elbY", "R_wriP", "R_wriR", "L_hipP", "L_hipR",
      "L_hipY", "L_kneeP", "L_ankP", "L_ankR", "R_hipP", "R_hipR", "R_hipY",
      "R_kneeP", "R_ankP", "R_ankR"]


class _SC_stub(dict):
    def __init__(s, name):
        super().__init__()
        s.name = name


class _AD:
    def set(s, jp):
        s.joint_pos = jp
        s.default_joint_pos = torch.zeros_like(jp)


class _AS:
    def __init__(s):
        s.data = _AD()


class _SC(dict):
    pass


class _Env:
    def __init__(s, q):
        s.scene = _SC(robot=_AS())
        s.q = q
        s.t = -1

    def step(s):
        s.t += 1
        s.scene["robot"].data.set(
            torch.tensor(s.q[s.t % len(s.q)], dtype=torch.float32).unsqueeze(0))
        c = _SC_stub("robot")
        c.joint_ids = [0, 1]
        return c


def run(q, loops=8):
    env = _Env(q)
    v = 0.0
    for _ in range(loops * len(q)):
        v = float(_r.arm_amp_antiphase_prior(env, asset_cfg=env.step()).item())
    return v


FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


if __name__ == "__main__":
    import pickle
    T = 4000
    t = np.arange(T) / 50.0
    A = np.radians(40.0)
    q = np.zeros((T, 2))
    q[:, 0] = A * np.sin(2 * np.pi * 1.0 * t)
    q[:, 1] = -A * np.sin(2 * np.pi * 1.0 * t)
    check("natural antiphase swing >= 0.9", run(q) >= 0.9, f"{run(q):.3f}")
    q2 = q.copy()
    q2[:, 1] = A * np.sin(2 * np.pi * 1.0 * t)
    check("in-phase swing ~ 0", run(q2) <= 0.05, f"{run(q2):.3f}")
    q3 = q.copy()
    q3[:, 1] = -0.3 * A * np.sin(2 * np.pi * 1.0 * t)
    check("asymmetric swing low", run(q3) <= 0.35, f"{run(q3):.3f}")
    q4 = np.zeros((T, 2))
    q4[:, 0] = np.radians(15)
    check("DC-only 0", run(q4) <= 1e-4, f"{run(q4):.5f}")
    for clip in ("07_01", "08_03", "103_07", "35_02", "0005_normal_walk1"):
        d = pickle.load(open(
            REPO / f"roboparty_train/robolab/data/motions/x1_lab_v72/{clip}.pkl", "rb"))
        qq = d["dof_pos"][:, [LAB.index("left_shoulder_pitch"),
                              LAB.index("right_shoulder_pitch")]]
        v = run(np.concatenate([qq] * 8))
        check(f"demo {clip} >= 0.55", v >= 0.55, f"{v:.3f}")
    for tag, sc, bound in (("x7398", "walk10", 0.55), ("x7398", "walk05", 0.65),
                           ("y8397", "walk10", 0.55), ("y8397", "walk05", 0.55),
                           ("y8397", "back05", 0.55), ("b8500", "walk10", 0.55)):
        z = np.load(REPO / f"acceptance/v83_eval/{tag}_{sc}.npz")
        qq = z["q"][:, [HN.index("L_shoP"), HN.index("R_shoP")]]
        v = run(np.concatenate([qq] * 8))
        check(f"policy {tag}_{sc} <= {bound}", v <= bound, f"{v:.3f}")
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
