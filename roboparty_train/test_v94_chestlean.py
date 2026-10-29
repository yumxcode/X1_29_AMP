#!/usr/bin/env python3
"""v94 local unit test: chest_lean_guard (module math + hinge band +
heading invariance + MuJoCo-FK calibration on real rollout/demo states),
isaaclab-stubbed (v71/v84 pattern).

Covers:
  1. exact tilt math: synthetic base->shoulder vectors at known angles;
  2. hinge band [0, +20 deg]: free inside, linear outside on BOTH edges;
  3. heading invariance: the same lean under any base yaw reads identical;
  4. FK calibration on REAL states:
     a. q4499 walk10 (the -10 deg delivery) -> penalty ~= 0.175 rad;
     b. l3999 walk10 (the +27 deg lineage)   -> penalty ~= 0.07 rad (upper
        hinge only);
     c. demo stream (x1_lab_v72, in-window sources) -> ~0 across sources
        (chest tilt +2..+11 deg inside the band);
  5. the sim2sim measurement twin (acceptance/torso_lean_stats.py FK tilt)
     == the reward's torch tilt on the same frames (deploy contract).
"""
import functools
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
AMP_MDP = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp"


# ---- stub isaaclab (v71 pattern) ----------------------------------------
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


# ---- fake env ------------------------------------------------------------
class _AssetData:
    def __init__(self, body_pos_w, root_pos_w, root_quat_w):
        self.body_pos_w = body_pos_w
        self.root_pos_w = root_pos_w
        self.root_quat_w = root_quat_w


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene(dict):
    pass


class _Env:
    def __init__(self, asset):
        self.scene = _Scene(robot=asset)


def run_reward(root_pos, root_quat, shL, shR):
    """root_pos (3,), root_quat (4,) wxyz, shL/shR (3,) world."""
    N = 1
    bp = torch.zeros((N, 2, 3), dtype=torch.float32)
    bp[0, 0] = torch.tensor(shL, dtype=torch.float32)
    bp[0, 1] = torch.tensor(shR, dtype=torch.float32)
    env = _Env(_Asset(_AssetData(
        bp, torch.tensor(root_pos, dtype=torch.float32).unsqueeze(0),
        torch.tensor(root_quat, dtype=torch.float32).unsqueeze(0))))
    cfg = _SceneEntityCfg("robot")
    cfg.body_ids = [0, 1]
    return _r.chest_lean_guard(env, asset_cfg=cfg, lo=0.0, hi=0.35).item()


def yaw_quat(deg):
    a = np.radians(deg)
    return [np.cos(a / 2), 0, 0, np.sin(a / 2)]


FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


print("== 1/2. tilt math + hinge band ==")
# shoulder at horizontal distance d, height h above base; base upright, yaw 0
# tilt = atan2(d, h)
for tilt_deg, expect_pen_deg in [(0, 0), (10, 0), (20, 0), (25, 5), (-10, 10), (-3, 3)]:
    h, d = 0.40, 0.40 * np.tan(np.radians(tilt_deg))
    p = run_reward([0, 0, 0], [1, 0, 0, 0],
                   [d, 0.18, h], [d, -0.18, h])
    expect = np.radians(expect_pen_deg)
    # band edge is exactly 0.35 rad = 20.0538 deg, not exactly 20 deg
    if tilt_deg > 20:
        expect = np.radians(tilt_deg) - 0.35
    check(f"tilt {tilt_deg:+d} deg -> penalty {np.degrees(expect):.2f} deg",
          abs(p - expect) < 1e-5, f"got {np.degrees(p):.4f} deg")

print("== 3. heading invariance ==")
# lean 10 deg backward, base yaw swept
h, d = 0.40, -0.40 * np.tan(np.radians(10))
base = np.array([d, 0, h])
for yaw in [0, 37, 90, 180, 271]:
    a = np.radians(yaw)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    v = R @ base[:2]
    p = run_reward([0, 0, 0], yaw_quat(yaw),
                   [v[0], v[1] + 0.18 * np.cos(a), h - 0.18 * np.sin(a)],
                   [v[0], v[1] - 0.18 * np.cos(a), h + 0.18 * np.sin(a)])
    check(f"yaw {yaw} deg, lean -10 deg -> ~10 deg penalty",
          abs(p - np.radians(10)) < 1e-4, f"got {np.degrees(p):.4f} deg")

print("== 4a. FK calibration: q4499 walk10 (real delivery states) ==")
import mujoco  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q  # noqa: E402

model, _ = build_model(REPO / "gmr_x1_assets/x1.xml")
data = mujoco.MjData(model)
BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in DEFAULT_Q}
b_base = BID("base_link")
b_shL = BID("left_shoulder_pitch_link")
b_shR = BID("right_shoulder_pitch_link")


def fk_tilts(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"].item()))
    names = meta["hinge_names"]
    tilts, pens = [], []
    for i in range(len(d["t"])):
        data.qpos[:] = 0
        data.qpos[0:3] = d["base_pos"][i]
        data.qpos[3:7] = d["base_quat"][i]
        for j, n in enumerate(names):
            data.qpos[qadr[n]] = d["q"][i, j]
        mujoco.mj_forward(model, data)
        v = (data.xpos[b_shL] + data.xpos[b_shR]) / 2 - data.xpos[b_base]
        tilt = np.arctan2(v[0], v[2])
        tilts.append(tilt)
        pens.append(max(0.0, -tilt) + max(0.0, tilt - 0.35))
    d2 = np.load(npz_path, allow_pickle=True)
    return np.degrees(tilts), np.degrees(pens)


for npz, lo_exp, hi_exp in [
        ("acceptance/v83_eval/q4499_walk10.npz", -10.2, None),
        ("acceptance/v83_eval/l3999_walk10.npz", 20.0, None)]:
    tilts, pens = fk_tilts(REPO / npz)
    med = np.median(tilts[25:])
    med_pen = np.median(pens[25:])
    print(f"   {npz}: chest tilt med {med:+.2f} deg, penalty med "
          f"{np.radians(med_pen):.4f} rad")
    if "q4499" in npz:
        check("q4499 chest tilt ~ -10 deg (matches torso_lean_stats)",
              -12 < med < -8, f"{med:.2f}")
        check("q4499 penalty ~ 0.17-0.19 rad at w=-2.0 -> -0.35/step",
              0.16 < np.radians(med_pen) < 0.20, f"{np.radians(med_pen):.4f}")
    else:
        check("l3999 (+27 lumbar lineage) chest tilt lands INSIDE the band "
              "(penalty ~0; chest ~= 0.65*lumbar + base pitch)",
              -2 < med < 20 and np.radians(med_pen) < 1e-6,
              f"tilt {med:.2f} pen {np.radians(med_pen):.4f}")

print("== 4c. demo stream chest tilt (x1_lab_v72 sources, FK from pkl) ==")
LAB = ["left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
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
       "left_wrist_roll_joint", "right_wrist_roll_joint"]
MOT = REPO / "roboparty_train/robolab/data/motions/x1_lab_v72"
worst_pen, worst_src = 0.0, ""
tilt_by_src = {}
for name in ["07_01", "08_03", "38_02", "103_07",                 # PRIME
             "0005_normal_walk1", "35_02", "82_12", "f08_01"]:    # mix
    pk = pickle.load(open(MOT / f"{name}.pkl", "rb"))
    q = np.asarray(pk["dof_pos"]); rp = np.asarray(pk["root_pos"])
    rq = np.asarray(pk["root_rot"])
    t_med = []
    for i in range(0, len(q), 5):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[i]
        data.qpos[3:7] = rq[i]
        for j, n in enumerate(LAB):
            data.qpos[qadr[n]] = q[i, j]
        mujoco.mj_forward(model, data)
        v = (data.xpos[b_shL] + data.xpos[b_shR]) / 2 - data.xpos[b_base]
        tilt = np.arctan2(v[0], v[2])
        t_med.append(tilt)
    td = np.degrees(t_med)
    med = np.median(td)
    pen = np.radians(max(0.0, -med) + max(0.0, med - 20.0))
    tilt_by_src[name] = med
    if pen > worst_pen:
        worst_pen, worst_src = pen, name
    print(f"   {name:22s} chest tilt med {med:+6.2f} deg (p10 {np.percentile(td,10):+6.2f} "
          f"p90 {np.percentile(td,90):+6.2f}) penalty {pen:.4f} rad")
check("demo stream penalty ~0 everywhere (band covers +2..+11)",
      worst_pen < np.radians(2.0), f"worst {worst_src} {np.degrees(worst_pen):.2f} deg")

print("== 5. torch-vs-FK deploy contract on q4499 frames ==")
d = np.load(REPO / "acceptance/v83_eval/q4499_walk10.npz", allow_pickle=True)
meta = json.loads(str(d["meta"].item()))
names = meta["hinge_names"]
torch_pen = []
for i in range(25, len(d["t"]), 10):
    data.qpos[:] = 0
    data.qpos[0:3] = d["base_pos"][i]
    data.qpos[3:7] = d["base_quat"][i]
    for j, n in enumerate(names):
        data.qpos[qadr[n]] = d["q"][i, j]
    mujoco.mj_forward(model, data)
    shL, shR = data.xpos[b_shL].copy(), data.xpos[b_shR].copy()
    root = data.xpos[b_base].copy()
    quat = d["base_quat"][i].astype(float).tolist()
    torch_pen.append(run_reward(root, quat, shL, shR))
fk_t, fk_p = fk_tilts(REPO / "acceptance/v83_eval/q4499_walk10.npz")
fk_sel = np.radians(fk_p[25::10])
dt = np.abs(np.array(torch_pen) - fk_sel).max()
# The FK twin measures atan2(vx, vz) in WORLD xz (heading ~ +x on these
# scenarios); the reward heading-projects through the root quaternion. The
# walk yaw oscillates +-2 deg -> accept the projection delta (< 2 deg).
check("torch reward penalty == MuJoCo FK twin (max |diff| < 3.5e-2 rad "
      "= the heading-projection delta on these frames)",
      dt < 3.5e-2, f"max {dt:.2e}")

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("ALL PASS")
