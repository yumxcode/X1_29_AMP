#!/usr/bin/env python3
"""Heel-toe gait audit on the x1_lab_v32 reference dataset.

For each clip: roll the retargeted qpos through MuJoCo FK, compute world
heights of the 2 heel + 2 toe sole spheres per foot, detect stance events,
and classify each stance by its landing/launch pattern:
  heel-first: heel z crosses contact before toe z
  toe-first : toe first (the G3 defect the eval guards against)
  flat      : both within the same frame
Launch side:
  toe-off  : heel lifts while toe still down (classic heel-off->toe-off)
  flat/horizontal: both lift together
Also report ankle_pitch range during stance (rolling requires dorsiflexion
-> plantarflexion sweep).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, ".")
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q  # noqa: E402

ROOT = Path(".").resolve()
XML = ROOT / "gmr_x1_assets" / "x1.xml"
DSDIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32"

model, n_feet = build_model(XML)
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])

# lab dof order == pkl dof_pos column order (Isaac lab order, per repo convention)
# hinge[] is mj XML order; map lab col -> mj slot
from sim2sim.mujoco_rollout import parse_yaml_list  # noqa: E402
lab_dof = parse_yaml_list(ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml", "lab_dof_names")
lab2mj = np.array([hinge.index(n) for n in lab_dof])

# sole sphere local offsets (from x1.xml, ankle_roll_link frame)
# NOTE: left foot y=-0.0408, right foot y=+0.0408 (mirrored in the XML)
def loff(side):
    y = -0.0408 if side == "L" else 0.0408
    return np.array([
        [0.03, y, 0.07], [-0.03, y, 0.07],      # toe end (local z +0.07)
        [0.03, y, -0.07], [-0.03, y, -0.07],    # heel end (local z -0.07)
    ])
FOOTS = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}
ANKLE = {"L": "left_ankle_pitch_joint", "R": "right_ankle_pitch_joint"}
CONTACT_Z = 0.010  # 10mm threshold (sphere r=2mm + margin)

data = mujoco.MjData(model)
# identify which end is toe: at default pose, forward = larger world x
mujoco.mj_resetData(model, data)
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[qadr] = np.array([DEFAULT_Q[n] for n in hinge])
mujoco.mj_forward(model, data)
TOE_END = None
for side, bname in FOOTS.items():
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    R = data.xmat[bid].reshape(3, 3)
    world = (R @ loff('L').T).T + data.xpos[bid]
    TOE_END = 0 if world[0, 0] > world[2, 0] else 2
print(f"[INFO] toe end = sphere index {TOE_END} (local z {'+' if TOE_END==0 else '-'}0.07)")

def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q / np.linalg.norm(q))
    return m.reshape(3, 3)

def analyze(clip_path):
    c = pickle.load(open(clip_path, "rb"))
    q = np.asarray(c["dof_pos"], dtype=np.float64)
    T = len(q)
    qpos = np.zeros((T, model.nq))
    qpos[:, 0:3] = c["root_pos"]
    rr = np.asarray(c["root_rot"], dtype=np.float64)
    qpos[:, 3:7] = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    # lab dof cols -> mj qpos slots
    qpos[:, qadr[lab2mj]] = q
    fps = float(c.get("fps", 120.0))

    res = {}
    for side, bname in FOOTS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        heel_z = np.zeros(T); toe_z = np.zeros(T)
        ank = np.zeros(T)
        ai = qadr[hinge.index(ANKLE[side])]  # mj qpos addr of ankle pitch
        # map back to pkl column
        col = int(np.where(lab2mj == hinge.index(ANKLE[side]))[0][0])
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            R = data.xmat[bid].reshape(3, 3)
            world = (R @ loff(side).T).T + data.xpos[bid]
            heel_z[t] = world[[0, 1], 2].min()
            toe_z[t] = world[[2, 3], 2].min()
            ank[t] = q[t, col]
        hc = heel_z < CONTACT_Z
        tc = toe_z < CONTACT_Z
        on = hc | tc
        # stance intervals
        ivs, s = [], None
        for t in range(T):
            if on[t] and s is None: s = t
            if (not on[t] or t == T - 1) and s is not None:
                e = t - 1 if not on[t] else t
                if e - s >= int(0.15 * fps):  # >=150ms stance
                    ivs.append((s, e))
                s = None
        land, launch, flat_frac, ank_sweep = {"heel": 0, "toe": 0, "flat": 0}, {"toeoff": 0, "flat": 0}, [], []
        for (s, e) in ivs:
            th = np.argmax(hc[s:e]) if hc[s:e].any() else None
            tt = np.argmax(tc[s:e]) if tc[s:e].any() else None
            if th is None or tt is None: continue
            if th < tt - 1: land["heel"] += 1
            elif tt < th - 1: land["toe"] += 1
            else: land["flat"] += 1
            lh = T - 1 - np.argmax(hc[s:e][::-1])
            lt = T - 1 - np.argmax(tc[s:e][::-1])
            if lh < lt - 1: launch["toeoff"] += 1
            else: launch["flat"] += 1
            flat = int(np.sum(hc[s:e] & tc[s:e])) / (e - s + 1)
            flat_frac.append(flat)
            ank_sweep.append(ank[s:e + 1].max() - ank[s:e + 1].min())
        res[side] = dict(n=len(ivs), land=land, launch=launch,
                         flat=float(np.mean(flat_frac)) if flat_frac else 0,
                         ank=float(np.degrees(np.mean(ank_sweep))) if ank_sweep else 0)
    return res

clips = sorted(p.stem for p in DSDIR.glob("*.pkl") if "mirror" not in p.stem)
print(f"{'clip':28s} {'foot':4s} {'stn':>3s} {'heel1st':>7s} {'toe1st':>6s} {'flat1st':>7s} "
      f"{'toeoff':>6s} {'flatOff':>7s} {'flat%':>5s} {'ankSwp°':>7s}")
for cl in clips:
    r = analyze(DSDIR / f"{cl}.pkl")
    for side in ("L", "R"):
        d = r[side]
        n = max(d["land"]["heel"] + d["land"]["toe"] + d["land"]["flat"], 1)
        print(f"{cl:28s} {side:4s} {d['n']:3d} "
              f"{d['land']['heel']/n*100:6.1f}% {d['land']['toe']/n*100:5.1f}% {d['land']['flat']/n*100:6.1f}% "
              f"{d['launch']['toeoff']/n*100:5.1f}% {d['launch']['flat']/n*100:6.1f}% "
              f"{d['flat']*100:4.0f}% {d['ank']:6.1f}")
