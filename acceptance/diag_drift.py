#!/usr/bin/env python3
"""Diagnose lateral drift: heading bias vs crab walk vs torso asymmetry.

v32b drift +0.30 m -> v33b +4.46 -> v34 +3.49 (cmd 1.0 straight-x).
Uses the rollout npz logs (base_pos/base_quat/q) to separate:
  heading bias : constant yaw offset, machine walks straight but pointed wrong
  crab walk    : body-frame y velocity while heading is correct
  torso asym   : lumbar roll/yaw DC that could couple into steering
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model  # noqa: E402

import mujoco  # noqa: E402

model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
data = mujoco.MjData(model)
BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)


def yaw_of(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


for label, path in [("v32b", "/tmp/p7_v32b.npz"),
                    ("v33b", "/tmp/p7_v33b3999.npz"),
                    ("v34", "/tmp/p7_v34.npz"),
                    ("v35", "/tmp/p7_v35.npz"),
                    ("v36", "/tmp/p7_v36.npz"),
                    ("v37", "/tmp/p7_v37.npz")]:
    p = Path(path)
    if not p.exists():
        continue
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    settle = int(meta.get("settle_steps", 0))
    hinge = list(meta["hinge_names"])
    q = np.asarray(d["q"], float)[settle:]
    bp = np.asarray(d["base_pos"], float)[settle:]
    bq = np.asarray(d["base_quat"], float)[settle:]
    T = len(q)
    idx = {n: k for k, n in enumerate(hinge)}

    # heading over time
    yaw = np.array([yaw_of(bq[t]) for t in range(T)])
    # body-frame velocity from world velocity + heading (approx via finite diff)
    v_world = np.gradient(bp, axis=0) * 50.0
    t0, t1 = 50, T - 50
    vb_mid = []
    for t in range(t0, t1):
        th = np.radians(yaw[t])
        c, s = np.cos(th), np.sin(th)
        vx = c * v_world[t, 0] + s * v_world[t, 1]
        vy = -s * v_world[t, 0] + c * v_world[t, 1]
        vb_mid.append((vx, vy))
    vb_mid = np.array(vb_mid)
    drift_rate = (bp[-1, 1] - bp[0, 1]) / (T * 0.02)
    print(f"\n== {label} ==")
    print(f"  world y drift {bp[-1,1]-bp[0,1]:+.2f} m ({drift_rate:+.2f} m/s) | final heading {yaw[-1]:+.1f} deg | heading drift {(yaw[-1]-yaw[0])/(T*0.02):+.2f} deg/s")
    print(f"  body-frame vel (mid): vx {vb_mid[:,0].mean():+.2f} vy {vb_mid[:,1].mean():+.2f} m/s (cmd 1.0/0.0)")
    lr = np.degrees(q[:, idx["lumbar_roll_joint"]])
    ly = np.degrees(q[:, idx["lumbar_yaw_joint"]])
    hr = np.degrees(q[:, idx["right_hip_roll_joint"]])
    hl = np.degrees(q[:, idx["left_hip_roll_joint"]])
    print(f"  torso DC: lumRoll {lr.mean():+5.1f} | lumYaw {ly.mean():+5.1f} | hipRoll L {hl.mean():+5.1f} R {hr.mean():+5.1f} deg")
