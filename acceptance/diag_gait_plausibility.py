#!/usr/bin/env python3
"""Cross-validate gait plausibility of x1_lab refs: speed vs cadence vs stride,
penetration depth distribution, and foot pitch during contact.

Uses ONLY root_pos (velocity) + geometric sole FK (correct lab-order mapping).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, find_sole_geoms, DEFAULT_Q, parse_yaml_list  # noqa: E402
import mujoco  # noqa: E402

XML = ROOT / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
LAB = parse_yaml_list(YAML, "lab_dof_names")
IDX = {n: i for i, n in enumerate(LAB)}

model, _ = build_model(XML)
data = mujoco.MjData(model)
soles, _ = find_sole_geoms(model)
feet = sorted(soles)
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LAB}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in LAB])

for name in ["0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
             "0000_treadmill_norm", "0002_treadmill_slow"]:
    p = ROOT / f"roboparty_train/robolab/data/motions/x1_lab/{name}.pkl"
    clip = pickle.load(open(p, "rb"), encoding="latin1")
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float)
    rr = np.asarray(clip["root_rot"], float)
    T, fps = len(q), float(clip["fps"])
    dt = 1.0 / fps

    v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) / dt
    # sole FK
    low = np.zeros((T, 2))
    pitch = np.zeros((T, 2))
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[qadr] = q[t]
        mujoco.mj_forward(model, data)
        for f in range(2):
            pts = np.array([data.geom_xpos[g] for g in soles[feet[f]]])
            low[t, f] = pts[:, 2].min()
            front, back = pts[:2].mean(0), pts[2:].mean(0)
            pitch[t, f] = np.degrees(np.arctan2(front[2] - back[2], 0.14))
    contact = low < 0.03
    # stance events per foot: rising edges of contact
    cads, strides = [], []
    for f in range(2):
        d = np.diff(contact[:, f].astype(int))
        up = np.where(d > 0)[0]
        if len(up) >= 3:
            cads.append(60.0 * len(up) / (T * dt))
        # stride length: root horizontal travel between consecutive stance onsets
        for i in range(len(up) - 1):
            seg = rp[up[i]:up[i + 1], :2]
            strides.append(float(np.max(np.linalg.norm(seg - seg[0], axis=1))))
    cad = np.mean(cads) if cads else 0.0
    stride = np.mean(strides) if strides else 0.0
    # implied speed from cadence x stride (2 steps per cycle)
    v_imp = cad / 60.0 * stride / 2.0 if cad else 0.0
    print(f"{name:22s} v={np.median(v):5.2f} m/s | cad={cad:6.1f} spm stride={stride:4.2f} m "
          f"v_impl={v_imp:5.2f} | pen: {np.mean(low[contact] < -0.008)*100:4.1f}% "
          f"max={np.abs(low.min())*1000:4.1f}mm | stancePitch med "
          f"L={np.median(pitch[contact[:,0],0]):+5.1f} R={np.median(pitch[contact[:,1],1]):+5.1f} deg "
          f"p90={np.percentile(np.abs(pitch[contact]),90):5.1f}")
