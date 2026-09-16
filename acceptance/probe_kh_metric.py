#!/usr/bin/env python3
"""Calibration probe: incline-lead H1/H2/H3 classification (v56 metric design).

Compares contact-ORDER classification vs GEOMETRIC LEAD at touchdown/liftoff
on the calibration references (0002 heel-first exemplar, 0005 toe-drag
anti-exemplar) and the soup534_40 policy baseline. See gait_metrics.py KH
section for the final locked definition.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, ".")
from sim2sim.mujoco_rollout import build_model, parse_yaml_list  # noqa: E402
from sim2sim.gait_metrics import schmitt_contact, stance_windows  # noqa: E402

model, _ = build_model("gmr_x1_assets/x1.xml")
data = mujoco.MjData(model)
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def loff(y):
    return np.array([[0.03, y, 0.07], [-0.03, y, 0.07],
                     [0.03, y, -0.07], [-0.03, y, -0.07]])


def fk_sole(clip, ds_to_50hz=True):
    c = pickle.load(open(
        f"roboparty_train/robolab/data/motions/x1_lab_v32/{clip}.pkl", "rb"))
    q = np.asarray(c["dof_pos"]); rr = np.asarray(c["root_rot"]); rp = np.asarray(c["root_pos"])
    fps = float(c.get("fps", 120.0))
    ds = max(1, int(round(fps / 50.0))) if ds_to_50hz else 1
    idx = np.arange(0, len(q), ds)
    q, rr, rp = q[idx], rr[idx], rp[idx]
    T = len(q)
    hinge = [model.joint(i).name for i in range(model.njnt)
             if model.joint(i).name != "root"]
    hinge = [n for n in hinge if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    lab = parse_yaml_list("roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml", "lab_dof_names")
    lab2mj = np.array([hinge.index(n) for n in lab])
    sole = np.zeros((T, 2, 4, 3))
    qh = np.zeros((T, len(hinge)))
    qh[:, lab2mj] = q
    for t in range(T):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = rp[t]
        n = np.linalg.norm(rr[t])
        data.qpos[3:7] = rr[t] / n
        data.qpos[qadr] = qh[t]
        mujoco.mj_forward(model, data)
        for fi, bn in enumerate(FOOT_NAMES):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            R = data.xmat[bid].reshape(3, 3)
            y = -0.0408 if fi == 0 else 0.0408
            sole[t, fi] = ((R @ loff(y).T).T + data.xpos[bid])
    return sole


def classify(sole, foot, lead=0.008, start=3):
    bottom = sole[:, foot, :, 2] - 0.002
    heel_b = bottom[:, 0:2].min(1)
    toe_b = bottom[:, 2:4].min(1)
    contact = schmitt_contact(bottom.min(1))
    wins = stance_windows(contact, start)
    land = {"heel": 0, "toe": 0, "flat": 0}
    launch = {"toeoff": 0, "flat": 0}
    for (td, lo) in wins:
        lh = toe_b[td] - heel_b[td]          # + = heel lower at TD
        if lh >= lead:
            land["heel"] += 1
        elif lh <= -lead:
            land["toe"] += 1
        else:
            land["flat"] += 1
        lt = heel_b[lo] - toe_b[lo]          # + = toe lower at liftoff
        if lt >= lead:
            launch["toeoff"] += 1
        else:
            launch["flat"] += 1
    n = max(1, len(wins))
    return (land["heel"] / n * 100, land["toe"] / n * 100,
            launch["toeoff"] / n * 100, len(wins))


def main():
    for thr in (0.006, 0.008, 0.012):
        print(f"--- lead threshold {thr*1000:.0f}mm ---")
        for tag, clip in [("0002(ref)", "0002_treadmill_slow"),
                          ("0005(ref)", "0005_normal_walk1")]:
            sole = fk_sole(clip)
            for f in (0, 1):
                h, t, o, n = classify(sole, f, thr)
                print(f"  {tag} foot{f}: heel={h:5.1f}% toe={t:5.1f}% toeoff={o:5.1f}% n={n}")
        for fn in ["acceptance/v54_eval/soup534_40_walk10.npz",
                   "acceptance/v54_eval/soup534_40_walk05.npz"]:
            z = np.load(fn, allow_pickle=False)
            sole = z["sole_xyz"].astype(float)
            settle = int(json.loads(str(z["meta"]))["settle_steps"]) + 2
            for f in (0, 1):
                h, t, o, n = classify(sole[settle:], f, thr)
                print(f"  {Path(fn).stem[:20]} foot{f}: heel={h:5.1f}% toe={t:5.1f}% toeoff={o:5.1f}% n={n}")


if __name__ == "__main__":
    main()
