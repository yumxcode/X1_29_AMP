#!/usr/bin/env python3
"""v69: FK replay of demo library — quantify toe-out / waist sway in DEMO domain.

Reads x1_lab_v32 pkl (root_pos, root_rot, dof_pos in gmr order), replays
through the MuJoCo model, measures per-clip:
  - foot heading (base frame, deg) DC + AC: 外八
  - lumbar_roll / lumbar_yaw AC amplitude: 腰摆
  - arm swing decomposition: torso-yaw-driven vs shoulder-driven (world frame)
"""
import functools
import glob
import math
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

GMR = ["lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
       "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
       "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
       "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
       "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint",
       "left_ankle_pitch_joint", "left_ankle_roll_joint",
       "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
       "right_ankle_pitch_joint", "right_ankle_roll_joint"]


def detrend(x):
    t = np.arange(len(x), dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def main():
    import mujoco
    m = mujoco.MjModel.from_xml_path("gmr_x1_assets/x1.xml")
    d = mujoco.MjData(m)
    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in GMR}
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
           ("left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_pitch_link", "right_wrist_pitch_link",
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",
            "lumbar_pitch_link", "base_link")}

    print(f"{'clip':30s} {'L_ft_hdg':>16s} {'R_ft_hdg':>16s} {'lr_AC':>6s} {'ly_AC':>6s} "
          f"{'hipyawL':>8s} {'hipyawR':>8s} {'wristX_AC':>9s}")
    for f in sorted(glob.glob("roboparty_train/robolab/data/motions/x1_lab_v32/*.pkl")):
        if "mirror" in f:
            continue
        data = pickle.load(open(f, "rb"))
        root_pos = np.asarray(data["root_pos"])
        root_rot = np.asarray(data["root_rot"])   # (n,4) wxyz assumed
        dof = np.asarray(data["dof_pos"])
        n = len(dof)
        L_hd, R_hd, lr, ly, wl_x, wr_x, hyL, hyR = [], [], [], [], [], [], [], []
        for i in range(n):
            mujoco.mj_resetData(m, d)
            d.qpos[0:3] = root_pos[i]
            d.qpos[3:7] = root_rot[i] if np.linalg.norm(root_rot[i]) > 0.5 else [1, 0, 0, 0]
            for j, name in enumerate(GMR):
                d.qpos[qadr[name]] = dof[i, j]
            mujoco.mj_forward(m, d)
            # base yaw
            q = d.qpos[3:7]
            byaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                              1 - 2 * (q[2] ** 2 + q[3] ** 2))
            # foot heading: local +z of ankle frame in base frame
            for side, acc in (("left", L_hd), ("right", R_hd)):
                xmat = d.xmat[bid[f"{side}_ankle_roll_link"]].reshape(3, 3)
                fwd = xmat[:, 2]  # local z axis in world
                hdg = math.atan2(fwd[1], fwd[0]) - byaw
                acc.append((math.degrees(hdg) + 180) % 360 - 180)
            lr.append(math.degrees(d.qpos[qadr["lumbar_roll_joint"]]))
            ly.append(math.degrees(d.qpos[qadr["lumbar_yaw_joint"]]))
            hyL.append(math.degrees(d.qpos[qadr["left_hip_yaw_joint"]]))
            hyR.append(math.degrees(d.qpos[qadr["right_hip_yaw_joint"]]))
            wl_x.append(d.xpos[bid["left_wrist_pitch_link"]][0])
            wr_x.append(d.xpos[bid["right_wrist_pitch_link"]][0])
        L_hd, R_hd = np.array(L_hd), np.array(R_hd)
        lr, ly = np.array(lr), np.array(ly)
        wl_x, wr_x = np.array(wl_x), np.array(wr_x)
        print(f"{Path(f).stem:30s} {L_hd.mean():+6.1f}(sd{L_hd.std():4.1f}) "
              f"{R_hd.mean():+6.1f}(sd{R_hd.std():4.1f}) "
              f"{np.ptp(detrend(lr)):6.1f} {np.ptp(detrend(ly)):6.1f} "
              f"{np.mean(hyL):+7.1f} {np.mean(hyR):+7.1f} "
              f"{(np.ptp(detrend(wl_x))+np.ptp(detrend(wr_x)))/2*1000:8.0f}mm")


if __name__ == "__main__":
    main()
