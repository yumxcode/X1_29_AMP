#!/usr/bin/env python3
"""v69 supplement: WHOLE-LEG abduction (hip_roll) + stance width forensics.

User clarification (2026-10-15): the 外八 is the whole leg splayed outward
via hip_roll, not just foot/hip_yaw. Quantify:
  - hip_roll DC/AC per leg (sign: + = outward abduction, verified by FK)
  - ankle y separation (stance width) policy vs demos vs anthropometric
  - hip_yaw DC (already known) alongside
Also verify heel/toe semantic once via 0002 demo FK landing lead.
"""
import functools
import json
import pickle
import glob
import math
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
    # FK sign check: left_hip_roll + => foot moves toward +y (outward)?
    def foot_y(dq):
        mujoco.mj_resetData(m, d)
        d.qpos[qadr["left_hip_pitch_joint"]] = 0.3
        d.qpos[qadr["left_knee_pitch_joint"]] = 0.5
        d.qpos[qadr["left_ankle_pitch_joint"]] = -0.2
        d.qpos[qadr["left_hip_roll_joint"]] = dq
        mujoco.mj_forward(m, d)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        return d.xpos[bid][1]
    y0, y1 = foot_y(0.0), foot_y(0.2)
    lhr_sign = "left_hip_roll + => OUTWARD(+y) abduction" if y1 > y0 else "left_hip_roll + => INWARD"
    print(f"[FK sign] left ankle y at roll0={y0:.4f} roll+0.2={y1:.4f} -> {lhr_sign}")
    mirror = "right_hip_roll - => outward (mirror)" if y1 > y0 else "right_hip_roll + => outward"

    # ---- policy rollouts ----
    print("\n=== policy rollouts ===")
    for p in sys.argv[1:]:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        hinge = list(meta["hinge_names"])
        q = z["q"].astype(np.float64)
        start = int(meta["settle_steps"]) + 2
        jn = lambda n: np.degrees(q[start:, hinge.index(n)])
        hrl, hrr = jn("left_hip_roll_joint"), jn("right_hip_roll_joint")
        # outward abduction magnitude (use signed with outward positive)
        out_sign = 1.0 if y1 > y0 else -1.0
        l_out, r_out = out_sign * hrl, -out_sign * hrr
        sole = z["sole_xyz"].astype(np.float64)
        # stance width: y distance between feet mid points
        fy = np.abs(sole[start:, 0, :, 1].mean(1) - sole[start:, 1, :, 1].mean(1))
        bq = z["base_quat"].astype(np.float64)
        print(f"{Path(p).stem:34s} hipRoll_out L={l_out.mean():+5.1f} R={r_out.mean():+5.1f} deg "
              f"(AC ptp L={np.ptp(detrend(hrl)):.1f} R={np.ptp(detrend(hrr)):.1f}) "
              f"stanceW mean={fy.mean()*1000:.0f}mm p95={np.percentile(fy,95)*1000:.0f}mm")

    # ---- demos (FK replay for stance width; dof for hip_roll) ----
    print("\n=== demos (dof stats + FK stance width) ===")
    for f in sorted(glob.glob("roboparty_train/robolab/data/motions/x1_lab_v32/*.pkl")):
        if "mirror" in f:
            continue
        data = pickle.load(open(f, "rb"))
        dof = np.asarray(data["dof_pos"])
        root_pos, root_rot = np.asarray(data["root_pos"]), np.asarray(data["root_rot"])
        i = GMR.index
        hrl = np.degrees(dof[:, i("left_hip_roll_joint")])
        hrr = np.degrees(dof[:, i("right_hip_roll_joint")])
        l_out = out_sign * hrl
        r_out = -out_sign * hrr
        # FK stance width every 5th frame
        bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        ws = []
        for k in range(0, len(dof), 5):
            mujoco.mj_resetData(m, d)
            d.qpos[0:3] = root_pos[k]
            d.qpos[3:7] = root_rot[k] if np.linalg.norm(root_rot[k]) > 0.5 else [1, 0, 0, 0]
            for j, nm in enumerate(GMR):
                d.qpos[qadr[nm]] = dof[k, j]
            mujoco.mj_forward(m, d)
            ws.append(abs(d.xpos[bl][1] - d.xpos[br][1]))
        ws = np.array(ws)
        print(f"{Path(f).stem:30s} hipRoll_out L={l_out.mean():+5.1f} R={r_out.mean():+5.1f} deg "
              f"(AC ptp L={np.ptp(detrend(hrl)):.1f} R={np.ptp(detrend(hrr)):.1f}) "
              f"stanceW mean={ws.mean()*1000:.0f}mm p95={np.percentile(ws,95)*1000:.0f}mm")


if __name__ == "__main__":
    main()
