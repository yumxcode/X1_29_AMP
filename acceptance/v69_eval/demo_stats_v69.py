#!/usr/bin/env python3
"""v69: demo-library hip_yaw / lumbar_roll statistics vs m11498 policy.

Question: is the 外八/toe-out gait DEMO-driven (AMP style) or POLICY-side
(reset-pose bias + reward blind spot)? Compares x1_lab_v32 dof stats with
the m11498 rollout npz (same joint names, gmr dof order for pkl).
"""
import functools
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

# gmr_dof_names order (retarget config, verified)
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
    print("=== DEMO library (x1_lab_v32, non-mirror) ===")
    print(f"{'clip':30s} {'L_hipyaw m/p95':>16s} {'R_hipyaw m/p95':>16s} {'lr_mean':>8s} {'lr_ptp':>7s} {'hippitch_ptp':>12s}")
    for f in sorted(glob.glob("roboparty_train/robolab/data/motions/x1_lab_v32/*.pkl")):
        if "mirror" in f:
            continue
        d = pickle.load(open(f, "rb"))
        dof = np.asarray(d["dof_pos"])
        i = GMR.index
        L = np.degrees(dof[:, i("left_hip_yaw_joint")])
        R = np.degrees(dof[:, i("right_hip_yaw_joint")])
        lr = np.degrees(dof[:, i("lumbar_roll_joint")])
        hpp = np.degrees(dof[:, i("left_hip_pitch_joint")])
        print(f"{Path(f).stem:30s} {L.mean():+6.1f}/{np.percentile(L,95):+6.1f} "
              f"{R.mean():+6.1f}/{np.percentile(R,95):+6.1f} "
              f"{lr.mean():+7.1f} {lr.max()-lr.min():6.1f} {np.ptp(detrend(hpp)):11.1f}")

    print("\n=== m11498 policy rollouts ===")
    for p in sys.argv[1:]:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        hinge = list(meta["hinge_names"])
        q = z["q"].astype(np.float64)
        start = int(meta["settle_steps"]) + 2
        jn = lambda n: np.degrees(q[start:, hinge.index(n)])
        L, R = jn("left_hip_yaw_joint"), jn("right_hip_yaw_joint")
        lr = jn("lumbar_roll_joint")
        # in-place AC amplitude (peak-to-peak of detrended)
        print(f"{Path(p).stem:34s} L={L.mean():+6.1f} [{L.min():+5.1f},{L.max():+5.1f}] "
              f"R={R.mean():+6.1f} [{R.min():+5.1f},{R.max():+5.1f}] "
              f"lr={lr.mean():+6.1f} ptp={lr.max()-lr.min():5.1f}")


if __name__ == "__main__":
    main()
