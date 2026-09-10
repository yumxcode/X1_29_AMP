#!/usr/bin/env python3
"""Diagnose arm swing in the RAW AMASS SMPLX source (stageii npz).

SMPLX body joint order (first 22 of 55):
 0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2,
 7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck,
 13 L_collar, 14 R_collar, 15 head, 16 L_shoulder, 17 R_shoulder,
 18 L_elbow, 19 R_elbow, 20 L_wrist, 21 R_wrist

Conventions (T-pose = identity local frames, arms along +-x, world y up):
 - shoulder "swing" = rotation about local y (pendulum in xz plane).
   NOTE: SAME-sign theta_L & theta_R = symmetric swing (L fwd & R back),
   because the arms point in opposite x directions.
 - elbow bend = rotation about local x (hinge), >0 = flex.
 - upper-body-vs-pelvis yaw = spine1*spine2*spine3 chain, y-Euler of R.
 - pelvis global yaw for reference.

All angles in degrees. Compare with acceptance/diag_arm_swing.py output.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = {
    "BMLrub": ROOT / "AMASS_minimal/BMLrub_stageii",
    "CMU": ROOT / "AMASS_minimal/CMU",
}
CLIPS = ["0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
         "0000_treadmill_norm", "0002_treadmill_slow", "0009_normal_jog1",
         "0026_circle_walk", "114_08", "114_09", "127_04", "127_06",
         "36_01", "36_11"]


def aa_to_mat(aa):
    t = np.linalg.norm(aa)
    if t < 1e-9:
        return np.eye(3)
    k = aa / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def yaw_of(R):
    """Y-Euler angle (deg) of rotation matrix (ZYX decomposition)."""
    return np.degrees(np.arctan2(-R[2, 0], R[0, 0]))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def analyze(path):
    z = np.load(path, allow_pickle=True)
    P = np.asarray(z["poses"], dtype=np.float64)[:, :66].reshape(-1, 22, 3)
    T = len(P)
    shoL = np.array([aa_to_mat(p[16])[0, 2] for p in P])  # sin of local-y rot
    shoR = np.array([aa_to_mat(p[17])[0, 2] for p in P])
    angL = np.degrees(np.arcsin(np.clip(shoL, -1, 1)))
    angR = np.degrees(np.arcsin(np.clip(shoR, -1, 1)))
    elbL = np.array([np.degrees(aa_to_mat(p[18])[1, 2]) for p in P])  # flex
    elbR = np.array([np.degrees(aa_to_mat(p[19])[1, 2]) for p in P])
    hipL = np.array([np.degrees(np.arcsin(np.clip(aa_to_mat(p[1])[0, 2], -1, 1))) for p in P])
    hipR = np.array([np.degrees(np.arcsin(np.clip(aa_to_mat(p[2])[0, 2], -1, 1))) for p in P])
    spine_yaw = []
    pelvis_yaw = []
    for p in P:
        R = aa_to_mat(p[3]) @ aa_to_mat(p[6]) @ aa_to_mat(p[9])
        spine_yaw.append(yaw_of(R))
        pelvis_yaw.append(yaw_of(aa_to_mat(p[0])))
    spine_yaw = np.array(spine_yaw); pelvis_yaw = np.array(pelvis_yaw)

    rng = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))
    return dict(clip=path.stem.replace("_stageii", ""), T=T,
                shoL=(float(angL.mean()), rng(angL)),
                shoR=(float(angR.mean()), rng(angR)),
                # same-sign corr => symmetric antiphase swing in world
                symm=corr(angL, angR),
                xL_hipR=corr(angL, hipR),
                elbL=(float(elbL.mean()), rng(elbL)),
                elbR=(float(elbR.mean()), rng(elbR)),
                spineY=(float(spine_yaw.mean()), rng(spine_yaw)),
                pelvYawsw=rng(pelvis_yaw), hipsw=rng(hipL))


if __name__ == "__main__":
    print(f"{'clip':22s} {'T':>4s} | {'shoL mean/swing':>15s} {'shoR':>15s} "
          f"| {'symLR':>6s} {'xL-hipR':>8s} | {'elbL mean/swing':>15s} {'elbR':>15s} "
          f"| {'spineYaw mean/swing':>19s} {'pelvYawsw':>9s} {'hipsw':>6s}")
    for c in CLIPS:
        for sub, d in SRC.items():
            p = d / f"{c}_stageii.npz"
            if p.exists():
                r = analyze(p)
                print(f"{r['clip']:22s} {r['T']:4d} | "
                      f"{r['shoL'][0]:6.1f} / {r['shoL'][1]:5.1f} "
                      f"{r['shoR'][0]:6.1f} / {r['shoR'][1]:5.1f} | "
                      f"{r['symm']:+.2f}  {r['xL_hipR']:+.2f} | "
                      f"{r['elbL'][0]:6.1f} / {r['elbL'][1]:5.1f} "
                      f"{r['elbR'][0]:6.1f} / {r['elbR'][1]:5.1f} | "
                      f"{r['spineY'][0]:6.1f} / {r['spineY'][1]:5.1f} "
                      f"{r['pelvYawsw']:9.1f} {r['hipsw']:6.1f}")
    print("""
Legend (deg):
  shoL/shoR mean/swing = shoulder swing (about local y, world-pendulum) mean / p95-p5 range
  symLR    = corr(swing_L, swing_R) IN SMPLX LOCAL AXES: +1 => symmetric swing
             (L fwd while R back) because arms point along opposite x. Compare
             with antiLR in diag_arm_swing.py (X1 axes are anti-aligned: -1 there = same motion).
  elbL/elbR = elbow flexion (about local x), >0 = forearm bent forward/up
  spineYaw = upper body (spine1-3 chain) yaw RELATIVE TO pelvis
  pelvYawsw = pelvis global yaw swing (whole-body heading wobble)
""")
