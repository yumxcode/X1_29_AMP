#!/usr/bin/env python3
"""Diagnose arm swing in retargeted AMASS clips (x1_gmr / x1_lab).

Question: does the TRAINING REFERENCE contain antiphase shoulder-pitch arm
swing (natural human walking), or elbow-flexed "forearm raise"? And how much
lumbar (waist) swing does the reference use?

Quantities per clip:
  - shoulder pitch L/R: mean, swing range (p95-p5), antiphase corr(L,R)
  - elbow pitch L/R and elbow yaw L/R: mean, range  (elbow bend -> forearm raised)
  - shoulder roll L/R: mean, range (lateral raise)
  - lumbar yaw: swing range, corr with L/R shoulder pitch
  - hip pitch L/R: swing range (gait rhythm reference), corr(arm, contralateral hip)

x1_gmr clips use gmr_dof_names order; x1_lab uses lab_dof_names order.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GMR_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_gmr"
LAB_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab"

GMR_ORDER = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
LAB_ORDER = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "right_knee_pitch_joint", "left_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "right_ankle_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_elbow_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]

CLIPS = ["0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
         "0000_treadmill_norm", "0002_treadmill_slow", "0009_normal_jog1",
         "0026_circle_walk", "114_08", "114_09", "127_04", "127_06", "36_01", "36_11"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def analyze(pkl_path, order):
    clip = pickle.load(open(pkl_path, "rb"), encoding="latin1")
    q = np.asarray(clip["dof_pos"])
    idx = {n: i for i, n in enumerate(order)}
    J = {n: np.degrees(q[:, i]) for n, i in idx.items()}
    rng = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))
    lsp, rsp = J["left_shoulder_pitch_joint"], J["right_shoulder_pitch_joint"]
    lhp, rhp = J["left_hip_pitch_joint"], J["right_hip_pitch_joint"]
    return {
        "clip": pkl_path.stem,
        "T": len(q),
        "shoP_L": (float(lsp.mean()), rng(lsp)),
        "shoP_R": (float(rsp.mean()), rng(rsp)),
        "antiLR": corr(lsp, rsp),
        "xL_hipR": corr(lsp, rhp),       # left arm vs RIGHT leg (natural sync)
        "xR_hipL": corr(rsp, lhp),
        "elbP_L": (float(J["left_elbow_pitch_joint"].mean()), rng(J["left_elbow_pitch_joint"])),
        "elbP_R": (float(J["right_elbow_pitch_joint"].mean()), rng(J["right_elbow_pitch_joint"])),
        "elbY_L": (float(J["left_elbow_yaw_joint"].mean()), rng(J["left_elbow_yaw_joint"])),
        "elbY_R": (float(J["right_elbow_yaw_joint"].mean()), rng(J["right_elbow_yaw_joint"])),
        "shoR_L": (float(J["left_shoulder_roll_joint"].mean()), rng(J["left_shoulder_roll_joint"])),
        "shoR_R": (float(J["right_shoulder_roll_joint"].mean()), rng(J["right_shoulder_roll_joint"])),
        "lumY": rng(J["lumbar_yaw_joint"]),
        "lumY_xLsho": corr(J["lumbar_yaw_joint"], lsp),
        "hipP_L": rng(lhp), "hipP_R": rng(rhp),
    }


def report(tag, rows):
    print(f"\n===== {tag} =====")
    hdr = (f"{'clip':22s} {'T':>4s} | {'shoP_L mean/swing':>18s} {'shoP_R':>15s} "
           f"| {'antiL/R':>8s} {'xL-hipR':>8s} | {'elbP_L':>12s} {'elbP_R':>12s} "
           f"| {'elbY_L':>12s} {'elbY_R':>12s} | {'lumYsw':>7s} {'hipsw':>6s}")
    print(hdr)
    for r in rows:
        print(f"{r['clip']:22s} {r['T']:4d} | "
              f"{r['shoP_L'][0]:6.1f} / {r['shoP_L'][1]:5.1f}   "
              f"{r['shoP_R'][0]:6.1f} / {r['shoP_R'][1]:5.1f} | "
              f"{r['antiLR']:+.2f}  {r['xL_hipR']:+.2f} | "
              f"{r['elbP_L'][0]:5.1f}/{r['elbP_L'][1]:4.1f} {r['elbP_R'][0]:5.1f}/{r['elbP_R'][1]:4.1f} | "
              f"{r['elbY_L'][0]:5.1f}/{r['elbY_L'][1]:4.1f} {r['elbY_R'][0]:5.1f}/{r['elbY_R'][1]:4.1f} | "
              f"{r['lumY']:5.1f} {r['hipP_L']:5.1f}")


if __name__ == "__main__":
    for tag, d, order in [("x1_gmr (GMR auto-IK output)", GMR_DIR, GMR_ORDER),
                          ("x1_lab (training reference)", LAB_DIR, LAB_ORDER)]:
        rows = []
        for c in CLIPS:
            p = d / f"{c}.pkl"
            if p.exists():
                rows.append(analyze(p, order))
        report(tag, rows)
    print("""
Legend (deg):
  shoP_L mean/swing = shoulder pitch mean / swing range (p95-p5)
  antiL/R  = corr(L sho pitch, R sho pitch); natural arm swing -> NEGATIVE (antiphase)
  xL-hipR  = corr(L shoulder pitch, R hip pitch); natural walking -> POSITIVE (arm syncs w/ opposite leg)
  elbP/elbY = elbow pitch / elbow yaw mean and range (bend -> forearm raised)
  lumYsw = lumbar yaw swing range; hipsw = hip pitch swing (gait rhythm scale)
""")
