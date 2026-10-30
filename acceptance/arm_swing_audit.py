#!/usr/bin/env python3
"""Arm-swing audit (2026-10-30, user defect: 手臂摆动别扭).

Same-caliber arm statistics for BOTH the demo library (x1_lab_v72 pkls,
LAB_DOF_NAMES order) and policy rollouts (npz, hinge/asset order), so the
two can be diffed directly.

Per input:
  JOINT space   shoP/shoR/shoY/elbP/elbY/wriP/wriR x {L,R}:
                mean (DC), std, p2p, p05/p95   [deg]
  FK space      (base frame, +x fwd +z up; removes world drift)
    phi_arm     upper-arm hang angle: atan2(u.x, -u.z) on the unit
                shoulder->elbow vector. 0 = straight down, + = arm forward.
    psi_fore    forearm hang angle on the unit elbow->hand vector.
    swing_amp   p2p of phi (per side); corr(phiL, phiR) and frac_opp
                (natural gait: L/R antiphase => corr ~ -1, frac_opp ~ 1)
    arm-leg     corr(phi_arm, contra hip_pitch) & ipsi hip_pitch
                (natural: contra +, ipsi -)
    hand_offs   hand pos in base frame: forward (x), lateral (y, + = right
                of body), height z; DC + p2p. Lateral DC ~ arm abduction
                carriage; large lateral p2p = arms flailing sideways.
    hand_jerk   median |d2hand/dt2| (base frame) in g — tremor/smoothness.
"""
import argparse
import functools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import mujoco

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = str(ROOT / "gmr_x1_assets" / "x1.xml")

LAB_DOF_NAMES = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
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
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
              "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.ascontiguousarray(q))
    return m.reshape(3, 3)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def audit(path):
    p = Path(path)
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n != "floating_base"]
    jadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in hinge}
    B = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    bid = {"shoL": B("left_shoulder_pitch_link"), "shoR": B("right_shoulder_pitch_link"),
           "elbL": B("left_elbow_pitch_link"), "elbR": B("right_elbow_pitch_link"),
           "handL": B("left_wrist_roll_link"), "handR": B("right_wrist_roll_link"),
           "base": B("base_link")}

    if p.suffix == ".pkl":
        d = pickle.load(open(p, "rb"))
        T = d["dof_pos"].shape[0]
        idxs = list(range(T))
        base_pos, base_rot = d["root_pos"], d["root_rot"]
        dof = d["dof_pos"]
        col = {n: LAB_DOF_NAMES.index(n) for n in LAB_DOF_NAMES}
        quat_wxyz = bool(abs(base_rot[0, 0]) > 0.9)
        dt = 1.0 / float(d.get("fps", 120.0))
    else:
        d = np.load(p)
        T = d["q"].shape[0]
        idxs = list(range(T))
        base_pos, base_rot = d["base_pos"], d["base_quat"]
        dof = d["q"]
        col = {n: i for i, n in enumerate(hinge)}
        quat_wxyz = True
        dt = 0.02

    J = {f"{s}_{jn}": [] for s in "LR" for jn in ARM_JOINTS}
    phi = {"L": [], "R": []}
    psi = {"L": [], "R": []}
    hand_b = {"L": [], "R": []}          # hand pos in base frame
    hip = {"L": [], "R": []}

    for t in idxs:
        data.qpos[:] = 0.0
        data.qpos[0:3] = base_pos[t]
        br = base_rot[t]
        if not quat_wxyz:
            br = np.array([br[3], br[0], br[1], br[2]])
        data.qpos[3:7] = br
        for n, i in col.items():
            data.qpos[jadr[n]] = dof[t, i]
        mujoco.mj_forward(model, data)
        Rb = quat2mat(data.qpos[3:7])     # base->world
        RbT = Rb.T
        for s in "LR":
            e = data.xpos[bid[f"elb{s}"]]
            sh = data.xpos[bid[f"sho{s}"]]
            ha = data.xpos[bid[f"hand{s}"]]
            u = RbT @ (e - sh); f = RbT @ (ha - e)
            nrm = lambda v: v / max(np.linalg.norm(v), 1e-9)
            u, f = nrm(u), nrm(f)
            phi[s].append(math.degrees(math.atan2(u[0], -u[2])))
            psi[s].append(math.degrees(math.atan2(f[0], -f[2])))
            hand_b[s].append(RbT @ (ha - data.xpos[bid["base"]]))
            for jn in ARM_JOINTS:
                J[f"{s}_{jn}"].append(math.degrees(dof[t, col[f"{'left' if s=='L' else 'right'}_{jn}_joint"]]))
            hip[s].append(math.degrees(dof[t, col[f"{'left' if s=='L' else 'right'}_hip_pitch_joint"]]))

    out = {"log": str(p), "frames": len(idxs), "metrics": {}}
    M = out["metrics"]
    for s in "LR":
        for jn in ARM_JOINTS:
            j = np.array(J[f"{s}_{jn}"])
            M[f"{s}_{jn}_mean"] = float(np.mean(j))
            M[f"{s}_{jn}_std"] = float(np.std(j))
            M[f"{s}_{jn}_p2p"] = float(np.ptp(j))
            M[f"{s}_{jn}_p05"] = float(np.percentile(j, 5))
            M[f"{s}_{jn}_p95"] = float(np.percentile(j, 95))
        ph = np.array(phi[s]); ps = np.array(psi[s])
        M[f"{s}_phi_mean"] = float(np.mean(ph)); M[f"{s}_phi_p2p"] = float(np.ptp(ph))
        M[f"{s}_phi_std"] = float(np.std(ph))
        M[f"{s}_psi_mean"] = float(np.mean(ps)); M[f"{s}_psi_p2p"] = float(np.ptp(ps))
        hb = np.array(hand_b[s])                      # (T,3) base frame
        M[f"{s}_hand_fwd_mean"] = float(np.mean(hb[:, 0])); M[f"{s}_hand_fwd_p2p"] = float(np.ptp(hb[:, 0]))
        M[f"{s}_hand_lat_mean"] = float(np.mean(hb[:, 1])); M[f"{s}_hand_lat_p2p"] = float(np.ptp(hb[:, 1]))
        M[f"{s}_hand_z_mean"] = float(np.mean(hb[:, 2])); M[f"{s}_hand_z_p2p"] = float(np.ptp(hb[:, 2]))
        if len(hb) > 4:
            acc = np.diff(hb, 2, axis=0) / dt**2      # base-frame accel (g)
            M[f"{s}_hand_jerk_med_g"] = float(np.median(np.linalg.norm(acc, axis=1)) / 9.81)
    phL, phR = np.array(phi["L"]), np.array(phi["R"])
    M["corr_phiLR"] = corr(phL, phR)
    M["frac_opp"] = float(np.mean(phL * phR < 0))
    M["amp_asym"] = float(abs(np.ptp(phL) - np.ptp(phR)))
    M["dc_asym"] = float(abs(np.mean(phL) - np.mean(phR)))
    for s in "LR":
        c = "R" if s == "L" else "L"
        M[f"corr_phi{s}_contraHip"] = corr(np.array(phi[s]), np.array(hip[c]))
        M[f"corr_phi{s}_ipsiHip"] = corr(np.array(phi[s]), np.array(hip[s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep = [audit(p) for p in args.paths]
    if args.json:
        json.dump(rep, open(args.json, "w"), indent=1)
    for r in rep:
        M = r["metrics"]
        print(f"\n=== {r['log']} ({r['frames']} fr)")
        print("  joint | mean | std | p2p | p05-p95   [deg]")
        for s in "LR":
            seg = " ".join(
                f"{jn.replace('_pitch','P').replace('_roll','R').replace('_yaw','Y').replace('shoulder_','sho').replace('elbow_','elb').replace('wrist_','wri')}:{M[f'{s}_{jn}_mean']:+6.1f}/{M[f'{s}_{jn}_std']:4.1f}/{M[f'{s}_{jn}_p2p']:5.1f}"
                for jn in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch"))
            print(f"  {s} " + seg)
            print(f"    elbY {M[f'{s}_elbow_yaw_mean']:+.1f}/{M[f'{s}_elbow_yaw_std']:.1f}/{M[f'{s}_elbow_yaw_p2p']:.1f}"
                  f"  wriP {M[f'{s}_wrist_pitch_mean']:+.1f}/{M[f'{s}_wrist_pitch_std']:.1f}/{M[f'{s}_wrist_pitch_p2p']:.1f}"
                  f"  wriR {M[f'{s}_wrist_roll_mean']:+.1f}/{M[f'{s}_wrist_roll_std']:.1f}/{M[f'{s}_wrist_roll_p2p']:.1f}")
        print(f"  FK phi (upper-arm fwd angle): L {M['L_phi_mean']:+.1f} p2p {M['L_phi_p2p']:.1f} | "
              f"R {M['R_phi_mean']:+.1f} p2p {M['R_phi_p2p']:.1f} | corr {M['corr_phiLR']:+.2f} frac_opp {M['frac_opp']:.2f}")
        print(f"  FK psi (forearm): L {M['L_psi_mean']:+.1f} p2p {M['L_psi_p2p']:.1f} | R {M['R_psi_mean']:+.1f} p2p {M['R_psi_p2p']:.1f}")
        print(f"  hand(base) fwd L {M['L_hand_fwd_mean']:+.3f} p2p {M['L_hand_fwd_p2p']:.3f} | R {M['R_hand_fwd_mean']:+.3f} p2p {M['R_hand_fwd_p2p']:.3f}  [m]")
        print(f"  hand(base) lat L {M['L_hand_lat_mean']:+.3f} p2p {M['L_hand_lat_p2p']:.3f} | R {M['R_hand_lat_mean']:+.3f} p2p {M['R_hand_lat_p2p']:.3f}  [m]")
        print(f"  hand(base) z   L {M['L_hand_z_mean']:+.3f} p2p {M['L_hand_z_p2p']:.3f} | R {M['R_hand_z_mean']:+.3f} p2p {M['R_hand_z_p2p']:.3f}  [m]")
        if "L_hand_jerk_med_g" in M:
            print(f"  hand jerk med: L {M['L_hand_jerk_med_g']:.2f}g | R {M['R_hand_jerk_med_g']:.2f}g")
        print(f"  arm-leg: corr(phiL,hipR) {M['corr_phiL_contraHip']:+.2f} corr(phiL,hipL) {M['corr_phiL_ipsiHip']:+.2f} "
              f"corr(phiR,hipL) {M['corr_phiR_contraHip']:+.2f} corr(phiR,hipR) {M['corr_phiR_ipsiHip']:+.2f}")


if __name__ == "__main__":
    main()
