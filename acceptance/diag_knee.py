#!/usr/bin/env python3
"""Straight-knee audit: reference dataset (pkl) vs policy rollouts (npz).

Knee pitch convention: 0 = fully extended, positive = flexion (bend).
DEFAULT_Q knee = 0.632 rad = 36 deg (the X1 crouch default).

For each clip/rollout: split stance/swing using sole contact (pkl via FK
contact_z, npz via contact_dist), then report knee angle at stance
mid-phase, min knee (most extended) during stance, and swing peak.
Human reference: mid-stance knee flexion ~5-15 deg.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, ".")
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list

ROOT = Path(".").resolve()
XML = ROOT / "gmr_x1_assets" / "x1.xml"
DSDIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32"

model, _ = build_model(XML)
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
lab_dof = parse_yaml_list(ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml", "lab_dof_names")
LK, RK = lab_dof.index("left_knee_pitch_joint"), lab_dof.index("right_knee_pitch_joint")

def loff(side):
    y = -0.0408 if side == "L" else 0.0408
    return np.array([[0.03, y, 0.07], [-0.03, y, 0.07],
                     [0.03, y, -0.07], [-0.03, y, -0.07]])

FOOTS = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}
CONTACT_Z = 0.010
data = mujoco.MjData(model)

def stance_ivs_from_sole(sole_z, fps):
    """sole_z: (T,4) world z of 4 spheres. Returns list of (s,e)."""
    on = sole_z.min(axis=1) < CONTACT_Z
    ivs, s = [], None
    T = len(on)
    for t in range(T):
        if on[t] and s is None: s = t
        if (not on[t] or t == T - 1) and s is not None:
            e = t - 1 if not on[t] else t
            if e - s >= int(0.15 * fps): ivs.append((s, e))
            s = None
    return ivs

def pkl_stance_ivs(clip_path):
    c = pickle.load(open(clip_path, "rb"))
    q = np.asarray(c["dof_pos"], dtype=np.float64)
    T = len(q)
    qpos = np.zeros((T, model.nq))
    qpos[:, 0:3] = c["root_pos"]
    rr = np.asarray(c["root_rot"], dtype=np.float64)
    qpos[:, 3:7] = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    qpos[:, qadr[np.argsort(qadr).argsort()]] = 0  # placeholder (not used)
    # map lab cols -> mj qadr (same as diag_heeltoe)
    lab2mj = np.array([hinge.index(n) for n in lab_dof])
    qpos[:, qadr[lab2mj]] = q
    out = {}
    for side, bname in FOOTS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        sz = np.zeros((T, 4))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            R = data.xmat[bid].reshape(3, 3)
            sz[t] = ((R @ loff(side).T).T + data.xpos[bid])[:, 2]
        out[side] = stance_ivs_from_sole(sz, float(c.get("fps", 120.0)))
    return out, q, float(c.get("fps", 120.0))

def knee_stats(knee, ivs):
    """knee: (T,) radians. Returns stance-mid flexion, stance min, swing peak."""
    if not ivs: return None
    mid_f, min_f, sw_f = [], [], []
    on = np.zeros(len(knee), bool)
    for s, e in ivs: on[s:e+1] = True
    for s, e in ivs:
        n = e - s + 1
        mid_f.append(np.degrees(knee[s + n // 4: s + 3 * n // 4].mean()))
        min_f.append(np.degrees(knee[s:e+1].min()))
    swing = ~on
    if swing.any(): sw_f.append(np.degrees(knee[swing].max()))
    return np.mean(mid_f), np.mean(min_f), (np.mean(sw_f) if sw_f else np.nan)

print(f"{'source':30s} {'ft':2s} {'stnMid°':>8s} {'stnMin°':>8s} {'swPeak°':>8s}")
print("--- reference dataset (retargeted pkl) ---")
for cl in sorted(p.stem for p in DSDIR.glob("*.pkl") if "mirror" not in p.stem):
    ivs, q, fps = pkl_stance_ivs(DSDIR / f"{cl}.pkl")
    for side, col in (("L", LK), ("R", RK)):
        st = knee_stats(q[:, col], ivs[side])
        if st: print(f"{cl:30s} {side:2s} {st[0]:8.1f} {st[1]:8.1f} {st[2]:8.1f}")

print("--- policy rollouts (sim2sim npz) ---")
# npz q columns are in MJ HINGE order (data.qpos[qadr]), NOT lab order
LK_MJ, RK_MJ = hinge.index("left_knee_pitch_joint"), hinge.index("right_knee_pitch_joint")
for npz in ["acceptance/v53_eval/v53_walk10.npz", "acceptance/v53_eval/v53_walk05.npz",
            "acceptance/v46_eval/v46s_walk10.npz"]:
    z = np.load(npz)
    q, s = z["q"], z["sole_xyz"]
    for f, side in ((0, "L"), (1, "R")):
        on = s[:, f, :, 2].min(axis=1) < 0.006   # world z of 4 sole spheres
        ivs, st_ = [], None
        T = len(on)
        for t in range(T):
            if on[t] and st_ is None: st_ = t
            if (not on[t] or t == T - 1) and st_ is not None:
                e = t - 1 if not on[t] else t
                if e - st_ >= 8: ivs.append((st_, e))
                st_ = None
        col = LK_MJ if f == 0 else RK_MJ
        st = knee_stats(q[:, col], ivs)
        if st: print(f"{Path(npz).stem:30s} {side:2s} {st[0]:8.1f} {st[1]:8.1f} {st[2]:8.1f}")
