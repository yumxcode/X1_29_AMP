#!/usr/bin/env python3
"""v70 forensics: forearm inward-fold / torso collision audit (2026-10-15).

User defect (soup69_drift video review): 右臂小臂内折，摆动时和躯干碰撞.

This tool replays rollout npz logs (and, optionally, demo pkl clips)
through MuJoCo FK and measures TRUE mesh-level distance between each
forearm's geoms (elbow_pitch/elbow_yaw/wrist/hand links) and the torso
geoms (base_link + lumbar links) via mj_geomDistance — collision-filter
immune (the sim2sim MJCF disables self-collision, which is why the
capsule self_collision_audit.py under-reports: its torso proxy is only
the short base_link->lumbar_pitch segment).

Usage:
  python acceptance/v70_eval/arm_fold_v70.py a.npz b.npz [--pkl x.pkl ...]
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, ".")
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

ROOT = Path(".").resolve()
model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
data = mujoco.MjData(model)

TORSO_BODIES = ("base_link", "lumbar_yaw_link", "lumbar_roll_link", "lumbar_pitch_link")
GEOM_SETS = {"torso": [], "farm_R": [], "farm_L": []}
for i in range(model.ngeom):
    nm = model.body(model.geom_bodyid[i]).name
    if nm in TORSO_BODIES:
        GEOM_SETS["torso"].append(i)
    elif nm.startswith("right_elbow") or nm.startswith("right_wrist") or "right_hand" in nm:
        GEOM_SETS["farm_R"].append(i)
    elif nm.startswith("left_elbow") or nm.startswith("left_wrist") or "left_hand" in nm:
        GEOM_SETS["farm_L"].append(i)

SY = {"L": "left_shoulder_yaw_joint", "R": "right_shoulder_yaw_joint"}
EP = {"L": "left_elbow_pitch_joint", "R": "right_elbow_pitch_joint"}


def fk(t, bp, br, q, qadr):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = bp[t]
    nq = np.linalg.norm(br[t])
    data.qpos[3:7] = br[t] / (nq if nq > 0 else 1.0)
    data.qpos[qadr] = q[t]
    mujoco.mj_forward(model, data)
    return data.xpos.copy(), data.xmat.copy()


def audit_npz(path, stride=2):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    hinge = list(meta["hinge_names"])
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    idx = {n: i for i, n in enumerate(hinge)}
    settle = int(meta.get("settle_steps", 0))
    bp, br, q = np.asarray(z["base_pos"]), np.asarray(z["base_quat"]), np.asarray(z["q"])
    mins = {"farm_R": np.inf, "farm_L": np.inf}
    inward_R = []
    syR, syL, epR = [], [], []
    for t in range(settle, len(q), stride):
        xpos, xmat = fk(t, bp, br, q, qadr)
        for side, farm in (("farm_R", GEOM_SETS["farm_R"]), ("farm_L", GEOM_SETS["farm_L"])):
            for gi in farm:
                for gj in GEOM_SETS["torso"]:
                    mins[side] = min(mins[side], mujoco.mj_geomDistance(model, data, gi, gj, 1.0, None))
        e = xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_yaw_link")]
        w = xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")]
        R = xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")].reshape(3, 3)
        v = w - e
        v /= np.linalg.norm(v)
        inward_R.append(float((R.T @ v)[1]))       # base +y (left) = inward for R arm
        syR.append(q[t, idx[SY["R"]]]); syL.append(q[t, idx[SY["L"]]]); epR.append(q[t, idx[EP["R"]]])
    g = 180.0 / math.pi
    return {
        "min_R_mm": round(mins["farm_R"] * 1000, 1),
        "min_L_mm": round(mins["farm_L"] * 1000, 1),
        "sy_R_deg": round(float(np.mean(syR)) * g, 1),
        "sy_L_deg": round(float(np.mean(syL)) * g, 1),
        "elbow_R_deg": round(float(np.mean(epR)) * g, 1),
        "inward_R_mean": round(float(np.mean(inward_R)), 2),
    }


def audit_pkl(path, stride=4):
    lab_dof = parse_yaml_list(
        ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml", "lab_dof_names")
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n in DEFAULT_Q]
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    lab2mj = np.array([hinge.index(n) for n in lab_dof])
    c = pickle.load(open(path, "rb"))
    q = np.asarray(c["dof_pos"], dtype=np.float64)
    bp = np.asarray(c["root_pos"], dtype=np.float64)
    br = np.asarray(c["root_rot"], dtype=np.float64)
    qh = np.zeros((len(q), len(hinge)))
    qh[:, lab2mj] = q
    mins = {"farm_R": np.inf, "farm_L": np.inf}
    for t in range(0, len(qh), stride):
        fk(t, bp, br, qh, qadr)
        for side, farm in (("farm_R", GEOM_SETS["farm_R"]), ("farm_L", GEOM_SETS["farm_L"])):
            for gi in farm:
                for gj in GEOM_SETS["torso"]:
                    mins[side] = min(mins[side], mujoco.mj_geomDistance(model, data, gi, gj, 1.0, None))
    return {"min_R_mm": round(mins["farm_R"] * 1000, 1), "min_L_mm": round(mins["farm_L"] * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pkl", nargs="*", default=[])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = {}
    for p in args.files:
        r = audit_npz(p)
        rows[str(p)] = r
        print(f"{p}: R {r['min_R_mm']:7.1f} mm | L {r['min_L_mm']:7.1f} mm | "
              f"sy {r['sy_L_deg']:+.1f}/{r['sy_R_deg']:+.1f} deg | "
              f"elbowR {r['elbow_R_deg']:.1f} deg | inward_R {r['inward_R_mean']:+.2f}")
    for p in args.pkl:
        r = audit_pkl(p)
        rows[str(p)] = r
        print(f"{p} [ref]: R {r['min_R_mm']:7.1f} mm | L {r['min_L_mm']:7.1f} mm")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"[JSON] {args.json}")


if __name__ == "__main__":
    main()
