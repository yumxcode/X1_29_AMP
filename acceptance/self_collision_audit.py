#!/usr/bin/env python3
"""Self-collision (mesh-penetration, "穿模") audit for X1 rollouts & references.

User goal (human-gait arc): arms / legs must not interpenetrate while
walking. The sim2sim MJCF disables all collisions except foot spheres +
floor, and the Isaac side trains with enabled_self_collisions=True +
undesired_contacts (-10) — but nothing MEASURES penetration on the rollout
artifacts. This tool replays a logged trajectory (rollout npz or reference
pkl) through MuJoCo FK and measures analytic CLEARANCE between capsule
proxies of the limb chains:

  arms   : shoulder_pitch -> elbow_pitch -> wrist_roll (upper + forearm)
  torso  : base_link -> lumbar_pitch_link (+ the big chest sphere)
  legs   : hip_pitch -> knee_pitch (thigh), knee -> ankle_roll (shank)

Pairs audited (non-adjacent by construction):
  L-arm vs torso / R-arm vs torso / L-arm vs R-arm /
  arm vs opposite-side leg (hand passes thigh at swing-through) /
  L-leg vs R-leg (crossing)

Clearance = segment-segment distance - (r_a + r_b); negative = penetration
depth in meters. Capsule radii come from the vendor xml sphere geoms
(max radius of the two endpoint spheres).

Usage:
  # policy rollout npz (any --log from mujoco_rollout.py)
  python acceptance/self_collision_audit.py a.npz b.npz [--json out.json]
  # reference pkl (retargeted clips) for threshold calibration
  python acceptance/self_collision_audit.py --pkl dir/*.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, ".")
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

ROOT = Path(".").resolve()
XML = ROOT / "gmr_x1_assets" / "x1.xml"

model, _ = build_model(XML)
BID = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): n
       for n in [model.body(i).name for i in range(model.nbody)]}
NAME2ID = {v: k for k, v in BID.items()}

# sphere radius per body (max over its geoms — vendor visual spheres)
RAD = {}
for i in range(model.ngeom):
    b = model.geom_bodyid[i]
    nm = BID.get(b)
    r = model.geom_size[i][0]
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
        RAD[nm] = max(RAD.get(nm, 0.0), float(r))

CHAINS = {
    "LARM": ["left_shoulder_pitch_link", "left_elbow_pitch_link", "left_wrist_roll_link"],
    "RARM": ["right_shoulder_pitch_link", "right_elbow_pitch_link", "right_wrist_roll_link"],
    "TORSO": ["base_link", "lumbar_pitch_link"],
    "LTHIGH": ["left_hip_pitch_link", "left_knee_pitch_link"],
    "LSHANK": ["left_knee_pitch_link", "left_ankle_roll_link"],
    "RTHIGH": ["right_hip_pitch_link", "right_knee_pitch_link"],
    "RSHANK": ["right_knee_pitch_link", "right_ankle_roll_link"],
}
PAIRS = [
    ("LARM", "TORSO"), ("RARM", "TORSO"), ("LARM", "RARM"),
    ("LARM", "RTHIGH"), ("LARM", "RSHANK"),
    ("RARM", "LTHIGH"), ("RARM", "LSHANK"),
    ("LTHIGH", "RTHIGH"), ("LSHANK", "RSHANK"),
    ("LTHIGH", "RSHANK"), ("RTHIGH", "LSHANK"),
]


def chain_capsules(chain, xpos, xmat_unused):
    """-> list of (p0, p1, r) capsules between consecutive body origins."""
    pts = [xpos[NAME2ID[b]] for b in chain]
    out = []
    for i in range(len(pts) - 1):
        r = max(RAD.get(chain[i], 0.0), RAD.get(chain[i + 1], 0.0))
        out.append((pts[i].copy(), pts[i + 1].copy(), r))
    # degenerate 1-point chains -> zero-length capsule
    if len(pts) == 1:
        out.append((pts[0].copy(), pts[0].copy(), RAD.get(chain[0], 0.0)))
    return out


def seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between two 3D segments (Ericson, Real-Time
    Collision Detection)."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e = float(d1 @ d1), float(d2 @ d2)
    f = float(d2 @ r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - p2))
    if a <= EPS:
        s = 0.0
        t = np.clip(f / e, 0, 1)
    else:
        c = float(d1 @ r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-c / a, 0, 1)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0:
                t = 0.0
                s = np.clip(-c / a, 0, 1)
            elif t > 1:
                t = 1.0
                s = np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def capsule_clearance(caps_a, caps_b):
    best = np.inf
    for p1, q1, r1 in caps_a:
        for p2, q2, r2 in caps_b:
            best = min(best, seg_seg_dist(p1, q1, p2, q2) - r1 - r2)
    return best


data = mujoco.MjData(model)


def fk_step(base_pos, base_quat, q_hinge, hinge, qadr):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = base_pos
    nq = np.linalg.norm(base_quat)
    data.qpos[3:7] = base_quat / (nq if nq > 0 else 1.0)
    data.qpos[qadr] = q_hinge
    mujoco.mj_forward(model, data)
    return data.xpos.copy()


def audit_trajectory(base_pos, base_quat, q_hinge_rows, hinge, qadr, stride=2):
    """Returns {pair_name: min_clearance_m} over the trajectory."""
    worst = {f"{a}|{b}": np.inf for a, b in PAIRS}
    for t in range(0, len(q_hinge_rows), stride):
        xpos = fk_step(base_pos[t], base_quat[t], q_hinge_rows[t], hinge, qadr)
        caps = {nm: chain_capsules(ch, xpos, None) for nm, ch in CHAINS.items()}
        for a, b in PAIRS:
            worst[f"{a}|{b}"] = min(worst[f"{a}|{b}"], capsule_clearance(caps[a], caps[b]))
    return {k: (v if np.isfinite(v) else None) for k, v in worst.items()}


def load_npz_traj(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    hinge = list(meta["hinge_names"])
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    settle = int(meta.get("settle_steps", 0))
    return (np.asarray(z["base_pos"])[settle:], np.asarray(z["base_quat"])[settle:],
            np.asarray(z["q"])[settle:], hinge, qadr)


def load_pkl_traj(path):
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
    return bp, br, qh, hinge, qadr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pkl", nargs="*", default=[])
    ap.add_argument("--json", default=None)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    rows = []
    for p in args.files:
        bp, br, qh, hinge, qadr = load_npz_traj(p)
        res = audit_trajectory(bp, br, qh, hinge, qadr, args.stride)
        rows.append({"file": str(p), "kind": "rollout", "min_clearance_m": res})
        print(f"{p}: worst={min(v for v in res.values() if v is not None)*1000:.1f}mm")
        for k, v in res.items():
            if v is not None and v < 0.02:
                print(f"    {k:24s} {v*1000:8.1f} mm")
    for p in args.pkl:
        bp, br, qh, hinge, qadr = load_pkl_traj(p)
        res = audit_trajectory(bp, br, qh, hinge, qadr, max(args.stride, 4))
        rows.append({"file": str(p), "kind": "reference", "min_clearance_m": res})
        print(f"{p} [ref]: worst={min(v for v in res.values() if v is not None)*1000:.1f}mm")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"[JSON] {args.json}")


if __name__ == "__main__":
    main()
