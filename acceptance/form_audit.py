#!/usr/bin/env python3
"""v80 arc FORM audit — foot & arm posture, convention-calibrated.

Frame conventions (calibrated 2026-10-26 against gmr_x1_assets/x1.xml with
the sole-corner geometry of m5500_walk10.npz):
  - base (free-joint) frame = Isaac base frame: +x FORWARD, +z UP (tilt code
    in mujoco_rollout.py uses R[2,2]; commands are body-frame vx).
  - ankle_roll_link body: local +z = TOE/forward, local +y = SOLE NORMAL
    (up away from ground), local +x = lateral.

Metrics per foot:
  F1 sole_tilt   = angle(sole_normal, world-up)      [edge-walking]
  F2 tip_signed  = atan2(n.lat, n.up), lat = h x up = body-right
                   (+ = normal tips body-right; per-foot outward = -lat for
                    L, +lat for R)
  F3 toe_rel     = signed angle heading -> toe (projected); toe-OUT reported
                   per foot (outward positive)
  F4 lift mech   = hip_pitch p2p vs hip_roll p2p
Arm: shoulder pitch/roll/yaw + elbow stats (joint space).

Inputs: rollout npz (q=29 xml-order joints, base_quat wxyz, contact_dist,
sole_xyz) or demo pkl (dof_pos LAB_DOF_NAMES, root_pos/root_rot).
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
MODEL = str(HERE.parent / "gmr_x1_assets" / "x1.xml")
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
UP = np.array([0.0, 0.0, 1.0])


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.ascontiguousarray(q))
    return m.reshape(3, 3)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def audit(path):
    p = Path(path)
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n != "floating_base"]
    jadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in hinge}
    bids = {"L": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
            "R": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")}

    if p.suffix == ".pkl":
        d = pickle.load(open(p, "rb"))
        T = d["dof_pos"].shape[0]
        idxs = list(range(0, T, max(1, T // 500)))
        base_pos, base_rot = d["root_pos"], d["root_rot"]
        dof = d["dof_pos"]
        col = {n: LAB_DOF_NAMES.index(n) for n in LAB_DOF_NAMES}
        quat_wxyz = bool(abs(base_rot[0, 0]) > 0.9)
        cd = None
        ankle_z_src = None
    else:
        d = np.load(p)
        T = d["q"].shape[0]
        idxs = list(range(T))
        base_pos, base_rot = d["base_pos"], d["base_quat"]
        dof = d["q"]
        col = {n: i for i, n in enumerate(hinge)}
        quat_wxyz = True
        cd = d["contact_dist"] if "contact_dist" in d.files else None
        ankle_z_src = d["sole_xyz"][:, :, :, 2].min(axis=2) if "sole_xyz" in d.files else None

    F = {s: dict(tilt=[], tip=[], toe=[]) for s in "LR"}
    sep_ankle, lat_hist = [], []
    if p.suffix != ".pkl" and "sole_xyz" in d.files:
        fc = d["sole_xyz"].mean(axis=2)          # (T,2,3) sole-corner centers
    # per-foot local axes (calibrated 2026-10-26): L sole-normal=+y; the R
    # frame is rotated 180 deg about z => sole-normal=-y, toe stays +z.
    N_LOCAL = {"L": np.array([0, 1.0, 0]), "R": np.array([0, -1.0, 0])}
    T_LOCAL = {"L": np.array([0, 0, 1.0]), "R": np.array([0, 0, 1.0])}
    J = {n: [] for n in LAB_DOF_NAMES if "hip" in n or "ankle" in n
         or "shoulder" in n or "elbow_pitch" in n}

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
        Rb = quat2mat(data.qpos[3:7])
        h = Rb @ np.array([1.0, 0, 0])  # base +x = forward
        h[2] = 0
        h = unit(h)
        lat = unit(np.cross(h, UP))  # body-right
        pa = {s: data.xpos[bids[s]].copy() for s in "LR"}
        sep_ankle.append(abs(float((pa["L"] - pa["R"]) @ lat)))
        lat_hist.append(lat.copy())
        for s in "LR":
            Rf = data.xmat[bids[s]].reshape(3, 3).copy()
            n_vec = unit(Rf @ N_LOCAL[s])   # sole normal (per-foot calibrated)
            t_vec = unit(Rf @ T_LOCAL[s])   # toe
            t_vec[2] = 0
            t_vec = unit(t_vec)
            F[s]["tilt"].append(math.degrees(math.acos(max(-1.0, min(1.0, float(n_vec @ UP))))))
            F[s]["tip"].append(math.degrees(math.atan2(float(n_vec @ lat), float(n_vec @ UP))))
            F[s]["toe"].append(math.degrees(math.atan2(float(t_vec @ lat), float(t_vec @ h))))
        for n in J:
            J[n].append(math.degrees(dof[t, col[n]]))

    out = {"log": str(p), "frames": len(idxs), "metrics": {}}
    M = out["metrics"]
    for s in "LR":
        arr = {k: np.array(v) for k, v in F[s].items()}
        if ankle_z_src is not None:
            # rollout: sole-corner heights (Schmitt 'on' level 8 mm)
            z = np.array([ankle_z_src[t, 0 if s == "L" else 1] for t in idxs])
            st = z < 0.008
        else:
            # demo: ankle body sits ~7-9 cm above the sole in stance; use the
            # low-percentile + 2 cm band (per-clip adaptive)
            z = demo_ankle_z(model, data, p, idxs, bids[s], col, jadr, quat_wxyz)
            st = z < (np.percentile(z, 30) + 0.02)
        sw = ~st
        out_sign = -1.0 if s == "L" else 1.0   # toe_rel>0 = toe right of heading
        M[f"{s}_stance_n"] = int(st.sum())
        M[f"{s}_sole_tilt_med"] = float(np.median(arr["tilt"][st]))
        M[f"{s}_sole_tilt_p95"] = float(np.percentile(arr["tilt"][st], 95))
        M[f"{s}_tip_med"] = float(np.median(arr["tip"][st]))
        M[f"{s}_toe_out_med"] = out_sign * float(np.median(arr["toe"][st]))
        M[f"{s}_toe_out_p95"] = out_sign * float(np.percentile(arr["toe"][st], 95))
        M[f"{s}_swing_tilt_p95"] = float(np.percentile(arr["tilt"][sw], 95)) if sw.any() else float("nan")
        side = "left" if s == "L" else "right"
        for jn in ("hip_pitch", "hip_roll", "hip_yaw", "ankle_roll", "ankle_pitch",
                   "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch"):
            j = np.array(J[f"{side}_{jn}_joint"])
            M[f"{s}_{jn}_mean"] = float(np.mean(j))
            M[f"{s}_{jn}_p2p"] = float(np.ptp(j))
            M[f"{s}_{jn}_ap95"] = float(np.percentile(np.abs(j), 95))
        M[f"{s}_lift_ratio"] = M[f"{s}_hip_pitch_p2p"] / max(M[f"{s}_hip_roll_p2p"], 1e-6)
    # v83 F5: stance separation (walking-heading lateral distance between
    # the feet). npz rollouts: sole-corner centers (authoritative stat);
    # pkls: ankle origins (report-only). Baseline v81-m2500 walk10
    # 447/527 mm FAIL vs shoulder width 292 / demo 35_xx 281-318.
    _sa = np.array(sep_ankle)
    M["sep_ankle_p50"] = float(np.percentile(_sa, 50))
    M["sep_ankle_p90"] = float(np.percentile(_sa, 90))
    if p.suffix != ".pkl" and "sole_xyz" in d.files:
        fc = d["sole_xyz"].mean(axis=2)[idxs]   # (N,2,3)
        _lat = np.array(lat_hist)               # (N,3) walking lateral axes
        _rel = fc[:, 0] - fc[:, 1]              # (N,3) R<-L world offset
        _proj = np.abs(np.einsum("ij,ij->i", _rel, _lat))
        M["sep_p50"] = float(np.percentile(_proj, 50))
        M["sep_p90"] = float(np.percentile(_proj, 90))
    return out


def demo_ankle_z(model, data, p, idxs, bid, col, jadr, quat_wxyz):
    """ankle body world z per sampled frame (demo path)."""
    d = pickle.load(open(p, "rb"))
    base_pos, base_rot, dof = d["root_pos"], d["root_rot"], d["dof_pos"]
    zs = []
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
        zs.append(float(data.xpos[bid, 2]))
    return np.array(zs)




# ---------------------------------------------------------------------------
# v80 BINDING FORM GATES (pre-registered 2026-10-26, RECORD_BOOK §10).
# Calibrated on: demo library x1_lab_v72 (healthy) vs v76-m5500 (defective).
#   F1 stance sole-tilt  med <= 10, p95 <= 30   (m5500: 9-14/65-67 FAIL)
#   F2 toe-out           med <= 15, p95 <= 35   (m5500: 54-57/72  FAIL; demos <=20/41)
#   F3 hip_yaw DC        |mean| <= 12, p95 <= 30 (m5500: 31/64-66 FAIL; demos <=18/24)
#   F4 hip_roll outward  |mean| <= 10           (m5500: 22-24 FAIL; demos <=9)
#   A1 arm DC            shoP <= 25, shoR <= 20, shoY <= 15, elbow p95 <= 70
#   F5 stance separation p50 <= 0.33 m, p90 <= 0.38 m (v83 2026-10-27,
#     user defect 劈腿走: v81-m2500 447/527 FAIL; shoulder width 0.292,
#     hip 0.268, demo 35_xx 0.281-0.318 / p90 0.338-0.395)
# -------------------------------------------------------form-gates---------
GATE = {
    "sole_tilt_med": 10.0, "sole_tilt_p95": 30.0,
    "toe_out_med": 15.0, "toe_out_p95": 35.0,
    "hip_yaw_mean": 12.0, "hip_yaw_p95": 30.0,
    "hip_roll_mean": 10.0,
    "shoP_dc": 25.0, "shoR_dc": 20.0, "shoY_dc": 15.0, "elbow_p95": 70.0,
    "sep_p50": 0.33, "sep_p90": 0.38,
}


def form_gate(report, side="both"):
    """BINDING form verdict for one rollout report (walk scenarios)."""
    m = report["metrics"]
    fails, checks = [], {}
    for s in ("L", "R") if side == "both" else (side,):
        for key, thr, val in (
            (f"{s}_sole_tilt_med", GATE["sole_tilt_med"], m[f"{s}_sole_tilt_med"]),
            (f"{s}_sole_tilt_p95", GATE["sole_tilt_p95"], m[f"{s}_sole_tilt_p95"]),
            (f"{s}_toe_out_med", GATE["toe_out_med"], abs(m[f"{s}_toe_out_med"])),
            (f"{s}_toe_out_p95", GATE["toe_out_p95"], abs(m[f"{s}_toe_out_p95"])),
            (f"{s}_hip_yaw_mean", GATE["hip_yaw_mean"], abs(m[f"{s}_hip_yaw_mean"])),
            (f"{s}_hip_yaw_ap95", GATE["hip_yaw_p95"], m[f"{s}_hip_yaw_ap95"]),
            (f"{s}_hip_roll_mean", GATE["hip_roll_mean"], abs(m[f"{s}_hip_roll_mean"])),
            (f"{s}_shoulder_pitch_mean", GATE["shoP_dc"], abs(m[f"{s}_shoulder_pitch_mean"])),
            (f"{s}_shoulder_roll_mean", GATE["shoR_dc"], abs(m[f"{s}_shoulder_roll_mean"])),
            (f"{s}_shoulder_yaw_mean", GATE["shoY_dc"], abs(m[f"{s}_shoulder_yaw_mean"])),
            (f"{s}_elbow_pitch_ap95", GATE["elbow_p95"], m[f"{s}_elbow_pitch_ap95"]),
        ):
            ok = val <= thr
            checks[key] = {"val": round(float(val), 2), "thr": thr, "pass": bool(ok)}
            if not ok:
                fails.append(key)
    if "sep_p50" in m:  # npz only (authoritative sole-corner statistic)
        for key, thr, val in (
            ("sep_p50", GATE["sep_p50"], m["sep_p50"]),
            ("sep_p90", GATE["sep_p90"], m["sep_p90"]),
        ):
            ok = val <= thr
            checks[key] = {"val": round(float(val), 4), "thr": thr, "pass": bool(ok)}
            if not ok:
                fails.append(key)
    return {"verdict": "PASS" if not fails else "FAIL", "fails": fails,
            "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--gate", action="store_true",
                    help="emit the pre-registered v80 FORM gate verdict (npz only)")
    args = ap.parse_args()
    reps = []
    for lp in args.logs:
        r = audit(lp)
        reps.append(r)
        m = r["metrics"]
        print(f"\n=== {Path(r['log']).name} ({r['frames']}f) ===")
        for s in "LR":
            print(f"[{s}] STANCE n={m[f'{s}_stance_n']:4d} sole-tilt med/p95 "
                  f"{m[f'{s}_sole_tilt_med']:5.1f}/{m[f'{s}_sole_tilt_p95']:5.1f} deg"
                  f" | tip(+ =right) {m[f'{s}_tip_med']:+6.1f}"
                  f" | TOE-OUT med/p95 {m[f'{s}_toe_out_med']:+6.1f}/{m[f'{s}_toe_out_p95']:+6.1f}"
                  f" | swing tilt p95 {m[f'{s}_swing_tilt_p95']:5.1f}")
            print(f"[{s}] hipP p2p {m[f'{s}_hip_pitch_p2p']:5.1f} vs hipR p2p {m[f'{s}_hip_roll_p2p']:5.1f}"
                  f" (x{m[f'{s}_lift_ratio']:.2f}) | hip_roll {m[f'{s}_hip_roll_mean']:+6.1f}"
                  f" | hip_yaw {m[f'{s}_hip_yaw_mean']:+6.1f} p95 {m[f'{s}_hip_yaw_ap95']:5.1f}"
                  f" | ank_roll {m[f'{s}_ankle_roll_mean']:+5.1f} p95 {m[f'{s}_ankle_roll_ap95']:5.1f}"
                  f" | ankP p2p {m[f'{s}_ankle_pitch_p2p']:5.1f}")
            print(f"[{s}] ARM shoP {m[f'{s}_shoulder_pitch_mean']:+6.1f} p2p {m[f'{s}_shoulder_pitch_p2p']:5.1f}"
                  f" | shoR {m[f'{s}_shoulder_roll_mean']:+6.1f} p95 {m[f'{s}_shoulder_roll_ap95']:5.1f}"
                  f" | shoY {m[f'{s}_shoulder_yaw_mean']:+6.1f}"
                  f" | elbow {m[f'{s}_elbow_pitch_mean']:+5.1f} p95 {m[f'{s}_elbow_pitch_ap95']:5.1f}")
        if "sep_p50" in m:
            print(f"[SEP] stance separation p50/p90 {m['sep_p50']*1000:.0f}/{m['sep_p90']*1000:.0f} mm"
                  f" (ankle-basis {m['sep_ankle_p50']*1000:.0f}/{m['sep_ankle_p90']*1000:.0f})")
    if args.gate:
        for r in reps:
            if r["log"].endswith(".npz"):
                g = form_gate(r)
                print(f"FORM-GATE {Path(r['log']).name}: {g['verdict']}"
                      + ("" if g["verdict"] == "PASS" else " fails=" + ",".join(g["fails"])))
                r["form_gate"] = g
    if args.json:
        Path(args.json).write_text(json.dumps(reps, indent=1))


if __name__ == "__main__":
    main()
