#!/usr/bin/env python3
"""F1w v3 library: X1 joint-trajectory transplant + per-frame root grounding.

Root cause chain (fnd_eb7d87a9, exp_42a04333):
- v1 (X1 targets -> F1w IK): grounded but joint-velocity polluted (IK branch jumps).
- v2 (GMR direct): velocity-clean but geometrically wild (splayed/floating, stance 6-8%).
- v3 synthesis: take the X1 lab clips' JOINTS directly (no IK -> no branch jumps by
  construction; jv inherits the X1 envelope), time-warp with the SAME rho as v2
  (validated C'/F' numbers), then ground the root per frame: shift root z so the
  LOWER ankle tracks the sole height (the residual float comes from the F1w-vs-X1
  geometry, ~4.5 cm mean). kb re-derived via F1w FK -> R2' residual is 0 by
  construction.

Absolute-physics gates (new, source-calibrated on the X1 v72 clips with the SAME
code path): stance%/foot >= 40%, swing p95 <= 0.25 m, penetration >= -0.02 m.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10

F1W_URDF = HERE.parent / "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf"
X1_LIB = HERE / "robolab/data/motions/x1_lab_v72"
SOLE_Z = 0.036          # flat-foot ankle-roll origin height (stand 0.5876 - 0.552)
WARPS = {"07_01": 1.0, "103_07": 1.0, "0005_normal_walk1": 1.0, "0007_normal_walk3": 1.0,
         "0008_normal_walk4": 1.0, "0026_circle_walk": 1.0, "35_01": 1.0, "35_06": 1.0,
         "35_09": 1.0, "35_12": 1.0, "35_16": 1.0, "38_01": 1.0}
# v4: 12 clips across families (fnd_c4ad7d1d diversity hypothesis — X1's healthy
# disc ran on a 22-clip mixture; every F1w collapse used a 2-clip library).


def soft_clamp(x, lo, hi, knee=0.02):
    """Continuous saturation: excursions beyond lo+knee/hi-knee are halved.
    Keeps |dx/dt| <= original (Lipschitz halved outside the knee)."""
    y = x.copy()
    for j in range(x.shape[1]):
        l, h = lo[j], hi[j]
        m = x[:, j] < l + knee
        y[m, j] = l + knee + 0.5 * (x[m, j] - l - knee)
        y[m, j] = np.maximum(y[m, j], l)
        m = x[:, j] > h - knee
        y[m, j] = h - knee + 0.5 * (x[m, j] - h + knee)
        y[m, j] = np.minimum(y[m, j], h)
    return y


def time_warp(q, rp, rr, rho):
    """Uniform slow-down by 1/rho: T' = T/rho, fps unchanged, slerp on quat via lerp+norm."""
    T = len(q)
    Tn = int(round(T / rho))
    ts = np.linspace(0, T - 1, Tn)
    i0 = np.floor(ts).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (ts - i0)[:, None]
    q2 = q[i0] * (1 - w) + q[i1] * w
    rp2 = rp[i0] * (1 - w) + rp[i1] * w
    rr2 = rr[i0] * (1 - w) + rr[i1] * w
    rr2 /= np.linalg.norm(rr2, axis=1, keepdims=True)
    return q2, rp2, rr2


def ground_root(q, rp, rr, fk, lab, kb_names):
    """Shift root z per frame so the LOWER ankle sits at SOLE_Z (smoothed)."""
    zlow = np.zeros(len(q))
    for t in range(len(q)):
        Tb = np.eye(4); Tb[:3, :3] = quat_wxyz_to_R(rr[t]); Tb[:3, 3] = rp[t]
        T = fk.fk_all(dict(zip(lab, q[t])), Tb)
        zlow[t] = min(T["left_ankle_roll"][:3, 3][2], T["right_ankle_roll"][:3, 3][2])
    shift = zlow - (SOLE_Z + 0.012)   # +12mm hover target: keeps smoothed penetration > -0.02
    # smooth: median 0.15s-ish window then light gaussian
    k = 9
    med = np.array([np.median(shift[max(0, t - k):t + k + 1]) for t in range(len(shift))])
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 1.8)) ** 2); g /= g.sum()
    pad = np.pad(med, (k, k), mode="edge")
    sm = np.convolve(pad, g, mode="valid")
    rp2 = rp.copy(); rp2[:, 2] -= sm
    return rp2, sm


def fk_all_kb(q, rp, rr, fk, lab, kb_names):
    kb = np.zeros((len(q), len(kb_names), 3), dtype=np.float32)
    zL = np.zeros(len(q)); zR = np.zeros(len(q))
    for t in range(len(q)):
        Tb = np.eye(4); Tb[:3, :3] = quat_wxyz_to_R(rr[t]); Tb[:3, 3] = rp[t]
        T = fk.fk_all(dict(zip(lab, q[t])), Tb)
        for i, b in enumerate(kb_names):
            kb[t, i] = T[b][:3, 3]
        zL[t] = T["left_ankle_roll"][:3, 3][2]; zR[t] = T["right_ankle_roll"][:3, 3][2]
    return kb, zL, zR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="robolab/data/motions/f1w_lab_v4")
    args = ap.parse_args()
    lab = load_lab_order(str(HERE / "robolab/scripts/tools/retarget/config/x1.yaml"))
    fk = UrdfFK(str(F1W_URDF))
    fk_links = {jd["child"] for jd in fk.joints.values()} | {"base_link"}
    kb_names = [b if b in fk_links else b[:-len("_link")] for b in KEY10]
    import xml.etree.ElementTree as ET
    root = ET.parse(str(F1W_URDF)).getroot()
    lo = np.zeros(29); hi = np.zeros(29)
    for j in root.iter("joint"):
        if j.get("name") in lab:
            l = j.find("limit"); i = lab.index(j.get("name"))
            lo[i] = float(l.get("lower")); hi[i] = float(l.get("upper"))

    dst = HERE / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rho in WARPS.items():
        d = pickle.load(open(X1_LIB / f"{name}.pkl", "rb"))
        q = np.asarray(d["dof_pos"], float); rp = np.asarray(d["root_pos"], float)
        rr = np.asarray(d["root_rot"], float); fps = float(d["fps"])
        q, rp, rr = time_warp(q, rp, rr, rho)
        q = soft_clamp(q, lo, hi)
        rp, sm = ground_root(q, rp, rr, fk, lab, kb_names)
        kb, zL, zR = fk_all_kb(q, rp, rr, fk, lab, kb_names)
        out = {"dof_pos": q.astype(np.float32), "root_pos": rp.astype(np.float32),
               "root_rot": rr.astype(np.float32), "key_body_pos": kb,
               "fps": fps, "loop_mode": d.get("loop_mode", 0)}
        # ---- gates ----
        thr = SOLE_Z + 0.012 + 0.015   # hover target + stance window
        stL = float((zL < thr).mean()); stR = float((zR < thr).mean())
        swing = float(np.percentile(np.maximum(zL, zR), 95))
        pen = float(min(zL.min(), zR.min())) - SOLE_Z
        jv = np.abs(np.diff(q, axis=0) * fps)
        jv_max_flat = jv.max(axis=1)
        vxy = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps
        vmed = float(np.median(vxy))
        # knee corr lag-aligned (F'): corr of L/R knee pitch after best lag
        kL = q[:, lab.index("left_knee_pitch_joint")]; kR = q[:, lab.index("right_knee_pitch_joint")]
        best = 0.0
        for lag in range(-60, 61):
            a, b = (kL[max(0, lag):], kR[max(0, -lag):]) if lag >= 0 else (kL[max(0, -lag):], kR[max(0, lag):])
            n = min(len(a), len(b))
            c = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 30 else 0.0
            best = max(best, abs(c))
        g = {
            "G1_stance_pct": {"L": round(100 * stL, 1), "R": round(100 * stR, 1), "gate": ">=40"},
            "G2_swing_p95_m": round(swing, 3), "gate2": "<=0.25",
            "G3_penetration_m": round(pen, 3), "gate3": ">=-0.02",
            "R4_jv_p99": round(float(np.percentile(jv_max_flat, 99)), 2), "gate4": "<=18.5",
            "R4_jv_max": round(float(jv_max_flat.max()), 2), "gate4b": "<=40",
            "C_prime_v_med": round(vmed, 3), "gate5": "0.41-1.0x of X1-side (rho=1 -> ratio 1.0) + cmd band 0.3-1.0",
            "F_prime_knee_corr": round(best, 3), "gate6": ">=0.5",
            "root_shift_m": {"mean": round(float(sm.mean()), 3), "ptp": round(float(np.ptp(sm)), 3),
                              "max_step": round(float(np.abs(np.diff(sm)).max()), 4)},
            "limits_out_frac": 0.0,
        }
        g["PASS"] = bool(stL >= 0.40 and stR >= 0.40 and swing <= 0.25 and pen >= -0.02
                         and g["R4_jv_p99"] <= 18.5 and g["R4_jv_max"] <= 40
                         and 0.3 <= vmed <= 1.0 and best >= 0.5)
        if not g["PASS"]:
            print(f"[DROP] {name}: gates failed — excluded from the library")
            report[name] = g
            continue
        pickle.dump(out, open(dst / f"{name}.pkl", "wb"))
        report[name] = g
        print(name, json.dumps(g))
    json.dump(report, open(dst / "_v3_report.json", "w"), indent=1)
    print("ALL PASS" if all(r["PASS"] for r in report.values()) else "SOME FAIL")


if __name__ == "__main__":
    main()
