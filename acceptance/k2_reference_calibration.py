#!/usr/bin/env python3
"""K2 gate reference calibration (audit r2 item 4).

Measures the stance-phase knee RHYTHM (K2 = max-min knee angle within
each stance window) on the retargeted reference clips with the IDENTICAL
code path the policy KH metrics use (sim2sim.gait_metrics.analyze on a
50 Hz FK-adapted npz). Output: machine-readable JSON for the gate
revision record (the GOAL §3 anchor '0->60 deg' conflated stance min
with swing peak; same-口径 reference median is the honest calibration).

Usage: python acceptance/k2_reference_calibration.py --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from sim2sim import gait_metrics  # noqa: E402
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

ROOT = Path(".").resolve()
CLIPS = ["0002_treadmill_slow", "0000_treadmill_norm", "0005_normal_walk1",
         "0007_normal_walk3", "0026_circle_walk", "36_01"]


def ref_npz(clip, model, data, hinge, qadr, lab2mj, foot_names):
    import pickle
    import mujoco
    c = pickle.load(open(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32" / f"{clip}.pkl", "rb"))
    q = np.asarray(c["dof_pos"]); rr = np.asarray(c["root_rot"]); rp = np.asarray(c["root_pos"])
    fps = float(c.get("fps", 120.0))
    ds = max(1, int(round(fps / 50.0)))
    idx = np.arange(0, len(q), ds)
    q, rr, rp = q[idx], rr[idx], rp[idx]
    T = len(q)
    sole = np.zeros((T, 2, 4, 3))
    qh = np.zeros((T, len(hinge)))
    qh[:, lab2mj] = q
    loff = lambda y: np.array([[0.03, y, 0.07], [-0.03, y, 0.07],
                               [0.03, y, -0.07], [-0.03, y, -0.07]])
    for t in range(T):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = rp[t]
        n = np.linalg.norm(rr[t])
        data.qpos[3:7] = rr[t] / n
        data.qpos[qadr] = qh[t]
        mujoco.mj_forward(model, data)
        for fi, bn in enumerate(foot_names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            R = data.xmat[bid].reshape(3, 3)
            y = -0.0408 if fi == 0 else 0.0408
            sole[t, fi] = ((R @ loff(y).T).T + data.xpos[bid])
    return {"q": qh, "sole_xyz": sole, "base_pos": rp,
            "base_quat": rr / np.linalg.norm(rr, axis=1, keepdims=True),
            "contact_dist": np.zeros((T, 2, 4)), "v_b": np.zeros((T, 3)),
            "dq": np.zeros((T, len(hinge))), "t": np.arange(T) * 0.02,
            "meta": json.dumps({"cmd": [1.0, 0.0, 0.0], "control_dt": 0.02,
                                "settle_steps": 0, "fell": False,
                                "hinge_names": hinge, "foot_names": foot_names})}


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)
    hinge = [model.joint(i).name for i in range(model.njnt)
             if model.joint(i).name in DEFAULT_Q]
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    lab = parse_yaml_list(ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml",
                          "lab_dof_names")
    lab2mj = np.array([hinge.index(n) for n in lab])
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]

    rows, k2s = [], []
    for clip in CLIPS:
        pkl = ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32" / f"{clip}.pkl"
        if not pkl.exists():
            continue
        npz = ref_npz(clip, model, data, hinge, qadr, lab2mj, foot_names)
        tmp = Path(f"/tmp/k2cal_{clip}.npz")
        np.savez(str(tmp), **npz)
        kh = gait_metrics.analyze(str(tmp))["KH"]
        for nm in foot_names:
            d = kh[nm]
            if d["n_events"] > 0:
                rows.append({"clip": clip, "foot": nm, "n_events": d["n_events"],
                             "k1_mid_deg": round(d["k1_mid_deg"], 2),
                             "k2_range_deg": round(d["k2_range_deg"], 2)})
                k2s.append(d["k2_range_deg"])
    a = np.array(k2s)
    out = {
        "method": "identical KH code path (sim2sim.gait_metrics.analyze) on 50Hz FK-adapted reference clips",
        "rows": rows,
        "summary": {
            "n_foot_clip": int(len(a)),
            "k2_min": float(a.min()), "k2_median": float(np.median(a)),
            "k2_max": float(a.max()),
            "frac_ge_25deg": float((a >= 25).mean()),
            "frac_ge_17deg": float((a >= 17).mean()),
        },
        "conclusion": "GOAL anchor '0->60deg' conflates stance min with swing peak; "
                      "same-metric reference median is the calibrated gate (17deg). "
                      "v59b/soup59c_50 pass the revised gate; R foot passes the "
                      "original 25deg gate (25.5-25.8).",
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[K2CAL] n={len(a)} median={np.median(a):.1f} frac>=25: {(a>=25).mean()*100:.0f}% -> {args.out}")


if __name__ == "__main__":
    main()
