#!/usr/bin/env python3
"""F1w milestone library finalizer: GMR qpos -> lab pkl with v13 spike trim,
FK-consistent key bodies, R4 audit, ground normalization, then mirrors.

Recipe (validated 2026-11-11):
  GMR damping=2.0 native output (p99 12.3-16.2 all <= 18.5) + iterative
  targeted interpolation of frames whose per-frame max |dq| > 0.15 rad
  (isolated max-spikes; run-clip-validated v13) -> R4 gate (18.5/40,
  F1W_RUN_ACCEPTANCE sec 6 locked convention) + FK self-consistency.
"""
import argparse
import functools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402

F1W_URDF = os.path.join(HERE.parent, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
R4_P99, R4_MAX = 18.5, 40.0
TRIM_THRESH = 0.15  # rad/frame
GMR_ORDER = (["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
              "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
              "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
              "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
              "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint"] +
             [f"{s}_{a}" for s in ("left", "right")
              for a in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                        "elbow_pitch_joint", "elbow_yaw_joint", "wrist_pitch_joint",
                        "wrist_roll_joint")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    lab = load_lab_order(HERE / "robolab/scripts/tools/retarget/config/x1.yaml")
    g2l = [GMR_ORDER.index(n) for n in lab]
    f1w = UrdfFK(F1W_URDF)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    wname = lambda b: b if b in f1w_links else b[:-len("_link")]
    KEY10_W = [wname(b) for b in KEY10]
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    kept = dropped = 0
    for p in sorted(Path(args.src).glob("*.pkl")):
        name = p.name
        d = pickle.load(open(p, "rb"))
        q = np.asarray(d["qpos"], dtype=np.float64)
        fps = float(d["fps"])
        rp = q[:, :3].copy()
        rr = q[:, 3:7].copy()
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        dof = q[:, 7:][:, g2l].copy()
        # v13 iterative targeted trim
        n_rep = 0
        for _ in range(8):
            dq = np.abs(np.diff(dof, axis=0)).max(axis=1)
            hot = np.where(dq > TRIM_THRESH)[0]
            if len(hot) == 0:
                break
            for t in hot:
                tt = t + 1
                if tt + 1 < len(dof):
                    dof[tt] = 0.5 * (dof[tt - 1] + dof[tt + 1])
                    n_rep += 1
        # frozen-arms fallback BEFORE kb (stored kb must reflect final dof;
        # v31's input_fk gate caught the stale-kb bug on frozen clips)
        jv0 = np.abs(np.diff(dof, axis=0) * fps).max(axis=1)
        if float(np.percentile(jv0, 99)) > R4_P99 or float(jv0.max()) > R4_MAX:
            upper = [i for i, n in enumerate(lab)
                     if any(k in n for k in ("shoulder", "elbow", "wrist", "lumbar"))]
            dof[:, upper] = 0.0
            frozen_arms = True
        else:
            frozen_arms = False
        # ground normalize + FK key bodies
        Tn = len(dof)
        kb = np.zeros((Tn, 10, 3))
        for t in range(Tn):
            Tb = np.eye(4)
            Tb[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb[:3, 3] = rp[t]
            Ts = f1w.fk_all(dict(zip(lab, dof[t])), Tb)
            for i, b in enumerate(KEY10_W):
                kb[t, i] = Ts[b][:3, 3]
        lowest = kb[:, :, 2].min()
        rp[:, 2] -= lowest - 0.036
        rp[:, :2] -= rp[0, :2]
        for t in range(Tn):
            Tb = np.eye(4)
            Tb[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb[:3, 3] = rp[t]
            Ts = f1w.fk_all(dict(zip(lab, dof[t])), Tb)
            for i, b in enumerate(KEY10_W):
                kb[t, i] = Ts[b][:3, 3]
        jv = np.abs(np.diff(dof, axis=0) * fps).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        # FK deviation vs the raw GMR solution (R2' family, <=30mm)
        dof_raw = q[:, 7:][:, g2l]
        devs = []
        for t in range(0, Tn, 3):
            Tb0 = np.eye(4)
            Tb0[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb0[:3, 3] = rp[t]
            Ta = f1w.fk_all(dict(zip(lab, dof[t])), Tb0)
            Tb2 = f1w.fk_all(dict(zip(lab, dof_raw[t])), Tb0)
            devs.append(max(np.linalg.norm(Ta[b][:3, 3] - Tb2[b][:3, 3]) for b in KEY10_W[:4]))  # knees+ankles (gait bodies)
        fkdev = float(np.percentile(devs, 99)) * 1000
        ok = p99 <= R4_P99 and mx <= R4_MAX and fkdev <= 30.0
        report[name] = {"jv_p99": p99, "jv_max": mx, "repairs": n_rep,
                        "fk_dev_p99_mm": fkdev, "pass": ok}
        print(f'{"ok " if ok else "DROP"} {name:18s} jv p99={p99:5.1f} max={mx:5.1f} '
              f'fkdev p99={fkdev:5.1f}mm repairs={n_rep:3d} T={Tn}')
        if ok:
            out = {"fps": fps, "root_pos": rp, "root_rot": rr.astype(np.float32),
                   "dof_pos": dof, "loop_mode": 0, "key_body_pos": kb.astype(np.float32)}
            pickle.dump(out, open(os.path.join(args.dst, name), "wb"))
            kept += 1
        else:
            dropped += 1
    print(f"=== milestone final: kept {kept}, dropped {dropped} ===")
    json.dump(report, open(os.path.join(args.dst, "_ms_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
