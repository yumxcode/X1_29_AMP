#!/usr/bin/env python3
"""F1w library v2: smooth+decimate the v1 IK output; verify R4 (velocity)
and R2' (FK tracking sacrificed to smoothing).

v1 postmortem (exp_6c5f8f01 + this session's audit): the v1 leg IK hops
between position-equivalent branches (30-90 deg apart) frame to frame ->
stored joint velocities 40-190 rad/s -> the AMP discriminator's joint-vel
channels are unmatchable -> style collapses. Six IK-side variants (v8-v9d:
continuity seeds, trust regions, knee/hip_yaw locks, hysteresis) all left
the branch-lock stretches intact, so the flips persist.

v2 takes the DATA-LEVEL route: zero-phase 4 Hz low-pass + 4x decimation
(120 -> 30 fps). Single/multi-frame flips average out; gait content
(< 6 Hz) is preserved; stored finite-difference velocities become physical
(measured p99 8.8-12.1 rad/s across clips vs X1 delivered library 11.5).

GATES (per clip, honest):
  R4  jv p99 <= 15 and max <= 25 rad/s at the stored fps
  R2' FK(q_smooth) vs v1 key_body_pos p99 <= 30 mm (tracking sacrificed)
Clips failing either are DROPPED (gate discipline: no threshold changes;
the pre-registered R4 was source-calibrated on the X1 delivered library).
"""
import argparse
import functools
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

print = functools.partial(print, flush=True)

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402

F1W_URDF = os.path.join(HERE.parent, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
R4_P99, R4_MAX = 15.0, 25.0
R2P_P99_MM = 30.0
FC, DEC = 4.0, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    lab = load_lab_order(HERE / "robolab/scripts/tools/retarget/config/x1.yaml")
    f1w = UrdfFK(F1W_URDF)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    wname = lambda b: b if b in f1w_links else b[:-len("_link")]
    KEY10_W = [wname(b) for b in KEY10]
    sos = butter(4, FC, btype="low", fs=120.0, output="sos")
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    kept = dropped = 0
    for p in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        name = os.path.basename(p)
        if name.startswith("_"):
            continue
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        kb_raw = np.asarray(clip["key_body_pos"], dtype=np.float64)
        fps0 = float(clip["fps"])
        q_s = sosfiltfilt(sos, q, axis=0)[::DEC]
        rp_s = rp[::DEC]
        rr_s = rr[::DEC]
        kb_s = kb_raw[::DEC]
        fps1 = fps0 / DEC
        # R4: stored-fps finite differences
        jv = np.abs(np.diff(q_s, axis=0) * fps1).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        # R2': FK(q_smooth) vs the v1 key bodies (kept samples)
        Tn = q_s.shape[0]
        res = np.zeros((Tn, 10))
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr_s[t])
            Tbase[:3, 3] = rp_s[t]
            Ts = f1w.fk_all(dict(zip(lab, q_s[t])), Tbase)
            for i, b in enumerate(KEY10_W):
                res[t, i] = np.linalg.norm(Ts[b][:3, 3] - kb_s[t, i])
        r2p = float(np.percentile(res, 99)) * 1000.0
        ok = p99 <= R4_P99 and mx <= R4_MAX and r2p <= R2P_P99_MM
        report[name] = {"jv_p99": p99, "jv_max": mx, "fk_res_p99_mm": r2p, "pass": ok,
                        "fps": fps1, "frames": Tn}
        print(f'{"ok " if ok else "DROP"} {name:34s} jv p99={p99:5.1f} max={mx:5.1f} '
              f'fk p99={r2p:6.1f}mm T={Tn}')
        if ok:
            out = dict(clip)
            out["dof_pos"] = q_s
            out["root_pos"] = rp_s
            out["root_rot"] = rr_s
            out["key_body_pos"] = kb_s.astype(np.float32)
            out["fps"] = fps1
            pickle.dump(out, open(os.path.join(args.dst, name), "wb"))
            kept += 1
        else:
            dropped += 1
    fails = [k for k, v in report.items() if not v["pass"]]
    print(f"\n=== v2 smooth+decimate: kept {kept}, dropped {dropped} ===")
    json.dump(report, open(os.path.join(args.dst, "_v2_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
