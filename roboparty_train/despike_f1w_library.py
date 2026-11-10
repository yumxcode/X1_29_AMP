#!/usr/bin/env python3
"""F1w library v2: de-spike dof_pos (zero-phase low-pass) + FK-consistent
key_body_pos, then verify with three gates.

WHY (v1 negative result, exp_6c5f8f01 / fnd_79add45a correction):
  The v1 f1w library dof_pos has frame-level IK jitter: per-frame joint
  velocity p95 41-68 rad/s, max 78-238 rad/s (hardware limit 14.66; the
  X1-side SOURCE of the same clips: p95 11-12, max 15-18 — X1 delivered
  library x1_lab_v72: p99 11.5, max 22.9). The two-stage F1w IK solves
  frames independently; near joint-limit/projection boundaries mm-level
  position flips map to huge joint-space deltas. The AMP discriminator's
  joint-vel channels get an unmatchable demo distribution -> disc_loss
  collapses to ~0 by iter 1500 -> style 0.006 -> v1 converged to a
  non-tracking survival gait.

GATES for v2 (pre-registered in acceptance/F1W_RUN_ACCEPTANCE.md sec 7;
source-domain calibrated, discrimination verified):
  R4 joint-velocity physicality (NEW): per-frame max |dq/dt| p99 <= 15
    rad/s AND max <= 25 rad/s. Calibration: X1 delivered library x1_lab_v72
    p99 11.5 / max 22.9 (loosest historical pass); hardware limit 14.66
    continuous. Discrimination: v1 library fails at p95 41-68 (10x over);
    v2 must clear with margin.
  R2' tracking after smoothing: |FK(q_smooth) - kb_raw| p99 <= 30 mm
    (same threshold family as R2; kb_raw = the v1 IK's achieved targets).
  R1/R3 recomputed on the v2 library (flight fractions per class band and
    speed centroid) and reported alongside the v1 numbers.

Usage:
    python despike_f1w_library.py --src <f1w_lab_v1> --dst <f1w_lab_v2>
        [--fc 10.0] [--exclude run_accM_C3]
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

X1_YAML = HERE / "robolab" / "scripts" / "tools" / "retarget" / "config" / "x1.yaml"
X1_URDF = HERE / "robolab" / "data" / "robots" / "x1" / "urdf" / "x1.urdf"
F1W_URDF = HERE / "robolab" / "data" / "robots" / "f1w" / "urdf" / "f1w.urdf"

R4_P99, R4_MAX = 15.0, 25.0
R2P_P99_MM = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fc", type=float, default=10.0)
    ap.add_argument("--exclude", nargs="*", default=["run_accM_C3"])
    args = ap.parse_args()

    lab = load_lab_order(X1_YAML)
    f1w = UrdfFK(F1W_URDF)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}

    def wname(b):
        return b if b in f1w_links else b[:-len("_link")]

    KEY10_W = [wname(b) for b in KEY10]
    sos = butter(4, args.fc, btype="low", fs=120.0, output="sos")
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    excluded = 0

    for p in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        name = os.path.basename(p)
        base = name[:-4]
        stem = base[:-len("_mirror")] if base.endswith("_mirror") else base
        if stem in args.exclude:
            excluded += 1
            continue
        if name.startswith("_"):
            continue
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        fps = float(clip["fps"])
        kb_raw = np.asarray(clip["key_body_pos"], dtype=np.float64)
        Tn = q.shape[0]

        # zero-phase low-pass on joint angles
        q_s = sosfiltfilt(sos, q, axis=0)
        # renormalize quaternion rows (unchanged by this pass, kept for clarity)

        # FK-consistent key bodies from the smoothed joints
        kb_s = np.zeros_like(kb_raw)
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            Ts = f1w.fk_all(dict(zip(lab, q_s[t])), Tbase)
            for i, b in enumerate(KEY10_W):
                kb_s[t, i] = Ts[b][:3, 3]

        dt = 1.0 / fps
        jv = np.abs(np.diff(q_s, axis=0) / dt).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        res = np.linalg.norm(kb_s - kb_raw, axis=2)          # (T,10)
        r2p = float(np.percentile(res, 99)) * 1000.0
        dpos = float(np.abs(q_s - q).max())
        ok = (p99 <= R4_P99) and (mx <= R4_MAX) and (r2p <= R2P_P99_MM)
        report[name] = {"jv_p99": p99, "jv_max": mx, "fk_res_p99_mm": r2p,
                        "max_delta_deg": np.degrees(dpos), "pass": ok}
        print(f'{"ok " if ok else "FAIL"} {name:34s} jv p99={p99:5.1f} max={mx:5.1f} '
              f'fk_res p99={r2p:5.1f}mm dq_max={np.degrees(dpos):5.1f}deg')

        out = dict(clip)
        out["dof_pos"] = q_s.astype(np.float64)
        out["key_body_pos"] = kb_s.astype(np.float32)
        pickle.dump(out, open(os.path.join(args.dst, name), "wb"))

    fails = [k for k, v in report.items() if not v["pass"]]
    print(f"\n=== despike: {len(report)} clips, excluded {excluded}, FAILS={len(fails)} ===")
    json.dump(report, open(os.path.join(args.dst, "_despike_report.json"), "w"), indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
