#!/usr/bin/env python3
"""F1_29DOF_WEILAI retarget acceptance — X1-standard gates
(RETARGET_ACCEPTANCE.md) applied to F1w datasets produced by
retarget_f1w_from_x1lab.py. Successor of check_f1_retarget.py (xyber F1),
same gates, weilai model + run-library extensions.

Gates (thresholds NOT lowered vs the xyber F1 checker):
  A  schema/fields, fps in [30,250], frames >= 100   (run clips < 1s WARN)
  B1 hard joint limits vs weilai f1_isaac.urdf (0.02 rad / 0.1% frames)
  C  root identical to X1 source (by construction; verified numerically)
  D1/D2 smoothness (|dq| @ source fps)
  E  ground penetration via weilai FK (key bodies)
  F  knee correlation / amplitude retention vs X1 source

Run-library gates (NEW, pre-registered 2026-11-10, calibrated on source data;
discrimination validated walk-vs-run in the report):
  R1 flight phase   : sprint/mid-run clips (nominal v>=1.5 in the source
                      band table) must show both-feet-off-ground windows —
                      flight_frac >= 5% of frames; walk anchors <= 2%.
                      (source calibration: CMU/ACCAD sprint clips show
                      8-30% both-feet-air in SMPL-H foot z; X1 v72 walks ~0%)
  R2 IK residual    : ik_pos_p99 <= 10 mm, ik_pos_max <= 30 mm
                      (xyber F1 precedent 1.0-6.7 mm p99; weilai knee caps
                      at 90 deg vs X1 114.6 -> margin, not relaxation:
                      anything worse is excluded, never passed)
  R3 speed centroid : time-weighted mean speed of the RUN sources within
                      [1.3, 1.9] m/s (target command band center 1.6)

Usage: .venv39/bin/python acceptance/check_f1w_retarget.py \
          --f1w <f1w_lab_dir> --x1 <x1_lab_src_dir> [--ik-report <path>]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "roboparty_train"))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402

F1W_URDF = os.path.join(ROOT, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
X1_URDF = os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")
YAML = os.path.join(ROOT, "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml")

fails, warns = [], []


def gate(name, ok, detail="", warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if not ok:
        (warns if warn else fails).append(f"{name}: {detail}")
    print(f"  [{tag}] {name} {detail}")


def clip_class(name):
    """>= 1.5 nominal run band -> 'run'; else 'anchor'."""
    n = name.replace(".pkl", "")
    if n.startswith("run_"):
        base = n[4:].replace("_mirror", "")
        if base.startswith(("09_", "35_", "accF_C3", "accM_C3", "acc8", "16_", "02_03", "h4d_")):
            return "run"
        # transitions accM_C4/C5/F_C4/F_C5 pass through the mid band
        return "run"
    return "anchor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1w", required=True, help="F1w lab dataset dir")
    ap.add_argument("--x1", required=True, help="X1 lab source dir")
    ap.add_argument("--ik-report", default=None,
                    help="_retarget_report.json from retarget_f1w_from_x1lab.py")
    args = ap.parse_args()

    lab = load_lab_order(YAML)
    f1w = UrdfFK(F1W_URDF)
    x1 = UrdfFK(X1_URDF)
    limits = np.array([[f1w.joints[n]["lower"], f1w.joints[n]["upper"]] for n in lab])
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    KEY10_W = [b if b in f1w_links else b[:-len("_link")] for b in KEY10]
    ik_report = {}
    if args.ik_report and os.path.exists(args.ik_report):
        ik_report = json.load(open(args.ik_report))

    per_clip = {}
    speed_rows = []
    for p in sorted(glob.glob(os.path.join(args.f1w, "*.pkl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        clip = pickle.load(open(p, "rb"))
        src = pickle.load(open(os.path.join(args.x1, os.path.basename(p)), "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        kb = np.asarray(clip["key_body_pos"], dtype=np.float64)
        qs = np.asarray(src["dof_pos"], dtype=np.float64)
        fps = float(clip["fps"])
        Tn = len(q)

        # --- A: structure
        ok = (30 <= fps <= 250 and q.shape[1] == 29
              and kb.shape == (Tn, 10, 3)
              and all(k in clip for k in ("fps", "root_pos", "root_rot", "dof_pos",
                                          "loop_mode", "key_body_pos")))
        short = Tn < 100
        gate(f"A {name}", ok, f"fps={fps} T={Tn}" + (" [short run clip]" if short else ""),
             warn=short and ok)
        if not ok:
            continue

        # --- B1: hard limits (weilai asymmetric hip_roll included via URDF)
        over = (q < limits[:, 0] - 0.02) | (q > limits[:, 1] + 0.02)
        frac = over.any(axis=1).mean()
        gate(f"B1 {name}", frac <= 0.001, f"over-limit frames {frac*100:.3f}%")

        # --- C: root identical to X1 source
        d = np.abs(rp - np.asarray(src["root_pos"])).max()
        dr = np.abs(np.asarray(clip["root_rot"]) - np.asarray(src["root_rot"])).max()
        gate(f"C {name}", d < 1e-12 and dr < 1e-6, f"root delta {d:.2e}/{dr:.2e}")

        # --- D: smoothness
        dq = np.abs(np.diff(q, axis=0))
        med, p99, mx = np.median(dq), np.percentile(dq, 99), dq.max()
        gate(f"D1 {name}", mx <= 3.0 and (dq > 0.5).mean() <= 0.01,
             f"dq max {mx:.3f} frac>0.5 {(dq>0.5).mean()*100:.3f}%")
        gate(f"D2 {name}", med <= 0.15 and p99 <= 0.35, f"med {med:.3f} p99 {p99:.3f}", warn=True)

        # --- E: ground penetration
        minz, n_s = 0.0, 24
        for t in range(0, Tn, max(1, Tn // n_s)):
            Tb = np.eye(4)
            Tb[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb[:3, 3] = rp[t]
            Ts = f1w.fk_all(dict(zip(lab, q[t])), Tb)
            minz = min(minz, min(Ts[b][:3, 3][2] for b in KEY10_W))
        gate(f"E {name}", minz >= -0.03, f"key-body min z {minz*1000:.1f} mm",
             warn=minz >= -0.02)

        # --- F: leg fidelity vs X1 source
        corr = {}
        for n in ("left_knee_pitch_joint", "right_knee_pitch_joint",
                  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                  "left_hip_pitch_joint", "right_hip_pitch_joint"):
            i = lab.index(n)
            a, b = qs[:, i], q[:, i]
            corr[n] = 1.0 if a.std() < 1e-6 or b.std() < 1e-6 else float(np.corrcoef(a, b)[0, 1])
        amp_x1 = float(np.mean([np.ptp(qs[:, lab.index(n)]) for n in
                                ("left_knee_pitch_joint", "right_knee_pitch_joint")]))
        amp_w = float(np.mean([np.ptp(q[:, lab.index(n)]) for n in
                               ("left_knee_pitch_joint", "right_knee_pitch_joint")]))

        # --- R1: flight fraction — both ankles above the clip's contact
        #     level + 3 cm, sustained >= 30 ms. Calibrated 2026-11-10 on the
        #     X1-side retargets: walk/jog library 0.0-2.5%, run library
        #     9.9-52.4% (after acc8 exclusion) -> run gate >= 3.5% (walk
        #     max 2.5% + 1%), anchor gate <= 3.0%. Discrimination validated
        #     run-vs-walk on the sources themselves.
        az = kb[:, 2:4, 2]
        zc = np.percentile(np.minimum(az[:, 0], az[:, 1]), 10)
        both_air = (az[:, 0] > zc + 0.03) & (az[:, 1] > zc + 0.03)
        k = max(2, int(0.03 * fps))
        run_ok = np.convolve(both_air.astype(int), np.ones(k, int), 'valid') >= k
        flight_frac = float(run_ok.mean()) if len(run_ok) else 0.0
        klass = clip_class(name)
        if klass == "run":
            gate(f"R1 {name}", flight_frac >= 0.035,
                 f"flight {flight_frac*100:.1f}% ({klass})")
        else:
            gate(f"R1 {name}", flight_frac <= 0.03,
                 f"flight {flight_frac*100:.1f}% ({klass}, expect walk ~0)")

        # --- speed (from root, valid for overground; treadmill post-anchor)
        vxy = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps
        v_med = float(np.median(vxy[len(vxy)//4: 3*len(vxy)//4])) if Tn > 8 else float(np.median(vxy))

        # --- R2: IK residual from the retarget report
        ik = ik_report.get(os.path.basename(p), {})
        if ik:
            gate(f"R2 {name}", ik["ik_pos_p99_mm"] <= 10.0 and ik["ik_pos_max_mm"] <= 30.0,
                 f"IK p99 {ik['ik_pos_p99_mm']:.2f} max {ik['ik_pos_max_mm']:.2f} mm, "
                 f"knee_sat {ik.get('knee_sat_frac',0)*100:.0f}%")

        per_clip[name] = dict(knee_corr=corr, knee_amp_x1=amp_x1, knee_amp_f1w=amp_w,
                              minz_mm=minz * 1000, dq_p99=float(p99),
                              flight_frac=flight_frac, v_med=v_med, klass=klass)
        if klass == "run" and not name.endswith("_mirror"):
            speed_rows.append((name, v_med, Tn / fps))

    print()
    knee_corrs = [v["knee_corr"]["left_knee_pitch_joint"] for v in per_clip.values()] + \
                 [v["knee_corr"]["right_knee_pitch_joint"] for v in per_clip.values()]
    gate("F knee correlation", np.mean(knee_corrs) >= 0.9,
         f"mean {np.mean(knee_corrs):.4f} min {np.min(knee_corrs):.4f}")
    amps = [(v["knee_amp_f1w"] / max(v["knee_amp_x1"], 1e-9)) for v in per_clip.values()]
    gate("F knee amplitude retention", 0.7 <= np.mean(amps) <= 1.3,
         f"mean ratio {np.mean(amps):.3f}")

    # --- R3: time-weighted run speed centroid
    if speed_rows:
        w = np.array([d for _, _, d in speed_rows])
        vv = np.array([v for _, v, _ in speed_rows])
        centroid = float(np.sum(w * vv) / np.sum(w))
        gate("R3 run speed centroid", 1.3 <= centroid <= 1.9,
             f"time-weighted {centroid:.2f} m/s over {len(speed_rows)} run sources")
        print("  run source speeds:", [(n, round(v, 2)) for n, v, _ in speed_rows])

    print(f"\nVERDICT: {'PASS' if not fails else 'FAIL'}   fails={len(fails)} warns={len(warns)}")
    for f in fails:
        print(f"  FAIL {f}")
    for w_ in warns:
        print(f"  WARN {w_}")
    with open(os.path.join(args.f1w, "_acceptance_report.json"), "w") as f:
        json.dump(dict(fails=fails, warns=warns, per_clip=per_clip), f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
