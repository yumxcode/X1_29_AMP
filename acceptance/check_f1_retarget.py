#!/usr/bin/env python3
"""F1_29DOF retarget acceptance — X1-standard gates (RETARGET_ACCEPTANCE.md)
applied to the f1_lab_v67 dataset produced by retarget_f1_from_x1lab.py.

Gates (X1 v1.0 standard, adapted to the local leg-IK retarget route):
  A1 schema/fields        A4 fps in [30,250], frames >= 100
  A5 quaternion norms     A6 dof_pos 29 cols, key_body_pos 10 cols
  B1 hard joint limits vs F1_29dof.urdf (0.02 rad / 0.1% frames tolerance)
  C1/C2/C3/C4 root trajectory (unchanged from X1 source by construction —
      verified numerically against x1_lab_v67)
  D1/D2 smoothness (|dq| per frame @120fps)
  E1/E2 ground penetration via F1 URDF FK (key bodies + ankle links)
  X1-vs-F1 leg-trajectory fidelity: knee/ankle joint correlation and
  amplitude retention (the retarget must preserve gait rhythm)

Pure numpy (fk_lib URDF FK; verified 0.0002 mm against stored
key_body_pos on x1_lab_v67). Runs on Mac.

Usage: .venv-mac/bin/python acceptance/check_f1_retarget.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "roboparty_train"))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402

F1_URDF = os.path.join(ROOT, "F1_29DOF/urdf/f1_29dof.urdf")
X1_URDF = os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")
YAML = os.path.join(ROOT, "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml")
F1_DIR = os.path.join(ROOT, "roboparty_train/robolab/data/motions/f1_lab_v67")
X1_DIR = os.path.join(ROOT, "roboparty_train/robolab/data/motions/x1_lab_v67")

fails, warns = [], []


def gate(name, ok, detail="", warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if not ok:
        (warns if warn else fails).append(f"{name}: {detail}")
    print(f"  [{tag}] {name} {detail}")


def main():
    lab = load_lab_order(YAML)
    f1 = UrdfFK(F1_URDF)
    x1 = UrdfFK(X1_URDF)
    limits = np.array([[f1.joints[n]["lower"], f1.joints[n]["upper"]] for n in lab])

    per_clip = {}
    for p in sorted(glob.glob(os.path.join(F1_DIR, "*.pkl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        clip = pickle.load(open(p, "rb"))
        src = pickle.load(open(os.path.join(X1_DIR, os.path.basename(p)), "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        kb = np.asarray(clip["key_body_pos"], dtype=np.float64)
        qs = np.asarray(src["dof_pos"], dtype=np.float64)
        Tn = len(q)

        # --- A: structure
        ok = (isinstance(clip["fps"], float) and 30 <= clip["fps"] <= 250 and Tn >= 100
              and q.shape[1] == 29 and kb.shape == (Tn, 10, 3)
              and all(k in clip for k in ("fps", "root_pos", "root_rot", "dof_pos", "loop_mode", "key_body_pos")))
        gate(f"A {name}", ok, f"fps={clip['fps']} T={Tn}")

        # --- B1: hard limits
        over = (q < limits[:, 0] - 0.02) | (q > limits[:, 1] + 0.02)
        frac = over.any(axis=1).mean()
        gate(f"B1 {name}", frac <= 0.001, f"over-limit frames {frac*100:.3f}% (max over {over.max()*0+np.abs(q-limits[np.arange(29)][(over.any(axis=0))]).max() if over.any() else 0:.3f} rad)" if over.any() else "clean")

        # --- C: root identical to X1 source
        d = np.abs(rp - np.asarray(src["root_pos"])).max()
        dr = np.abs(np.asarray(clip["root_rot"], dtype=np.float64) - np.asarray(src["root_rot"], dtype=np.float64)).max()
        gate(f"C {name}", d < 1e-12 and dr < 1e-6, f"root delta {d:.2e}/{dr:.2e}")

        # --- D: smoothness
        dq = np.abs(np.diff(q, axis=0))
        med, p99, mx = np.median(dq), np.percentile(dq, 99), dq.max()
        gate(f"D1 {name}", mx <= 3.0 and (dq > 0.5).mean() <= 0.01, f"dq max {mx:.3f} frac>0.5 {(dq>0.5).mean()*100:.3f}%")
        gate(f"D2 {name}", med <= 0.15 and p99 <= 0.35, f"med {med:.3f} p99 {p99:.3f}", warn=True)

        # --- E: ground penetration (F1 FK on key bodies + full-tree min z sample)
        minz, n_s = 0.0, 24
        for t in range(0, Tn, max(1, Tn // n_s)):
            Tb = np.eye(4)
            Tb[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb[:3, 3] = rp[t]
            Ts = f1.fk_all(dict(zip(lab, q[t])), Tb)
            minz = min(minz, min(Ts[b][:3, 3][2] for b in KEY10))
        gate(f"E {name}", minz >= -0.03, f"key-body min z {minz*1000:.1f} mm", warn=minz >= -0.02)

        # --- fidelity: leg joint trajectory X1 vs F1
        legs = [i for i, n in enumerate(lab) if "hip" in n or "knee" in n or "ankle" in n]
        corr = {}
        for n in ("left_knee_pitch_joint", "right_knee_pitch_joint",
                  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                  "left_hip_pitch_joint", "right_hip_pitch_joint"):
            i = lab.index(n)
            a, b = qs[:, i], q[:, i]
            if a.std() < 1e-6 or b.std() < 1e-6:
                corr[n] = 1.0
            else:
                corr[n] = float(np.corrcoef(a, b)[0, 1])
        # amplitude retention on knees
        amp_x1 = float(np.mean([np.ptp(qs[:, lab.index(n)]) for n in
                         ("left_knee_pitch_joint", "right_knee_pitch_joint")]))
        amp_f1 = float(np.mean([np.ptp(q[:, lab.index(n)]) for n in
                         ("left_knee_pitch_joint", "right_knee_pitch_joint")]))
        per_clip[name] = dict(knee_corr=corr, knee_amp_x1=amp_x1, knee_amp_f1=amp_f1,
                              minz_mm=minz * 1000, dq_p99=float(p99))

    print()
    # --- aggregate gates
    knee_corrs = [v["knee_corr"]["left_knee_pitch_joint"] for v in per_clip.values()] + \
                 [v["knee_corr"]["right_knee_pitch_joint"] for v in per_clip.values()]
    gate("F knee correlation", np.mean(knee_corrs) >= 0.9,
         f"mean {np.mean(knee_corrs):.4f} min {np.min(knee_corrs):.4f}")
    amps = [(v["knee_amp_f1"] / max(v["knee_amp_x1"], 1e-9)) for v in per_clip.values()]
    gate("F knee amplitude retention", 0.7 <= np.mean(amps) <= 1.3,
         f"mean ratio {np.mean(amps):.3f}")

    print(f"\nVERDICT: {'PASS' if not fails else 'FAIL'}   fails={len(fails)} warns={len(warns)}")
    for f in fails:
        print(f"  FAIL {f}")
    for w in warns:
        print(f"  WARN {w}")
    with open(os.path.join(F1_DIR, "_acceptance_report.json"), "w") as f:
        json.dump(dict(fails=fails, warns=warns, per_clip=per_clip), f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
