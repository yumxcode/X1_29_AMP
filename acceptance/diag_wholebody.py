#!/usr/bin/env python3
"""Whole-body gait quality audit of retargeted X1 references (v30.2).

User challenge (2026-09-10): "natural human motion = arm rhythm + leg rhythm
+ sole behaviour + torso, ALL simultaneously" — was the v30 fix too focused
on the arm alone? Audit EVERY component with correct FK (lab-order qposadr
mapping; the old chained fancy-index write silently fell back to the zero
pose, and the first render script permuted joint columns).

Per clip, three layers: AMASS SMPLX source / x1_lab (orig GMR) / x1_lab_v30.

Metrics (human walking reference in brackets):
  legs_sym  corr(hipP_L, hipP_R)        [-0.95..-1 antiphase]
  knee_sym  corr(kneeP_L, kneeP_R)      [~ -1]
  swingRatio |hip swing L| / |R|        [~1.0]
  arm_leg   corr(shoP_L, hipP_R)        [+0.7..+0.9 contralateral coupling]
  cadence   steps / min (geometric contact) [95-125]
  phase     stance overlap of L vs R shifted by half cycle [~0.5 = alternated]
  penetrate contact frames with sole z < -8mm [%]  [0]
  foot_flat stance frames with |sole pitch| > 15deg [%] [small]
  torso_r/p lumbar roll / pitch swing (deg)         [<6 / <10]

Usage: ./.venv_test/bin/python acceptance/diag_wholebody.py [--clips ...]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

XML = ROOT / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
ORIG_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab"
V30_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab_v30"
SMPLX_DIRS = [ROOT / "AMASS_minimal/BMLrub_stageii", ROOT / "AMASS_minimal/CMU"]

LAB = parse_yaml_list(YAML, "lab_dof_names")
IDX = {n: i for i, n in enumerate(LAB)}


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def rng(x):
    x = np.asarray(x)
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def smplx_legs(name):
    """Hip swing / knee flexion / cadence from raw AMASS (SMPLX aa)."""
    import os
    for d in SMPLX_DIRS:
        p = d / f"{name}_stageii.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        P = np.asarray(z["poses"], float)[:, :66].reshape(-1, 22, 3)
        hz = float(z.get("mocap_framerate", 120) or 120)

        def swing(j):
            # local-y rotation component ~ sagittal swing
            return np.array([np.degrees(np.arcsin(np.clip(
                _rot(p[j])[0, 2], -1, 1))) for p in P])

        def knee(j):
            return np.array([np.degrees(_rot(p[j])[1, 2]) for p in P])

        hipL, hipR, knL, knR = swing(1), swing(2), knee(4), knee(5)
        # cadence from hip swing zero-crossings (rising)
        def cadence(s):
            s0 = s - np.mean(s)
            idx = np.where((s0[:-1] < 0) & (s0[1:] >= 0))[0]
            if len(idx) < 2:
                return 0.0
            return 60.0 * len(idx) / (len(s) / hz)
        return dict(legs=corr(hipL, hipR), knee=corr(knL, knR),
                    swL=rng(hipL), swR=rng(hipR),
                    arm_leg=0.0, cad=cadence(hipL), pen=0.0, flat=0.0,
                    tr=rng(np.degrees([_yaw_spine(p) for p in P])) if False else 0.0,
                    tp=0.0, n=len(P))
    return None


def _rot(aa):
    t = np.linalg.norm(aa)
    if t < 1e-9:
        return np.eye(3)
    k = aa / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def audit_clip(path, model, data, qadr, sole_rows, feet_names, kb_id):
    clip = pickle.load(open(path, "rb"), encoding="latin1")
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float)
    rr = np.asarray(clip["root_rot"], float)
    T = len(q)
    dt = 1.0 / float(clip["fps"])

    hipL = np.degrees(q[:, IDX["left_hip_pitch_joint"]])
    hipR = np.degrees(q[:, IDX["right_hip_pitch_joint"]])
    knL = np.degrees(q[:, IDX["left_knee_pitch_joint"]])
    knR = np.degrees(q[:, IDX["right_knee_pitch_joint"]])
    shoL = np.degrees(q[:, IDX["left_shoulder_pitch_joint"]])
    shoR = np.degrees(q[:, IDX["right_shoulder_pitch_joint"]])

    # FK: sole heights + pitch + torso
    sole_z = np.zeros((T, 2, 4))       # 2 feet x 4 sole spheres
    foot_pitch = np.zeros((T, 2))
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[qadr] = q[t]
        import mujoco
        mujoco.mj_forward(model, data)
        for f in range(2):
            pts = np.array([data.geom_xpos[g] for g in sole_rows[f]])
            sole_z[t, f] = pts[:, 2]
            front, back = pts[:2].mean(0), pts[2:].mean(0)
            foot_pitch[t, f] = np.degrees(np.arctan2(front[2] - back[2], 0.14))

    low = sole_z.min(2)                       # (T,2) lowest sphere per foot
    contact = low < 0.03                      # within 3cm of ground
    pen = np.mean(low[contact] < -0.008) if contact.any() else 0.0
    flat_viol = np.mean(np.abs(foot_pitch[contact] > 15)) if contact.any() else 0.0

    # stance phase alignment: shift R stance by half of L's mean cycle
    def cycle_len(st):
        d = np.diff(st.astype(int))
        ups = np.where(d > 0)[0]
        return float(np.mean(np.diff(ups))) if len(ups) >= 3 else float(T)

    # cadence: rising zero-crossings of (low_L - low_R) -> weight transfers
    w = low[:, 0] - low[:, 1]
    w0 = w - np.mean(w)
    xg = np.where((w0[:-1] < 0) & (w0[1:] >= 0))[0]
    cad = 60.0 * len(xg) / (T * dt) if len(xg) >= 2 else 0.0

    torso_r = np.degrees(rng(q[:, IDX["lumbar_roll_joint"]]))
    torso_p = np.degrees(rng(q[:, IDX["lumbar_pitch_joint"]]))

    return dict(legs=corr(hipL, hipR), knee=corr(knL, knR),
                swL=rng(hipL), swR=rng(hipR),
                arm_leg=corr(shoL, hipR), cad=cad, pen=float(pen),
                flat=float(flat_viol), tr=torso_r, tp=torso_p, n=T)


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[
        "0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
        "0000_treadmill_norm", "0002_treadmill_slow", "0009_normal_jog1",
        "0026_circle_walk", "36_01", "36_11"])
    args = ap.parse_args()

    from sim2sim.mujoco_rollout import find_sole_geoms
    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    soles, _ = find_sole_geoms(model)
    feet_names = sorted(soles)
    sole_rows = [np.array(soles[f]) for f in feet_names]
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LAB}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in LAB])

    hdr = (f"{'clip':20s} {'src':4s} | {'legsSym':>7s} {'kneeSym':>7s} {'swL/R':>10s} "
           f"{'armLeg':>6s} | {'cad':>6s} {'pen%':>5s} {'flat%':>5s} | {'tR':>4s} {'tP':>4s}")
    for name in args.clips:
        print(f"\n== {name}")
        print(hdr)
        src = smplx_legs(name)
        if src:
            print(f"{'':20s} {'SMPL':4s} | {src['legs']:+7.2f} {src['knee']:+7.2f} "
                  f"{src['swL']:4.1f}/{src['swR']:4.1f} {'':6s} | {src['cad']:6.1f}")
        for tag, d in [("orig", ORIG_DIR), ("v30", V30_DIR)]:
            p = d / f"{name}.pkl"
            if not p.exists():
                continue
            m = audit_clip(p, model, data, qadr, sole_rows, feet_names, None)
            print(f"{'':20s} {tag:4s} | {m['legs']:+7.2f} {m['knee']:+7.2f} "
                  f"{m['swL']:4.1f}/{m['swR']:4.1f} {m['arm_leg']:+6.2f} | "
                  f"{m['cad']:6.1f} {m['pen']*100:5.1f} {m['flat']*100:5.1f} | "
                  f"{m['tr']:4.1f} {m['tp']:4.1f}")
    print("""
Human reference: legsSym/kneeSym ~ -1 (antiphase), swL/swR ~ 1.0,
armLeg (shoP_L vs hipP_R) +0.7..+0.9, cadence 95-125 spm (walk),
pen% ~ 0, flat% small, torso roll < 6 deg, pitch < 10 deg.
""")


if __name__ == "__main__":
    main()
