#!/usr/bin/env python3
"""Policy/reference term-score audit (permanent tool, v34).

Scores any MuJoCo rollout npz (mujoco_rollout.py --log) AND the reference
clips on the CURRENT arm/torso reward statistics + world-frame arm metrics.
Used for: mid-training health checks, final acceptance, v-next calibration.

Usage: python acceptance/diag_policy_terms.py [--logs a.npz b.npz ...] [--refs ...]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LAB = ["left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
       "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
       "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
       "left_knee_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
       "right_knee_pitch_joint", "left_ankle_pitch_joint", "left_shoulder_roll_joint",
       "right_shoulder_roll_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint",
       "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "right_ankle_roll_joint",
       "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_elbow_yaw_joint",
       "right_elbow_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
       "left_wrist_roll_joint", "right_wrist_roll_joint"]


def ema_cont(x, alpha):
    out = np.empty_like(x)
    e = x[0]
    for i, v in enumerate(x):
        e = (1 - alpha) * e + alpha * v
        out[i] = e
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def fk_phis(q, bp, bq, hinge):
    """World-frame upper-arm forward angle (deg), sampled."""
    import mujoco
    model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)
    BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in hinge}
    T = len(q)
    stride = max(1, T // 150)
    out = {"left": [], "right": []}
    for t in range(0, T, stride):
        data.qpos[:] = 0
        data.qpos[0:3] = bp[t]
        data.qpos[3:7] = bq[t] / np.linalg.norm(bq[t])
        for n in hinge:
            data.qpos[qadr[n]] = q[t, hinge.index(n)]
        mujoco.mj_forward(model, data)
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, data.xquat[BID("base_link")])
        Rm = Rm.reshape(3, 3)
        fwd = Rm[:, 0]
        for side in ("left", "right"):
            u = data.xpos[BID(f"{side}_elbow_pitch_link")] - \
                data.xpos[BID(f"{side}_shoulder_pitch_link")]
            out[side].append(np.degrees(np.arctan2(u @ fwd, -(u @ Rm[:, 2]))))
    return np.array(out["left"]), np.array(out["right"])


def audit(name, shoL, shoR, hipL, hipR, lumP, extra=(), tail_frac=0.25):
    N = 40
    shoL = np.tile(shoL, N); shoR = np.tile(shoR, N)
    hipL = np.tile(hipL, N); hipR = np.tile(hipR, N); lumP = np.tile(lumP, N)
    hp = lambda x: x - ema_cont(x, 0.02)
    hp_l, hp_r = hp(shoL), hp(shoR)
    hp_hl, hp_hr = hp(hipL), hp(hipR)
    tail = slice(int(len(shoL) * (1 - tail_frac)), None)

    sync = -0.3 * np.abs(hp_l + hp_r)[tail].mean()
    asym = -0.2 * np.abs(ema_cont(shoL - shoR, 0.0025))[tail].mean()
    coup_t = 0.5 * (np.clip(hp_l * hp_hr, -0.15, 0.15)
                    + np.clip(-(hp_r * hp_hl), -0.15, 0.15))
    coup = 0.3 * coup_t[tail].mean()
    lum = -0.4 * np.abs(lumP[tail] - 0.26).mean()
    total = sync + asym + coup + lum
    rng = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))
    print(f"  {name:24s} amps(joint) L {np.degrees(rng(shoL)):5.1f} R {np.degrees(rng(shoR)):5.1f}"
          f" | corr {corr(shoL, shoR):+.2f}"
          f" | lumP {np.degrees(lumP[tail].mean()):+5.1f}"
          f" | terms: s{sync:+.3f} a{asym:+.3f} c{coup:+.3f} l{lum:+.3f} => {total:+.3f}")
    for e in extra:
        print(f"  {'':24s} {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--refs", nargs="*", default=[])
    args = ap.parse_args()

    global build_model
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "sim2sim"))
    from sim2sim.mujoco_rollout import build_model as _bm  # noqa: E402
    build_model = _bm

    print("=== current-term scores (v34 statistics formula) ===")
    for ref in args.refs:
        p = ROOT / f"roboparty_train/robolab/data/motions/x1_lab_v31/{ref}.pkl"
        c = pickle.load(open(p, "rb"))
        q = np.asarray(c["dof_pos"], float)
        g = lambda n: q[:, LAB.index(n)]
        audit(f"REF {ref[:9]}",
              g("left_shoulder_pitch_joint"), g("right_shoulder_pitch_joint"),
              g("left_hip_pitch_joint"), g("right_hip_pitch_joint"), g("lumbar_pitch_joint"))

    for lp in args.logs:
        path = Path(lp)
        if not path.exists():
            print(f"  {lp}: MISSING"); continue
        d = np.load(path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        settle = int(meta.get("settle_steps", 0))
        hinge = list(meta["hinge_names"])
        qr = np.asarray(d["q"], float)[settle:]
        bp = np.asarray(d["base_pos"], float)[settle:]
        bq = np.asarray(d["base_quat"], float)[settle:]
        idx = {n: k for k, n in enumerate(hinge)}
        phiL, phiR = fk_phis(qr, bp, bq, hinge)
        extras = [f"world phi L {phiL.mean():+5.1f} R {phiR.mean():+5.1f} "
                  f"| shared {abs((phiL.mean()+phiR.mean())/2):4.1f} asym {abs(phiL.mean()-phiR.mean()):4.1f}"
                  f" | amp wL {np.percentile(phiL,95)-np.percentile(phiL,5):4.1f}"
                  f" wR {np.percentile(phiR,95)-np.percentile(phiR,5):4.1f}"
                  f" | y drift {bp[-1,1]:+.2f}m"]
        audit(path.stem, qr[:, idx["left_shoulder_pitch_joint"]], qr[:, idx["right_shoulder_pitch_joint"]],
              qr[:, idx["left_hip_pitch_joint"]], qr[:, idx["right_hip_pitch_joint"]],
              qr[:, idx["lumbar_pitch_joint"]], extras)


if __name__ == "__main__":
    main()
