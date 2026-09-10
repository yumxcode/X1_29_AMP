#!/usr/bin/env python3
"""Settle the arm-phase sign convention with FK ground truth (v31d diagnosis).

Question: in JOINT space (X1 anti-aligned shoulder-pitch axes), does NATURAL
human arm swing (alternating, world-frame antiphase) give positive or negative
corr(lsp, rsp)? And which entities actually alternate?

Measures per entity (reference clip 0005, v27/v29/v31d policy rollouts):
  world: corr(phiL, phiR), phi = upper-arm forward angle (FK, root frame)
         frac_opp = fraction of frames with phiL*phiR < 0 (natural: high)
         lag = phase lag of R vs L in percent of cycle (natural: ~50%)
  joint: corr(lsp, rsp) read from columns BY NAME
         swing range of each side (deg)

Then: natural-alternating entities define the joint-space sign convention.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list

import mujoco

XML = ROOT / "gmr_x1_assets/x1.xml"
YAML = ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
LAB = parse_yaml_list(YAML, "lab_dof_names")

model, _ = build_model(XML)
data = mujoco.MjData(model)
BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LAB}
lab_qadr = {n: model.jnt_qposadr[jid[n]] for n in LAB}
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
hinge_qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
              for n in hinge}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def phase_stats(x, y, dt):
    x0 = x - x.mean(); y0 = y - y.mean()
    n = len(x0)
    best_lag, best_v = 0, -2
    for lag in range(-n // 2, n // 2):
        v = np.corrcoef(np.roll(x0, lag), y0)[0, 1]
        if v > best_v:
            best_v, best_lag = v, lag
    # cycle via first strong autocorr peak of x
    ac = np.correlate(x0, x0, "full")[n - 1:]
    ac /= ac[0]
    pk = None
    for i in range(5, len(ac) - 1):
        if ac[i] > 0.5 and ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]:
            pk = i
            break
    period = pk if pk else n
    return best_lag, (abs(best_lag) / period * 100.0 if period else 0.0), period * dt


def fk_series(rp, rq_wxyz, q_by_name, name_map):
    T = len(rp)
    phiL, phiR = np.zeros(T), np.zeros(T)
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rq_wxyz[t] / np.linalg.norm(rq_wxyz[t])
        for n, val in q_by_name.items():
            data.qpos[name_map[n]] = val[t]
        mujoco.mj_forward(model, data)
        R = data.xquat[BID("base_link")]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, R); Rm = Rm.reshape(3, 3)
        fwd = Rm @ np.array([1.0, 0, 0])
        for side, arr in (("left", phiL), ("right", phiR)):
            S = data.xpos[BID(f"{side}_shoulder_pitch_link")]
            E = data.xpos[BID(f"{side}_elbow_pitch_link")]
            u = E - S
            arr[t] = np.degrees(np.arctan2(np.dot(u, fwd), -u[2]))
    return phiL, phiR


def measure(name, rp, rq, lsp, rsp, phiL, phiR, dt):
    c_world = corr(phiL, phiR)
    c_joint = corr(lsp, rsp)
    frac_opp = float(np.mean(phiL * phiR < 0))
    lag, lagpct, period = phase_stats(phiL, phiR, dt)
    rng = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))
    print(f"\n== {name} ==")
    print(f"  world: corr(phiL,phiR) = {c_world:+.2f} | frac opposite-sign = {frac_opp:.2f} "
          f"| lag(R vs L) = {lag} frames ({lagpct:.0f}% cycle, period {period:.2f}s)")
    print(f"  joint: corr(lsp,rsp) = {c_joint:+.2f} | swing L {rng(lsp):.1f} R {rng(rsp):.1f} deg "
          f"| means L {lsp.mean():+.1f} R {rsp.mean():+.1f}")
    return c_world, c_joint, frac_opp


# ---- reference 0005 (x1_lab_v31) -----------------------------------------
clip = pickle.load(open(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v31/0005_normal_walk1.pkl", "rb"))
rp = np.asarray(clip["root_pos"]); rq = np.asarray(clip["root_rot"])
q = np.asarray(clip["dof_pos"])
q_by = {n: q[:, i] for i, n in enumerate(LAB)}
phiL, phiR = fk_series(rp, rq, q_by, lab_qadr)
lsp = np.degrees(q[:, LAB.index("left_shoulder_pitch_joint")])
rsp = np.degrees(q[:, LAB.index("right_shoulder_pitch_joint")])
measure("REFERENCE x1_lab_v31/0005 (retargeted human, visually approved)",
        rp, rq, lsp, rsp, phiL, phiR, 1.0 / float(clip["fps"]))

# ---- policy rollouts ------------------------------------------------------
for label, path in [("v27 policy", "acceptance/v27_eval/walk05.npz"),
                    ("v29e4b policy", "acceptance/v29e4b_eval/walk05_seed1.npz"),
                    ("v31d policy", "/tmp/p7_v31d.npz")]:
    p = ROOT / path if not path.startswith("/") else Path(path)
    if not p.exists():
        print(f"\n== {label}: MISSING {p}")
        continue
    d = np.load(p, allow_pickle=True)
    import json as _json
    meta = _json.loads(str(d["meta"]))
    settle = int(meta.get("settle_steps", 0))
    qr = np.asarray(d["q"])[settle:]
    q_by = {n: qr[:, hinge.index(n)] for n in hinge}
    rp = np.asarray(d["base_pos"])[settle:]
    rq = np.asarray(d["base_quat"])[settle:]
    phiL, phiR = fk_series(rp, rq, q_by, hinge_qadr)
    lsp = np.degrees(qr[:, hinge.index("left_shoulder_pitch_joint")])
    rsp = np.degrees(qr[:, hinge.index("right_shoulder_pitch_joint")])
    measure(label, rp, rq, lsp, rsp, phiL, phiR, 0.02)
