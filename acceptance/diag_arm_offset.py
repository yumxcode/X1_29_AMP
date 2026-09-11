#!/usr/bin/env python3
"""Diagnose v32b user-observed arm shape defects (2026-09-10):
1) swing amplitude too small  2) both arms reach FORWARD (shared lean)
3) torso/chest leans BACK.

Measures, for reference (x1_lab_v31/0005) vs policy rollouts (v27/v31d/v32b):
  per-side world phi (upper-arm forward angle): mean = shared lean,
  p95-p5 = amplitude, corr = phase
  joint lsp/rsp means + FK SIGN PROBE for BOTH shoulders (sets shoP=+10deg,
  reads world phi change — settles the encoding convention empirically)
  torso: lumbar_pitch_joint mean (root-relative), root pitch (world tilt
  from base quat), for backward-lean quantification
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

import mujoco  # noqa: E402

model, _ = build_model(ROOT / "gmr_x1_assets/x1.xml")
data = mujoco.MjData(model)
BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
LAB = parse_yaml_list(ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml", "lab_dof_names")
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
hinge_qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
              for n in hinge}
LAB_QADR = {n: hinge_qadr[n] for n in LAB}

# ---------- FK sign probes (both shoulders, zero-pose base) ----------
print("=== FK sign probes (shoP = +10 deg from zero pose, world phi of upper arm) ===")
for side in ("left", "right"):
    data.qpos[:] = 0
    data.qpos[0:3] = [0, 0, 0.8]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    def phi():
        S = data.xpos[BID(f"{side}_shoulder_pitch_link")]
        E = data.xpos[BID(f"{side}_elbow_pitch_link")]
        u = E - S
        return np.degrees(np.arctan2(u[0], -u[2]))
    p0 = phi()
    data.qpos[hinge_qadr[f"{side}_shoulder_pitch_joint"]] = np.radians(10)
    mujoco.mj_forward(model, data)
    p1 = phi()
    print(f"  {side}: shoP 0->+10deg : world phi {p0:+.1f} -> {p1:+.1f} "
          f"({'FORWARD' if p1 > p0 else 'BACKWARD'} by {p1-p0:+.1f} deg)")


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def measure(name, rp, rq, qcols):
    """qcols: dict joint_name -> np.array(T) in RADIANS."""
    T = len(rp)
    phiL = np.zeros(T); phiR = np.zeros(T); root_pitch = np.zeros(T)
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rq[t] / np.linalg.norm(rq[t])
        for n, arr in qcols.items():
            data.qpos[LAB_QADR[n]] = arr[t]
        mujoco.mj_forward(model, data)
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, data.xquat[BID("base_link")])
        Rm = Rm.reshape(3, 3)
        fwd = Rm[:, 0]
        for side, arr in (("left", phiL), ("right", phiR)):
            u = data.xpos[BID(f"{side}_elbow_pitch_link")] - \
                data.xpos[BID(f"{side}_shoulder_pitch_link")]
            arr[t] = np.degrees(np.arctan2(u @ fwd, -(u @ Rm[:, 2])))
        # root pitch: + = leaning back (nose up)
        root_pitch[t] = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    lumP = np.degrees(qcols["lumbar_pitch_joint"])
    rng = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5))
    print(f"\n== {name} ==")
    print(f"  world phiL: mean {phiL.mean():+6.1f}  amp {rng(phiL):5.1f} | "
          f"phiR: mean {phiR.mean():+6.1f}  amp {rng(phiR):5.1f} | corr {corr(phiL, phiR):+.2f}")
    lsp = np.degrees(qcols["left_shoulder_pitch_joint"])
    rsp = np.degrees(qcols["right_shoulder_pitch_joint"])
    print(f"  joint lsp : mean {lsp.mean():+6.1f}  amp {rng(lsp):5.1f} | "
          f"rsp : mean {rsp.mean():+6.1f}  amp {rng(rsp):5.1f} | corr {corr(lsp, rsp):+.2f} | "
          f"|devL+devR| mean {np.abs(np.radians(lsp)+np.radians(rsp)).mean():.2f} rad")
    print(f"  lumbar_pitch: mean {lumP.mean():+6.1f}  range {rng(lumP):5.1f} | "
          f"root pitch: mean {root_pitch.mean():+6.1f} (+ = back)")
    return phiL, phiR


# ---------- reference ----------
c = pickle.load(open(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v31/0005_normal_walk1.pkl", "rb"))
q = np.asarray(c["dof_pos"], float)
measure("REFERENCE x1_lab_v31/0005",
        np.asarray(c["root_pos"], float), np.asarray(c["root_rot"], float),
        {n: q[:, i] for i, n in enumerate(LAB)})

# ---------- policies ----------
for label, path in [("v27 policy", ROOT / "acceptance/v27_eval/walk05.npz"),
                    ("v31d policy", Path("/tmp/p7_v31d.npz")),
                    ("v32b policy", Path("/tmp/p7_v32b.npz"))]:
    if not path.exists():
        print(f"\n== {label}: missing"); continue
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    settle = int(meta.get("settle_steps", 0))
    qr = np.asarray(d["q"], float)[settle:]
    measure(label, np.asarray(d["base_pos"], float)[settle:],
            np.asarray(d["base_quat"], float)[settle:],
            {n: qr[:, i] for i, n in enumerate(hinge)})
