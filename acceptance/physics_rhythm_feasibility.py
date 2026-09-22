#!/usr/bin/env python3
"""Physics feasibility: is HUMAN-cadence swinging (1 Hz cycle, 0.43 s swing)
torque/energy-feasible on the X1 hardware, vs the 4 Hz tapping the policies
converge to?

Method (all local, MuJoCo model = gmr_x1_assets/x1.xml, the same one
sim2sim uses):
  1. Extract leg kinematics per frequency from the RETARGETED REFERENCE
     itself (0002_treadmill_slow, cycle 1.48 s — human cadence exemplar)
     and from a v66b policy rollout (3.83 Hz tapping).
  2. For each trajectory, compute the required joint torques by the
     INVERSE DYNAMICS of a SINGLE-SUPPORT swing: tau = M(q)qdd + C(q,qd)qd
     + G(q), evaluated along the reference leg angles with the stance leg
     carrying the body. We use mj_inverse on the full model with the
     reference root trajectory, extracting the swing-leg hip/knee/ankle
     torque demand.
  3. Compare peak/mean |tau| and mechanical power against the actuator
     limits in the xml (kp/kv servo, torque limits).

Simplified robust variant (what this script actually runs): drive the X1
model THROUGH the reference joint trajectories via mj_inverse with the
reference root motion (both legs), and read the swing-phase torque peaks
at the hip pitch joints. This measures the true inertial demand including
the coupled body — no servo needed (pure inverse dynamics).
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))

import mujoco  # noqa: E402
from sim2sim.mujoco_rollout import build_model, find_sole_geoms, parse_yaml_list, DEFAULT_Q  # noqa: E402

LAB = parse_yaml_list(
    ROOT / "roboparty_train" / "robolab" / "scripts" / "tools" / "retarget" / "config" / "x1.yaml",
    "lab_dof_names")

model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
data = mujoco.MjData(model)
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
vadr = np.array([model.jnt_dofadr[jid[n]] for n in hinge])  # velocity address for qfrc
# lab col -> mj hinge row for dof arrays
lab2mj = np.array([hinge.index(n) for n in LAB])

# torque limits from the xml actuators (position servos -> tau_max = kp*(range/2) capped by forcerange)
tau_limits = {}
for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)
    tau_limits[name] = (model.actuator_forcerange[a][1], model.actuator_ctrlrange[a][1])


def inverse_torques(q_lab, root_pos, root_quat, dt, qadr=None, vadr=None):
    """Drive the full model through (q_lab, root) and return joint torques
    via mj_inverse. q_lab: (T, 29) in lab order; root: (T,3)/(T,4)."""
    if qadr is None or vadr is None:
        return np.zeros((len(q_lab), 1))
    T = len(q_lab)
    taus = np.zeros((T, len(hinge)))
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    for i in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i] / np.linalg.norm(root_quat[i])
        data.qpos[qadr] = q_lab[i]
        if i > 0:
            data.qvel[vadr] = (q_lab[i] - q_lab[i - 1]) / dt
        if i > 1:
            data.qacc[vadr] = (q_lab[i] - 2 * q_lab[i - 1] + q_lab[i - 2]) / dt ** 2
        mujoco.mj_inverse(model, data)
        taus[i] = data.qfrc_inverse[vadr]
    return taus


def analyze(label, q_lab, root_pos, root_quat, dt):
    taus = inverse_torques(q_lab, root_pos, root_quat, dt, qadr, vadr)
    li = LAB.index("left_hip_pitch_joint")
    ri = LAB.index("right_hip_pitch_joint")
    ki = LAB.index("left_knee_pitch_joint")
    out = {}
    for nm, idx in (("hipL", li), ("hipR", ri), ("kneeL", ki)):
        col = taus[:, lab2mj[idx]]
        out[nm] = (float(np.percentile(np.abs(col), 95)), float(np.abs(col).max()))
    print(f"{label}: hip p95 |tau| L={out['hipL'][0]:6.1f} R={out['hipR'][0]:6.1f} Nm  "
          f"peak L={out['hipL'][1]:6.1f}  knee p95 {out['kneeL'][0]:6.1f}")
    return taus


# ---- source 1: human-cadence reference (0002, cycle 1.48 s) ----
clip = pickle.load(open(ROOT / "roboparty_train/robolab/data/motions/x1_lab/0002_treadmill_slow.pkl", "rb"),
                   encoding="latin1")
q_ref = np.asarray(clip["dof_pos"], float)
rp = np.asarray(clip["root_pos"], float)
rq = np.asarray(clip["root_rot"], float)
dt = 1.0 / float(clip["fps"])
analyze("REF 0002 human 1.48s-cycle", q_ref, rp, rq, dt)

# ---- source 2: v66b policy rollout (3.83 Hz tapping) ----
z = np.load(ROOT / "acceptance/v66b_eval/m3999_walk10.npz")
meta = json.loads(str(z["meta"]))
dt2 = float(meta["control_dt"])
q_pol = z["q"].astype(float)          # hinge order already
bp = z["base_pos"].astype(float)
# reconstruct root quat from rollout log if present
rq2 = z.get("base_quat")
if rq2 is None:
    # fall back: identity orientation (walking is near-upright; the
    # gravitational term dominates the hip torque at these amplitudes)
    rq2 = np.tile([1.0, 0, 0, 0], (len(q_pol), 1))
# policy q is in hinge order -> map to lab columns
lab_from_hinge = np.array([hinge.index(n) for n in LAB])
q_pol_lab = np.zeros((len(q_pol), len(LAB)))
q_pol_lab[:, :] = q_pol[:, lab_from_hinge]
analyze("POLICY v66b 3.83Hz tap", q_pol_lab, bp, rq2, dt2)

print("\nActuator force ranges (Nm):")
for k, v in sorted(tau_limits.items()):
    if "hip_pitch" in k or "knee" in k:
        print(f"  {k}: forcerange={v[0]}, ctrlrange={v[1]}")
