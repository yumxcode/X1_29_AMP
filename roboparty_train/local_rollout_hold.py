#!/usr/bin/env python3
"""Local (mujoco 3.1.1) emulation of the deployment rollout loop:
build_model floor + v26 frictionloss zeroing + v25 auto-calibrated spawn +
pd-every-5 substep loop. Holds q_default (zero action) for 6 s — a robot
whose standing pose is sound must stay up (X1 precedent: champion policies
settle cleanly)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim2sim"))
import numpy as np
import mujoco
from mujoco_rollout import PD, F1_DEFAULT_Q, DEFAULT_Q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLOOR = ('  <geom name="floor" type="plane" size="20 20 0.1" '
         'friction="1 0.005 0.0001" contype="1" conaffinity="1" '
         'rgba="0.35 0.45 0.5 1"/>')


def rollout_hold(xml_path, qtab, z_start, label, seconds=6.0):
    src = open(xml_path).read()
    src = src.replace("</worldbody>", FLOOR + "\n</worldbody>")
    tmp = os.path.join(os.path.dirname(xml_path), "_floor_test.xml")
    open(tmp, "w").write(src)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    m.dof_frictionloss[:] = 0.0
    # full v26 build_model alignment: damping 1.0 + armature 0.01 on every
    # hinge (Isaac URDF semantics; vendor 2.0 damping WITHOUT armature makes
    # the PD loop mismatch and topple — verified locally for BOTH robots)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            m.dof_damping[m.jnt_dofadr[j]] = 1.0
            m.dof_armature[m.jnt_dofadr[j]] = 0.01
    d = mujoco.MjData(m)
    names = [m.joint(i).name for i in range(m.njnt)]
    hinge = [n for n in names if n in qtab]
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in hinge])
    vadr = np.array([m.jnt_dofadr[jid[n]] for n in hinge])
    kp = np.array([PD[n][0] for n in hinge])
    kd = np.array([PD[n][1] for n in hinge])
    eff = np.array([PD[n][2] for n in hinge])
    q_t = np.array([qtab[n] for n in hinge])

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, z_start]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[qadr] = q_t
    mujoco.mj_forward(m, d)
    if d.ncon:
        worst = min(float(d.contact[i].dist) for i in range(d.ncon))
        d.qpos[2] -= worst + 0.002
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print(f"[{label}] spawn z={d.qpos[2]:.4f} sole contacts={d.ncon}")

    substeps = 20
    tau = np.zeros(len(hinge))
    for step in range(int(seconds / 0.02)):
        for s in range(substeps):
            if s % 5 == 0:
                tau = np.clip(kp * (q_t - d.qpos[qadr]) - kd * d.qvel[vadr], -eff, eff)
            d.qfrc_applied[vadr] = tau
            mujoco.mj_step(m, d)
    tilt = np.degrees(np.arccos(np.clip(d.qpos[6], -1, 1)))
    fn = [d.cfrc_ext[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{sd}_ankle_roll_link")][2]
          for sd in ("left", "right")]
    print(f"[{label}] {seconds:.0f}s hold: base z {d.qpos[2]:.4f} tilt {tilt:.2f}deg "
          f"ncon {d.ncon} footF {[f'{v:.0f}N' for v in fn]}")


if __name__ == "__main__":
    rollout_hold(os.path.join(ROOT, "gmr_x1_assets/x1.xml"), DEFAULT_Q, 0.55, "X1")
    rollout_hold(os.path.join(ROOT, "gmr_f1_assets/f1.xml"), F1_DEFAULT_Q, 0.50, "F1")
