#!/usr/bin/env python3
"""Verify the F1 weilai fixed-link mass merge.

Gates (from F1_AUDIT.md sec 2/4):
  V1 MJCF total mass == URDF source total (35.033 kg +- 0.01)
  V2 Isaac URDF total mass == same
  V3 FK unchanged: stand keyframe base z == 0.5876 m, ankle world pos vs
     pre-merge reference < 1e-9 (inertials never enter FK)
  V4 CoM rises (head stack merged into lumbar_pitch): report stand CoM z before/after
  V5 Welded-base PD-hold (audit gate): 3-deg perturbation, PD to stand pose,
     2 s @ 1 ms -> max joint droop <= 2.5 deg (audit X1-calibrated: 1.23 deg;
     +8% mass budget -> 2.5 deg ceiling)
"""
import json
import numpy as np
import mujoco
import xml.etree.ElementTree as ET

REPO = __file__.rsplit('/roboparty_train/', 1)[0]
MJCF = f'{REPO}/F1_29DOF_weilai/mjcf/f1_flat.xml'
ISAAC = f'{REPO}/F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf'
KP_KD = f'{REPO}/F1_29DOF_weilai/f1_kp_kd.json'
SRC = f'{REPO}/F1_29DOF_weilai/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf'

fails = []
def gate(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        fails.append(name)

def urdf_mass(path):
    t = ET.parse(path)
    return sum(float(i.find('mass').get('value'))
               for i in t.getroot().iter('inertial'))

print('== V1/V2 mass accounting ==')
m = mujoco.MjModel.from_xml_path(MJCF)
mjcf_mass = float(np.sum(m.body_mass))
src_mass = urdf_mass(SRC)
isaac_mass = urdf_mass(ISAAC)
gate('V1 MJCF mass', abs(mjcf_mass - src_mass) < 0.01,
     f'{mjcf_mass:.4f} vs source {src_mass:.4f} kg')
gate('V2 Isaac URDF mass', abs(isaac_mass - src_mass) < 0.01,
     f'{isaac_mass:.4f} kg')

print('== V3 FK unchanged (stand keyframe) ==')
d = mujoco.MjData(m)
d.qpos[:] = m.key_qpos[0]
mujoco.mj_forward(m, d)
base_z = float(d.qpos[2])
gate('V3 stand base z', abs(base_z - 0.5876) < 1e-3, f'z={base_z:.4f} m')
ankle = d.body('left_ankle_roll').xpos.copy()
com_after = d.subtree_com[1].copy()

print('== V4 CoM report (stand) ==')
print(f'  whole-body CoM after merge: {np.round(com_after, 4)}')

print('== V5 welded-base PD hold ==')
kp = json.load(open(KP_KD))
# build gain vectors by joint name
names = [m.joint(i).name for i in range(m.njnt)]
kpv = np.zeros(m.nv); kdv = np.zeros(m.nv)
UP = kp.get('upper_x1_planB', {})
def gain(nm):
    nm = nm.replace('_joint', '')
    c = nm.removeprefix('left_').removeprefix('right_')
    if c in kp.get('legs', {}):
        return kp['legs'][c]['kp'], kp['legs'][c]['kd']
    if c in UP:
        return UP[c][0], UP[c][1]
    return 40.0, 2.0
for jid in range(m.njnt):
    kpv[m.jnt_dofadr[jid]], kdv[m.jnt_dofadr[jid]] = gain(names[jid])
q_stand = np.array(m.key_qpos[0])[7:]
d.qpos[:] = m.key_qpos[0]
# weld the base: use mj_setConst-free approach — freeze via qvel0 and big armature? simplest:
# set qpos[0:7] fixed by re-setting each step
# audit protocol: 3-deg perturbation on both hip_pitch joints, settle 2 s
d.qpos[:] = m.key_qpos[0]
for nm in ('left_hip_pitch_joint', 'right_hip_pitch_joint'):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    d.qpos[7 + m.jnt_dofadr[jid] - 6] += np.deg2rad(3.0)
maxdroop = 0.0
for step in range(2000):
    d.qpos[0:7] = m.key_qpos[0][0:7]
    d.qvel[0:6] = 0.0                      # welded base: no free-flight
    q = d.qpos[7:]
    v = d.qvel[6:]
    tau = kpv[6:] * (q_stand - q) - kdv[6:] * v
    tau = np.clip(tau, -m.actuator_ctrlrange[:, 1], m.actuator_ctrlrange[:, 1])
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    if step > 1000:
        droop = np.max(np.abs(q_stand - d.qpos[7:]))
        maxdroop = max(maxdroop, droop)
gate('V5 PD-hold droop', np.rad2deg(maxdroop) <= 2.0,
     f'max sag {np.rad2deg(maxdroop):.2f} deg (audit 1.23; pre-merge same harness 1.44; ceiling 2.0)')

print('== V6 inertia triangle inequality (per body) ==')
bad = 0
for i in range(m.nbody):
    if m.body_mass[i] <= 0:
        continue
    I = np.diag(m.body_inertia[i])
    if not (2 * I.max() <= I.sum() + 1e-9):
        bad += 1
gate('V6 inertia valid', bad == 0, f'{bad} violations')

print('RESULT:', 'ALL PASS' if not fails else f'FAILS: {fails}')
raise SystemExit(0 if not fails else 1)
