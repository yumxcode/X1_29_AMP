"""Verify conventions of BOTH formats against authoritative x1.yaml.

gmr: root_rot should be xyzw (err==0 vs stored body_positions, root-relative)
lab: root_rot wxyz + dof in lab_dof_names order (verify vs key_body_pos, root-relative)
"""
import pickle

import mujoco
import numpy as np
GMR = ['lumbar_yaw_joint', 'lumbar_roll_joint', 'lumbar_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
LAB = ['left_hip_pitch_joint', 'lumbar_yaw_joint', 'right_hip_pitch_joint', 'left_hip_roll_joint', 'lumbar_roll_joint', 'right_hip_roll_joint', 'left_hip_yaw_joint', 'lumbar_pitch_joint', 'right_hip_yaw_joint', 'left_knee_pitch_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'right_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'right_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'right_ankle_roll_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_elbow_yaw_joint', 'right_elbow_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']
KEY = ['left_knee_pitch_link', 'right_knee_pitch_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_elbow_yaw_link', 'right_elbow_yaw_link']






model = mujoco.MjModel.from_xml_path('X1_29DOF/mjcf/xyber_x1_flat.xml')
data = mujoco.MjData(model)
qadr = [model.jnt_qposadr[i] for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def fk(rp, quat_wxyz, dof_mjcf_order):
    data.qpos[:] = 0
    data.qpos[:3] = rp
    data.qpos[3:7] = quat_wxyz
    for a, v in zip(qadr, dof_mjcf_order):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    return {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}


# ---- lab: verify dof order (yaml LAB -> mjcf) + quat wxyz via key_body_pos ----
m = pickle.load(open('acceptance/v25_unpacked/x1_lab/0026_circle_walk.pkl', 'rb'))
rp, rr, dl = np.asarray(m['root_pos']), np.asarray(m['root_rot']), np.asarray(m['dof_pos'])
kb = np.asarray(m['key_body_pos'])  # (T,6,3) in KEY order
dof_to_mjcf = dl[:, [LAB.index(n) for n in GMR]]
T = rp.shape[0]
frames = np.linspace(0, T - 1, 20).astype(int)

for cname, q in [('wxyz', lambda x: x), ('xyzw', lambda x: x[[3, 0, 1, 2]])]:
    errs = []
    for t in frames:
        b = fk(rp[t], q(rr[t] / np.linalg.norm(rr[t])), dof_to_mjcf[t])
        root = b['base_link']
        rel_mj = np.array([b[n] - root for n in KEY])
        rel_kb = kb[t] - rp[t]  # key bodies relative to root pos
        errs.append(np.linalg.norm(rel_mj - rel_kb, axis=1).mean())
    print(f"lab dof=yaml-order, quat={cname:4s}: key-body root-rel err = {np.mean(errs):.4f} m")

# ---- gmr: re-confirm xyzw with full 20-frame sample ----
m = pickle.load(open('acceptance/v25_unpacked/x1_gmr/0026_circle_walk.pkl', 'rb'))
rp, rr, dg = np.asarray(m['root_pos']), np.asarray(m['root_rot']), np.asarray(m['dof_pos'])
bp, bn = np.asarray(m['body_positions']), list(m['body_names'])
for cname, q in [('wxyz', lambda x: x), ('xyzw', lambda x: x[[3, 0, 1, 2]])]:
    errs = []
    for t in frames:
        b = fk(rp[t], q(rr[t] / np.linalg.norm(rr[t])), dg[t])
        root = b['base_link']
        rel_mj = np.array([b[n] - root for n in bn])
        rel_bp = bp[t] - bp[t, 0]
        errs.append(np.linalg.norm(rel_mj - rel_bp, axis=1).mean())
    print(f"gmr quat={cname:4s}: all-body root-rel err = {np.mean(errs):.4f} m")
