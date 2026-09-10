#!/usr/bin/env python3
"""Ground-truth FK probe of X1 arm/lumbar joint semantics (zero pose + single-joint perturbation)."""
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sim2sim'))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list

model, _ = build_model(ROOT / 'gmr_x1_assets' / 'x1.xml')
data = mujoco.MjData(model)
lab = parse_yaml_list(ROOT / 'roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml', 'lab_dof_names')
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in hinge}

BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)


def fk_at(qpos_overrides, root_pos=(0, 0, 0.8), root_quat=(1, 0, 0, 0)):
    data.qpos[:] = 0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    for n, v in qpos_overrides.items():
        data.qpos[qadr[n]] = v
    mujoco.mj_forward(model, data)
    P = {b: data.xpos[BID(b)].copy() for b in
         ['left_shoulder_pitch_link', 'left_elbow_yaw_link', 'left_wrist_roll_link',
          'right_shoulder_pitch_link', 'right_elbow_yaw_link', 'right_wrist_roll_link']}
    return P


def arm_stats(P, s):
    S, E, W = P[f'{s}_shoulder_pitch_link'], P[f'{s}_elbow_yaw_link'], P[f'{s}_wrist_roll_link']
    u1, u2 = E - S, W - E
    bend = np.degrees(np.arccos(np.clip(np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2), -1, 1)))
    swing_u1 = np.degrees(np.arctan2(u1[0], -u1[2]))   # + = upper arm forward
    swing_u2 = np.degrees(np.arctan2(u2[0], -u2[2]))   # + = forearm forward
    lat_u1 = np.degrees(np.arctan2(u1[1] * (1 if s == 'left' else -1), -u1[2]))  # + = outward
    return bend, swing_u1, swing_u2, lat_u1, S, E, W


PROBES = ['lumbar_yaw_joint', 'lumbar_roll_joint', 'lumbar_pitch_joint',
          'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
          'left_elbow_pitch_joint', 'left_elbow_yaw_joint',
          'left_wrist_pitch_joint', 'left_wrist_roll_joint']

P0 = fk_at({})
for s in ['left', 'right']:
    bend, sw1, sw2, lat1, S, E, W = arm_stats(P0, s)
    print(f"ZERO POSE {s}: elbow bend {bend:5.1f}deg | upper arm fwd {sw1:+6.1f} lateral {lat1:+6.1f} | forearm fwd {sw2:+6.1f}")
print()
print(f"{'joint @ +30deg':30s} | {'L bend':>7s} {'L upF':>6s} {'L foreF':>7s} {'L upLat':>7s} | {'dElbow mm':>9s} {'dWrist mm':>9s}")
for jn in PROBES:
    P1 = fk_at({jn: np.radians(30)})
    bend, sw1, sw2, lat1, S, E, W = arm_stats(P1, 'left')
    dE = np.linalg.norm(E - P0['left_elbow_yaw_link']) * 1000
    dW = np.linalg.norm(W - P0['left_wrist_roll_link']) * 1000
    print(f"{jn:30s} | {bend:7.1f} {sw1:+6.1f} {sw2:+7.1f} {lat1:+7.1f} | {dE:9.1f} {dW:9.1f}")
print()
print("Ranges (deg):", {n: np.degrees(model.jnt_range[model.joint(n).id]).tolist()
                        for n in ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                                  'left_shoulder_yaw_joint', 'left_elbow_pitch_joint',
                                  'left_elbow_yaw_joint', 'lumbar_yaw_joint']})
