# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""v70f: motion-dof <-> asset-joint NAME-level index map (isaaclab-free).

The retarget pipeline stores demo dof_pos in LAB_DOF_NAMES order (the
retarget config's column order), which is COMPLETELY different from the
Isaac asset joint order (verified 2026-10-15 against gmr_x1_assets/x1.xml
== Isaac order per the mujoco_rollout DEFAULT_Q equivalence):

    asset #18 left_hip_pitch  <->  lab #0  left_hip_pitch
    asset #4  left_shoulder_p <->  lab #10 left_shoulder_p ...
    (interleaved-legs lab order vs body-chain asset order)

The v70e/v70f first launches indexed the motion array with ASSET
indices (demo_joint_track paid exactly 0.0000; the reset event wrote
scrambled poses — v70e2's platform 8/13 degradation traces to this).

LAB_DOF_NAMES is fixed for this retarget lineage; test_v70f_track
cross-checks it against the retarget yaml so drift is caught locally.
"""
from __future__ import annotations

LAB_DOF_NAMES = [
"left_hip_pitch_joint",
    "lumbar_yaw_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "lumbar_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "lumbar_pitch_joint",
    "right_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "right_ankle_roll_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]


def motion_cols_for(joint_names) -> list[int]:
    """Indices into the motion dof array for the given (asset-named) joints."""
    return [LAB_DOF_NAMES.index(n) for n in joint_names]
