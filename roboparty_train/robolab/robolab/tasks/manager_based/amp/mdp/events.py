# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""v70e: DEMO-STATE phase-locked episode initialization (rhythm route 2).

16 prior falsifications (v64/65/66 + v70c1/c2/d) established that the
~1Hz human-cadence basin is not reachable by reward shaping, demo
RE-SAMPLING (VMATCH), soft action filtering or delay co-training from a
policy that always STARTS episodes from the standing default pose: the
on-policy state distribution never includes human-cadence walking states,
so the action function is never trained inside that basin.

This event changes the SAMPLING DISTRIBUTION directly: with probability
p_demo, a walking-command env's episode starts ON a retargeted demo frame
(same motion id + fetch time the discriminator will compare against —
phase-locked by construction): root pose (demo height + orientation at the
env origin), root velocities, joint positions and velocities. PPO then
trains the action function along human-cadence trajectory segments every
episode; the posture guards hold the arm/leg form while the state resets
hold the rhythm.

Registered in x1_amp_env_cfg via X1_DEMO_RESET_P (default 0 = off, every
earlier recipe byte-identical). Runs at mode="reset" AFTER the default
reset_base/reset_robot_joints terms (event terms execute in definition
order; this one is appended last in the X1 post-init).
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def reset_robot_from_demo(
    env,
    env_ids,
    p_demo: float = 0.5,
    min_cmd_speed: float = 0.3,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg | None = None,
):
    """Episode-start state reset onto a retargeted demo frame.

    Selects envs whose commanded walking speed exceeds min_cmd_speed with
    probability p_demo, fetches the animation manager's freshly-sampled
    (motion_id, fetch_time) for those envs — the SAME segment the disc
    will compare this episode against — and writes the demo frame's full
    robot state into the sim (root: demo z + quat + velocities at the env
    origin xy; joints: demo pos/vel).
    """
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    if len(env_ids) == 0:
        return
    env_ids = env_ids.to(asset.device)
    cmd = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(cmd[env_ids, :2], dim=1)
    sel_mask = (speed > min_cmd_speed) & (
        torch.rand(len(env_ids), device=asset.device) < p_demo)
    sel = env_ids[sel_mask]
    if sel.numel() == 0:
        return

    anim = env.animation_manager
    mids = anim.motion_ids[sel]
    mtimes = anim.motion_fetch_time[sel, 0]
    st = anim.motion_data_term.get_motion_state(mids, mtimes)

    # root state: demo height/orientation/velocities, xy at the env origin
    # (pure translation of the demo frame — dynamics- and phase-preserving)
    root = asset.data.default_root_state[sel].clone()
    root[:, 0] = env.scene.env_origins[sel, 0]
    root[:, 1] = env.scene.env_origins[sel, 1]
    root[:, 2] = st["root_pos_w"][:, 2]
    root[:, 3:7] = st["root_quat"]
    root[:, 7:10] = st["root_vel_w"]
    root[:, 10:13] = st["root_ang_vel_w"]
    asset.write_root_state_to_sim(root, env_ids=sel)
    asset.write_joint_state_to_sim(st["dof_pos"], st["dof_vel"], env_ids=sel)
    print(f"[v70e] demo-state reset applied to {sel.numel()}/{len(env_ids)} envs "
          f"(p={p_demo:.2f}, cmd>{min_cmd_speed} m/s)")
