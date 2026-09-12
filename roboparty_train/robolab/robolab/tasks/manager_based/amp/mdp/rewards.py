# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
    
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # return torch.exp(-lin_vel_error / std**2)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # return torch.exp(-ang_vel_error / std**2)
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def paired_joints_mean_deviation_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the shared offset of paired joints from their default positions.

    This keeps symmetric paired joints centered around their nominal pose while still allowing
    anti-phase motion, such as normal left/right arm swing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(torch.mean(joint_deviation, dim=1))


def paired_joints_deviation_difference_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None, max_cmd_yaw: float = 0.5,
) -> torch.Tensor:
    """Penalize the DIFFERENCE between paired-joint deviations (|dev_L - dev_R|).

    DO NOT USE for anti-aligned shoulder-pitch pairs on natural walking.
    v32 FK ground truth (acceptance/diag_arm_phase_truth.py, 2026-09-10):
    natural alternating arm swing measures corr(lsp, rsp) = -0.98 on the
    x1_lab_v31 references (world-frame antiphase, R lags L by 54% of cycle).
    This difference statistic is therefore LARGE on the natural pattern
    (dev_L ~ -dev_R) and penalizes it — enabling it (v28b..v31d, weight -0.3)
    flipped policies from antiphase (v27: -0.94) to SYNCHRONIZED arms
    (v28/v29/v31d: +0.96/+0.91/+0.78). The 'natural is same-sign' premise in
    the v28b note came from pre-fix CONTORTED data (corr(devL,devR)=+0.78 on
    v26 refs) and is invalid on clean references. Use
    paired_joints_deviation_sum_l1 instead; kept only for the future
    per-cycle amplitude comparison.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.abs(joint_deviation[:, 0] - joint_deviation[:, 1])
    if command_name is not None:
        # turning commands legitimately require L/R asymmetry — switch the
        # term off above a yaw-command threshold
        cmd = env.command_manager.get_command(command_name)
        gate = (torch.abs(cmd[:, 2]) < max_cmd_yaw).float()
        reward = reward * gate
    return reward


def paired_joints_deviation_sum_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DEPRECATED (v34) — DO NOT USE. Kept for history only.

    v32's "sum" statistic |dev_L + dev_R| was pitched as "~zero on natural
    alternation", but direct measurement (composite audit, 2026-09-11) shows
    the x1_lab_v31 REFERENCES score mean|sum| = 26.2 deg — dominated by a
    joint-space SHARED DC (both arms average -13 deg in joint space while
    being world-frame centered; the DC is an artifact of the shoulder
    yaw/roll chain, NOT a visible defect). At weight -0.3 the term taxed
    reference-like arms -0.137/step vs ~-0.005 for frozen arms — a net
    gradient TOWARD swing collapse (v33b arms fell to 4-8 deg amplitude).
    Superseded by arm_sync_residual (high-passed, DC-free).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(torch.sum(joint_deviation, dim=1))


_EMA_STATE: dict = {}


def _ema(env, key: str, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-env single-channel CONTINUOUS exponential moving average.

    v34 lesson: do NOT reseed per episode. The first v34 draft reseeded at
    every episode boundary; in the 20 s training episodes the antiphase
    swing difference (AC amplitude ~88 deg) then re-injected a fresh
    initial transient every episode, keeping |EMA| at ~30 deg for the first
    ~tau seconds of EVERY episode — polluting the frozen-offset guards far
    more than the offsets they exist to catch. Continuous tracking
    converges once (~5 tau after training start) and stays converged: a
    genuinely frozen offset is smoothed in, the swing AC is attenuated by
    alpha/(2*pi*f*tau). State keyed by (id(env), key); several terms share
    channels (same high-passed shoulder signal for sync + coupling).
    """
    skey = (id(env), key)
    ema = _EMA_STATE.get(skey)
    if ema is None or ema.shape != x.shape or ema.device != x.device:
        ema = x.detach().clone()
        _EMA_STATE[skey] = ema
    with torch.no_grad():
        ema.mul_(1.0 - alpha).add_(x.detach(), alpha=alpha)
    return ema


def paired_joints_deviation_difference_ema_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 0.0025,
) -> torch.Tensor:
    """Frozen antisymmetric-offset guard on a SLOW EMA of (dev_L - dev_R).

    v32b defect (diag_arm_offset.py): L shoulder held -9 deg / R +9.8 deg —
    a frozen antisymmetric offset. v33 shipped this EMA at alpha=0.02
    (tau = 1 s); the composite audit found that TOO FAST: for natural
    antiphase swing the DIFFERENCE carries the full AC with amplitude
    (A_L + A_R) (~88 deg on refs), and a 1 s EMA only attenuates it ~5x —
    the reference itself scored |EMA| = 28 deg -> -0.0998/step, i.e. the
    guard punished antiphase SWING AMPLITUDE, not just frozen offsets.
    v34: alpha = 0.0025 (tau = 8 s) attenuates the ~1 Hz swing ~50x
    (leak ~2 deg) while a frozen offset still converges in ~30 s. Measured
    separation after fix: ref ~5 deg (natural small asym, mild tax),
    v32b-style frozen offset 18.8 deg (caught).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dc = _ema(env, "arm_diff_dc", dev[:, 0] - dev[:, 1], alpha)
    return torch.abs(dc)


def arm_sync_residual(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 0.02,
) -> torch.Tensor:
    """In-phase swing detector on HIGH-PASSED residuals: |hpL + hpR|.

    v34 replacement for the deprecated sum statistic. hp = dev - EMA(dev)
    removes each arm's slow component (frozen offsets, shared DC), so the
    residual is the swing itself. For natural antiphase swing hpL ~ -hpR
    -> |hpL + hpR| ~ 0 at ANY amplitude; for synchronized (in-phase) swing
    hpL ~ +hpR -> |hpL + hpR| ~ 2A. Measured: ref 0.045 rad (amplitude-
    mismatch residual only), v31d-style sync would score ~8-10x larger.
    Unlike the old sum, this term has NO shared-DC sensitivity (ref sum_dc
    -26 deg no longer taxed) and NO direct amplitude force.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    hp_l = dev[:, 0] - _ema(env, "sho_L", dev[:, 0], alpha)
    hp_r = dev[:, 1] - _ema(env, "sho_R", dev[:, 1], alpha)
    return torch.abs(hp_l + hp_r)


def arm_opposite_leg_coupling(
    env: ManagerBasedRLEnv,
    shoulder_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg,
    cap: float = 0.15, alpha: float = 0.02,
) -> torch.Tensor:
    """Reward coherent arm/opposite-leg swing via HIGH-PASSED product.

    Joint-space sign structure (measured on refs, _tmp_coup_debug):
      corr(hp_Lsho, hp_Rhip) = +0.98  -> product POSITIVE on natural swing
      corr(hp_Rsho, hp_Lhip) = -0.98  -> product NEGATIVE on natural swing!
    The second pair carries the opposite joint-space sign (shoulder pair is
    same-sign convention; hip pair anti-aligned), so it is NEGATED in the
    reward. v33 shipped WITHOUT the negation: on the reference the two raw
    products (+0.105 / -0.105) cancelled to ~zero (composite audit) and the
    term actively penalized the natural R-arm/L-leg coherence. Fixed here:
      term = 0.5*(clamp(hp_sl*hp_hr) + clamp(-(hp_sr*hp_hl)))
    Reference now scores +0.105 per pair (coup +0.031/step at weight 0.3),
    collapsed-arm policies ~+0.001 -> small anti-collapse gradient toward
    natural coordinated swing. High-pass hp = x - EMA(x) keeps it mean-shift
    free; cap 0.15 rad^2 still guards against amplitude farming.
    """
    asset: Articulation = env.scene[shoulder_cfg.name]
    sho = asset.data.joint_pos[:, shoulder_cfg.joint_ids] - asset.data.default_joint_pos[:, shoulder_cfg.joint_ids]
    hip = asset.data.joint_pos[:, hip_cfg.joint_ids] - asset.data.default_joint_pos[:, hip_cfg.joint_ids]
    hp_sl = sho[:, 0] - _ema(env, "sho_L", sho[:, 0], alpha)
    hp_sr = sho[:, 1] - _ema(env, "sho_R", sho[:, 1], alpha)
    hp_hr = hip[:, 0] - _ema(env, "hip_R", hip[:, 0], alpha)   # col 0 = R hip
    hp_hl = hip[:, 1] - _ema(env, "hip_L", hip[:, 1], alpha)   # col 1 = L hip
    term = 0.5 * (torch.clamp(hp_sl * hp_hr, -cap, cap)
                  + torch.clamp(-(hp_sr * hp_hl), -cap, cap))
    return term


def lumbar_pitch_prior(
    env: ManagerBasedRLEnv, target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lumbar pitch deviation from the reference posture mean.

    FK probe (acceptance/probe_lumbar_sign.py): POSITIVE lumbar_pitch =
    chest FORWARD. All 11 v31 references lean forward, mean +12..+20 deg
    (median +14.9 = 0.26 rad) — natural human walking posture. v32b policy
    sat at -9.0 deg (chest BACKWARD, user-observed '胸腔后倾'); v31d at
    +26.2 (over-forward). Policies drift freely without a prior because no
    term references the torso pitch posture (flat_orientation acts on the
    BASE only, not the chest).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lum = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(lum[:, 0] - target)


def yaw_rate_bias_guard(
    env: ManagerBasedRLEnv, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 0.005,
) -> torch.Tensor:
    """Penalize a SLOW (DC) mismatch between base yaw rate and the yaw command.

    v34 sim2sim diagnosis (acceptance/diag_drift.py): body-frame tracking is
    fine (vx 0.89 / vy ~0) but the policy carries a CONSTANT +4.6 deg/s yaw
    bias on zero-yaw commands -> 55 deg of heading drift and +3.5 m lateral
    displacement over 12 s. The exp kernel (std 0.5 rad/s) is insensitive to
    0.08 rad/s (kernel 0.975), so nothing corrects it. An EMA of the
    ANG-VEL ERROR extracts exactly the DC bias: real turns (commanded AC or
    matched cmd) average out, a persistent mismatch converges in ~1/alpha
    steps and is taxed. Reference walks are straight (net heading ~0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2]
    err = asset.data.root_ang_vel_b[:, 2] - cmd_z
    dc = _ema(env, "yaw_bias", err, alpha)
    return torch.abs(dc)


def stance_sole_flat_walk(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_cmd_speed: float = 1.5,
) -> torch.Tensor:
    """Penalize a non-flat sole while the foot is in contact, at walking speeds.

    v27 strict sim2sim criteria: 落地要平稳顺滑, 不允许翘着脚面行走 (no
    toe-walking / ball-foot gait). v26 policies at cmd >= 1.0 m/s stood on the
    forefoot (heel raised 19% of mid-stance; sole pitch dipping to -42 deg),
    while the retargeted AMASS references keep the sole flat through stance.

    NOTE on frames: the built-in feet_orientation_l2 assumes the sole normal
    is the body-frame z axis. On X1 the ankle_roll link frame is rotated 90
    deg about y (vendor URDF rpy=(0, 1.5708, 0), identical in the MJCF): the
    sole occupies local y < 0, toe = +z_local, so the sole-normal unit vector
    in the body frame is (0, 1, 0) — using z there would penalize FLAT feet.
    We rotate the explicit up vector to world and penalize sin^2(tilt).

    Gated to walking speeds: at >= 1.5 m/s (jog references in the AMP dataset)
    a forefoot stance is natural and must not be punished.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    num_feet = len(sensor_cfg.body_ids)

    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]              # (N, M, 4)
    # v27.3c shape fix (task 248 died in learn(): 8192 vs 2 broadcast): the
    # flat quats are (N*M, 4), so the up vector must be expanded to (N*M, 3)
    flat_q = feet_quat.reshape(-1, 4)
    up_row = torch.tensor([0.0, 1.0, 0.0], device=env.device)                 # sole normal, foot frame
    up_world = math_utils.quat_apply(
        flat_q, up_row.unsqueeze(0).expand(flat_q.shape[0], 3)
    ).reshape(-1, num_feet, 3)
    cos_tilt = torch.clamp(up_world[:, :, 2], -1.0, 1.0)                      # dot with world up
    sin_sq = 1.0 - torch.square(cos_tilt)

    cmd = env.command_manager.get_command(command_name)
    gate = (torch.norm(cmd[:, :2], dim=1) < max_cmd_speed).float()
    return torch.sum(sin_sq * in_contact, dim=-1) * gate


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Only reward air time exceeding the threshold (prevents negative penalties)
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(positive_air * first_contact.float(), dim=1)
    # no reward for zero command
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Compute per-foot acceleration penalty for sound suppression.

    Penalize large vertical (z) accelerations when a foot is in contact with the ground.
    """

    asset = env.scene["robot"]

    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    acc_penalty = (foot_acc_z ** 2) * in_contact.float()
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)