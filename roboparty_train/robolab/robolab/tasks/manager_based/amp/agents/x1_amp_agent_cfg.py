# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from robolab import ROBOLAB_ROOT_DIR


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training."""

    disc_obs_buffer_size: int = 1000
    grad_penalty_scale: float = 10.0
    disc_trunk_weight_decay: float = 1.0e-4
    disc_linear_weight_decay: float = 1.0e-2
    disc_learning_rate: float = 1.0e-5
    disc_max_grad_norm: float = 1.0

    @configclass
    class AMPDiscriminatorCfg:
        hidden_dims: list[int] = MISSING
        activation: str = "elu"
        style_reward_scale: float = 1.0
        task_style_lerp: float = 0.0

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticConv2d"
    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    conv_linear_output_size: int = 16


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOAmp"
    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()


@configclass
class X1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    """X1 AMP runner config — symmetry DISABLED (29 DOF mirror indices TBD)."""
    class_name = "AMPRunner"
    num_steps_per_env = 24
    # v18: 3000 -> 4000. v16/v17 curves plateaued (lin kernel 0.835-0.839);
    # extra 1000 iters to close the gap toward the 0.85 TARGET (AMP_ACCEPTANCE.md).
    max_iterations = 4000
    save_interval = 4000  # v22: only model_0 + final model_3999 land on disk —
    # keeps the SDK-visible .pt count minimal (v16-v21b evidence suggests a
    # 5-per-task registration quota; intermediate ckpts only eat slots)
    experiment_name = "x1_amp"
    wandb_project = "x1_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,  # DISABLED: X1 29-DOF mirror indices not yet implemented
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=1.5,
                # v21: 0.75 -> back to 0.6. v20 (lerp 0.75, 4000 iters) empirical:
                # lin kernel 0.8422 vs v16 (lerp 0.6, 3000 iters) 0.8391 — the
                # +0.003 gain is noise-level, but style reward halved
                # 0.335 -> 0.146 (below P5a=0.15 threshold). 0.6 passes BOTH
                # P3a (0.839 >= 0.82, expected to reach ~0.84 with 4000 iters)
                # and P5a (0.335 >> 0.15) with large margins.
                task_style_lerp=0.6
            ),
            loss_type="LSGAN"
        ),
    )
