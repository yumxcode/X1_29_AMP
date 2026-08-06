import os
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg
from robolab.assets.robots.x1 import X1_CFG
from robolab import ROBOLAB_ROOT_DIR

# NOTE: KEY_BODY_NAMES must match lab_key_body_names in scripts/tools/retarget/config/x1.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_pitch_link",
    "right_knee_pitch_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3


@configclass
class X1AmpRewards():
    """Reward terms for AMP — mirrors RPO AMP reward structure."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)

    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0)
    arm_pitch_mean_offset = RewTerm(
        func=mdp.paired_joints_mean_deviation_l1,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_pitch_joint"],
            )
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )

    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    feet_distance_y = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "min": 0.14,
            "max": 0.50,
        },
    )

    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class X1AmpEnvCfg(AmpEnvCfg):
    rewards: X1AmpRewards = X1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = X1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "x1_lab"
        )
        # Motion weights will be configured after retarget produces x1_lab data.
        # Start with equal weights for all motions.
        self.motion_data.motion_dataset.motion_data_weights = None  # None = equal weights

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations — discriminator
        # ------------------------------------------------------
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.15

        # base
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.2

        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.arm_pitch_mean_offset.weight = -0.1

        # feet
        self.rewards.feet_slide.weight = -0.1
        self.rewards.sound_suppression.weight = -5e-5
        self.rewards.feet_distance_y.weight = 0.05

        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],
        )

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # ------------------------------------------------------
        # Terminations — X1 body names
        # ------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_hip_.*_link", "base_link", ".*_shoulder_.*_link", ".*_elbow_.*_link",
        ]
        if self.__class__.__name__ == "X1AmpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class X1AmpEnvCfg_PLAY(X1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
