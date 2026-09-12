import os
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg
from robolab.assets.robots.x1 import X1_CFG
from robolab import ROBOLAB_ROOT_DIR

# NOTE: KEY_BODY_NAMES must match lab_key_body_names in scripts/tools/retarget/config/x1.yaml
# v30 FIX: order was [ankles, knees] while the yaml/pkl schema is [knees, ankles].
# The discriminator compared policy key_body_pos_b (env order) against
# ref_key_body_pos_b (pkl schema order) with columns 0-3 PERMUTED between the
# two domains since v16 -> ankle channels were matched against reference KNEES.
KEY_BODY_NAMES = [
    "left_knee_pitch_link",
    "right_knee_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
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
    # v34: in-phase swing detector on HIGH-PASSED residuals (|hpL + hpR|).
    # History: v28b |devL-devR| penalized natural antiphase (-> sync arms);
    # v32's |devL+devR| sum turned out to tax the reference's joint-space
    # shared DC (-26 deg -> -0.137/step on ref vs -0.005 on frozen arms,
    # composite audit 2026-09-11) and helped collapse v33b arm swing to
    # 4-8 deg. Residual version: zero on antiphase at any amplitude, large
    # on in-phase swing, no DC sensitivity (see mdp.arm_sync_residual).
    arm_pitch_sync = RewTerm(
        func=mdp.arm_sync_residual,
        weight=0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
                preserve_order=True,
            )
        },
    )
    # v33/v34: frozen ANTISYMMETRIC arm offset guard (v32b defect: L -9 /
    # R +9.8 deg held). v34 slows the EMA to alpha=0.0025 (tau 8 s): at
    # alpha=0.02 the 1 Hz antiphase AC leaked ~28 deg into the EMA and the
    # guard taxed swing amplitude (ref scored -0.0998/step).
    arm_asym_lean = RewTerm(
        func=mdp.paired_joints_deviation_difference_ema_l1,
        weight=0,
        params={
            "alpha": 0.0025,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
                preserve_order=True,
            )
        },
    )
    # v33/v34: arm/opposite-leg coordination (refs corr +0.8..+0.99; v32b
    # -0.12). v34 fixes: (a) high-passed product — the raw-deviation product
    # cancelled via DC cross-terms (scored ~0 on ref); (b) the R-shoulder x
    # L-hip product is NEGATED — that pair carries the opposite joint-space
    # sign (measured corr -0.98 on refs; shoulder pair same-sign, hip pair
    # anti-aligned). v33 shipped without the negation so the two products
    # (+0.105/-0.105 on ref) mutually cancelled. Ref now +0.031/step;
    # collapsed arms ~+0.001 (small anti-collapse gradient toward natural
    # coordinated swing).
    arm_leg_coupling = RewTerm(
        func=mdp.arm_opposite_leg_coupling,
        weight=0,
        params={
            "shoulder_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
                preserve_order=True,
            ),
            "hip_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_hip_pitch_joint", "left_hip_pitch_joint"],
                preserve_order=True,
            ),
        },
    )
    # v33: chest posture (FK: +lumbar_pitch = chest forward; refs mean
    # +14.9 deg; v32b sat at -9 deg backward).
    lumbar_posture = RewTerm(
        func=mdp.lumbar_pitch_prior,
        weight=0,
        params={
            "target": 0.26,
            # v33b BUG FIX: without an explicit asset_cfg the term indexed
            # joint_ids[0] of ALL 29 joints (a mystery leg joint got pulled
            # toward 0.26 rad; the v33 run degenerated into side-walking
            # with the left hip held at +42 deg as compensation — root y
            # drift 6.27 m at iter 2500, hip ratio 0.27, lumbar_pitch
            # itself never moved from -10 deg. ALWAYS pin joint_names in
            # single-joint reward terms.)
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lumbar_pitch_joint"]),
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )

    # -- v27: strict gait quality (sim2sim criteria) ---------------------
    # sole must stay flat during stance at walking speeds (no toe-walk /
    # ball-foot); see stance_sole_flat_walk docstring for the frame trap that
    # rules out the built-in feet_orientation_l2 on X1.
    stance_sole_flat = RewTerm(
        func=mdp.stance_sole_flat_walk,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "max_cmd_speed": 1.5,
        },
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

    # v35: constant-yaw-bias guard. v34 sim2sim: body-frame tracking fine
    # (vx 0.89/vy ~0) but a persistent +4.6 deg/s yaw bias on straight
    # commands -> 55 deg heading drift / +3.5 m lateral over 12 s (v32b was
    # 0.4 deg/s). EMA(alpha 0.005) of (ang_vel_z - cmd_z) isolates the DC;
    # exp-kernel std 0.5 rad/s is blind at 0.08 rad/s (kernel 0.975).
    yaw_bias = RewTerm(
        func=mdp.yaw_rate_bias_guard,
        weight=0,
        params={"command_name": "base_velocity"},
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
        # v31: x1_lab_v31 = fix_arm_decomposition.py (v30 arm fix) +
        # fix_ground_root.py (sole penetration -> 0, L/R ankle stance-pitch
        # symmetrization, root_z ground anchoring) + 2 new CMU overground
        # walking clips (103_07, retargeted locally with the same
        # auto-IK config; real stride-speed dynamics the in-place BMLrub
        # demos lack). Dropped: 114_08/114_09/127_04/127_06.
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "x1_lab_v31"
        )
        # Motion weights: must explicitly list motion names (empty dict = load nothing)
        # v28: every clip now has a FK-verified left-right mirrored twin
        # (roboparty_train/mirror_lab_motions.py, FK mirror error < 0.8 mm on
        # all 14 clips). The AMASS refs themselves are heavily L/R asymmetric
        # (hip swing ratio 0.35-0.97) and v27 inherited it (0.5 m/s policy
        # ratio 0.832 < 0.85). A mirrored-pair dataset makes the AMP style
        # prior exactly symmetric by construction.
        # v30: DROP 114_08 / 114_09 / 127_04 / 127_06 (+ mirrors): non-walking
        # arm styles (raised/folded arms, waist swing 86-98 deg that even the
        # re-IK fallback tiers cannot compress - shoulder_roll saturates at
        # its abduction limit on 34-43% of frames). They were the main
        # asymmetric-forearm-pose style polluters in v29.
        # v31 additions at weight 1.0: 103_07 (1.2 m/s) —
        # real overground walking (genuinely counter-rotating torso, GMR
        # lumY swing 62 deg; kept at c=1.0 = exact elbows, subordinate
        # weight so the restrained BMLrub style stays dominant).
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1.0,
            "36_11": 1.0,
            "0000_treadmill_norm": 2.0,
            "0002_treadmill_slow": 2.0,
            "0003_treadmill_jog": 2.0,
            "0005_normal_walk1": 2.0,
            "0007_normal_walk3": 2.0,
            "0008_normal_walk4": 2.0,
            "0009_normal_jog1": 2.0,
            "0026_circle_walk": 2.0,
            "103_07": 1.0,
            "36_01_mirror": 1.0,
            "36_11_mirror": 1.0,
            "0000_treadmill_norm_mirror": 2.0,
            "0002_treadmill_slow_mirror": 2.0,
            "0003_treadmill_jog_mirror": 2.0,
            "0005_normal_walk1_mirror": 2.0,
            "0007_normal_walk3_mirror": 2.0,
            "0008_normal_walk4_mirror": 2.0,
            "0009_normal_jog1_mirror": 2.0,
            "0026_circle_walk_mirror": 2.0,
            "103_07_mirror": 1.0,
        }

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
        # v32: difference stat (v28b, wrong premise -> caused sync arms)
        # v34: now the high-passed residual statistic. Zero on natural
        # antiphase at any amplitude AND on the reference's shared DC
        # (measured ref score ~-0.015/step vs -0.137 for the deprecated
        # sum); in-phase swing (the v31d defect) still caught.
        self.rewards.arm_pitch_sync.weight = -0.3
        # v33/v34 (v32b defects, diag_arm_offset.py):
        # - frozen antisymmetric arm offset: L -9 / R +9.8 deg held -> slow
        #   EMA (tau 8 s) DC guard; ref ~5 deg (-0.019) vs frozen 18.8 deg
        #   (-0.066) -> separates natural small asym from frozen defects
        self.rewards.arm_asym_lean.weight = -0.2
        # - arm/opposite-leg coupling: refs +0.8..+0.99 vs policy -0.12;
        #   high-passed capped product (positive on ref, mean-shift free)
        self.rewards.arm_leg_coupling.weight = 0.3
        # - chest posture: refs mean +14.9 deg forward vs policy -9 deg back;
        #   0.42 rad error -> -0.17/step at weight -0.4
        self.rewards.lumbar_posture.weight = -0.4

        # feet
        self.rewards.feet_slide.weight = -0.1
        self.rewards.sound_suppression.weight = -5e-5
        self.rewards.feet_distance_y.weight = 0.05

        # v27 gait quality: v26 sim2sim evidence (gait_metrics on model_3999,
        # acceptance/v27_eval/v26_gait_report.json):
        #   - 1.0 m/s: heel raised 19% of mid-stance (ball-foot), sole pitch
        #     to -42 deg in stance -> stance_sole_flat kills it. Reference
        #     clips keep the sole flat (heelup=0.000), so this is aligned
        #     with the AMP style, not fighting it.
        # Magnitude at v26 gait: mean sin^2 tilt = 0.157 (1.0 m/s) -> weight
        # -1.5 gives ~-0.24/step: ~9% of task reward (2.65/step at perfect
        # tracking), dominant among always-on penalties but reducible to ~0
        # by flat soles (reference proves feasible).
        # NOTE: NO instantaneous hip-pair symmetry term: measured on v26 data,
        # corr(dev_L,dev_R)=+0.78 — gait mirror symmetry is invariance under
        # mirror x T/2 shift, so neither |mean| nor |diff| of instantaneous
        # deviations isolates amplitude asymmetry (both punish normal swing).
        # If v27 hip amplitude ratio still < 0.85, v28 needs a per-cycle
        # amplitude-energy term (stateful). 127_06 down-weight may improve it
        # incidentally (its toe-down style plausibly drove the asymmetry).
        self.rewards.stance_sole_flat.weight = -1.5

        # v35: yaw DC bias guard. Calibrated (offline, alpha 0.005):
        # v33b/v34 carry a 4.7-5.2 deg/s DC bias on zero-yaw commands
        # (-> 55 deg heading drift / 3.5 m lateral in 12 s sim2sim), v32b
        # only 0.4. At weight -0.5: v34-level bias (0.09 rad) -> -0.046/step
        # (comparable to feet_slide, enough to matter); good tracking (EMA
        # ~0.05 rad transient) -> -0.003; refs' slow-turn segments (e.g.
        # 103_07 local 30 deg/s) would score -0.26 — deliberate: turning
        # belongs to yaw COMMANDS, not to zero-yaw drift.
        self.rewards.yaw_bias.weight = -0.5
        # v35 note: task_style_lerp 0.6->0.7 lives in the AGENT cfg
        # (x1_amp_agent_cfg.py): P3a lin kernel sat at 0.78 since v31d
        # (v27/v28: 0.84, old dataset); style reward now 1.14-2.00 (deep in
        # saturation, marginal style gradient small). v21 history: lerp 0.75
        # halved style 0.335->0.146 (P5a FAIL); 0.7 keeps ~10x P5a margin
        # while shifting task/style 60/40 -> 70/30.
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
        # Events — override all RPO torso_link references to X1 body names
        # (AmpEnvCfg base hardcodes RPO body names in multiple EventTerms)
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = ["lumbar_pitch_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["lumbar_pitch_link", "base_link"]
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque.params["asset_cfg"].body_names = ["lumbar_pitch_link"]

        # ------------------------------------------------------
        # v29 sim2real hardening (toggle: X1_ROBUST_TRAIN=0 disables — used
        # by the v29e clean-phase fine-tune to re-converge style/kernels
        # onto a policy that already internalized recovery).
        # v29b postmortem (TASK_20260909_165, 5/13):
        # push +-1.2 m/s @2-5s made transient tilts trip the bad_orientation
        # termination constantly -> 94% of episodes ended by orientation
        # before recovery was learnable -> converged to "lean, never fall,
        # never walk" (ep_len 278 vs v28d 996, mean reward -0.44 vs +22.98).
        # v29c: moderate step from v28d's proven config:
        #   push_robot +-0.8 m/s every 4-8 s (was +-0.5 @5-10 s)
        #   action delay 0-1 step @ p=[.8,.2] (20 ms jitter; sweep showed
        #   v28d already tolerates 1-step fully, 5/5)
        # ------------------------------------------------------
        import os as _os
        if _os.environ.get("X1_ROBUST_TRAIN", "1") != "0":
            self.events.push_robot.interval_range_s = (4.0, 8.0)
            self.events.push_robot.params["velocity_range"] = {
                "x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-1.0, 1.0)}
            self.action_delay_steps = 1

        # ------------------------------------------------------
        # Terminations — X1 body names
        # X1 body structure: base_link → lumbar(yaw/roll/pitch) → arms
        #                     base_link → hip(pitch/roll/yaw) → knee → ankle
        # Terminate if any body except feet (ankle_*) touches ground
        # ------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_hip_.*_link",          # 髋关节着地
            ".*_knee_.*_link",         # 膝盖着地
            "base_link",               # 骨盆着地
            "lumbar_.*_link",          # 腰部着地
            ".*_shoulder_.*_link",     # 肩膀着地
            ".*_elbow_.*_link",        # 手肘着地
            ".*_wrist_.*_link",        # 手腕着地
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
        # eval/play must be clean (no delay injection)
        self.action_delay_steps = 0

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
