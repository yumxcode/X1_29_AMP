import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

# F1 weilai humanoid: 29 DOF, X1 upper body + weilai lower body
# (SYS-SPEC-0002-V1.1, mass-merged 35.033 kg; F1_AUDIT.md rev2 + fix commit).
#
# vs X1 (all from F1_AUDIT.md + f1_kp_kd.json, 2026-11-10):
#   - legs: hip_pitch +-90 deg / knee 0-90 / ankle +-30 deg; hip_roll
#     ASYMMETRIC (L -0.349..1.571); leg 573.0 mm (X1 574.1)
#   - ankle torque 241.9/218.4 Nm (X1 80) -> 2.6x margin at run GRF
#   - stand: base z 0.5876, hip_p -0.2915 / knee 0.450 / ankle_p -0.1585
#     (f1_stand.json, FK-verified symmetric, CoM over support center)
#   - kp/kd from f1_kp_kd.json (legs by I_stand + X1-validated band;
#     upper body = X1 identification plan B, links byte-identical)
#   - armature = f1_kp_kd.json estimates (UNVERIFIED, scaled by torque
#     class; sim2sim twin injects the SAME values -> domains consistent)

F1W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/f1w/urdf/f1w.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),  # stand 0.5876 + 4 cm settle margin
        joint_pos={
            # waist + arms identical to X1_CFG
            "lumbar_yaw_joint": 0.0,
            "lumbar_roll_joint": 0.0,
            "lumbar_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "left_elbow_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_elbow_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # weilai stand (f1_stand.json urdf order, FK-verified)
            "left_hip_pitch_joint": -0.2915291,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": 0.4500,
            "left_ankle_pitch_joint": -0.1584709,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2915291,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": 0.4500,
            "right_ankle_pitch_joint": -0.1584709,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # waist: lumbar yaw/roll 150, pitch 180 Nm (X1 plan-B gains)
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                "lumbar_yaw_joint",
                "lumbar_roll_joint",
                "lumbar_pitch_joint",
            ],
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            stiffness={
                "lumbar_yaw_joint": 90.0,
                "lumbar_roll_joint": 70.0,
                "lumbar_pitch_joint": 100.0,
            },
            damping=8.0,
            armature={"lumbar_yaw_joint": 0.022,
                      "lumbar_roll_joint": 0.021,
                      "lumbar_pitch_joint": 0.012},
            min_delay=0,
            max_delay=2,
        ),
        # hips + knees: 130 Nm, 14.66 rad/s (weilai R-class)
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim=130.0,
            velocity_limit_sim=14.66,
            stiffness={
                ".*_hip_pitch_joint": 70.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_yaw_joint": 90.0,
                ".*_knee_pitch_joint": 90.0,
            },
            damping={
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_yaw_joint": 2.0,
                ".*_knee_pitch_joint": 6.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.15,
                ".*_hip_roll_joint": 0.15,
                ".*_hip_yaw_joint": 0.02,
                ".*_knee_pitch_joint": 0.20,
            },
            min_delay=0,
            max_delay=2,
        ),
        # ankles: 241.9 (pitch) / 218.4 (roll) Nm — weilai's 3x-X1 authority
        "ankles": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 241.875,
                ".*_ankle_roll_joint": 218.4,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 7.57,
                ".*_ankle_roll_joint": 8.05,
            },
            stiffness={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 45.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 0.6,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # shoulders + elbows: 20 Nm
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=13.61,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 4.0,
                ".*_shoulder_roll_joint": 4.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_pitch_joint": 1.9,
                ".*_elbow_yaw_joint": 1.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # wrists: 10 Nm, 2.5 rad/s
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.5,
            stiffness=15.0,
            damping=1.0,
            armature=0.005,
            min_delay=0,
            max_delay=2,
        ),
    },
)

F1W_LINKS = [
    "base_link",
    "lumbar_yaw_link",
    "lumbar_roll_link",
    "lumbar_pitch_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
