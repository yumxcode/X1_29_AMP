import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

# X1 humanoid: 29 DOF
# - Waist: lumbar_yaw/roll/pitch (3)
# - Arms: shoulder_pitch/roll/yaw + elbow_pitch/yaw + wrist_pitch/roll (7 x 2 = 14)
# - Legs: hip_pitch/roll/yaw + knee_pitch + ankle_pitch/roll (6 x 2 = 12)
# Total: 3 + 14 + 12 = 29

X1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/x1/urdf/f1.urdf",
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
        pos=(0.0, 0.0, 0.75),  # base_link height in bent-knee standing pose (total robot height ~1.25m)
        joint_pos={
            # waist
            "lumbar_yaw_joint": 0.0,
            "lumbar_roll_joint": 0.0,
            "lumbar_pitch_joint": 0.0,
            # left arm
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "left_elbow_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            # right arm
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_elbow_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # left leg
            "left_hip_pitch_joint": 0.48891,
            "left_hip_roll_joint": 0.06213,
            "left_hip_yaw_joint": -0.33853,
            "left_knee_pitch_joint": 0.63204,
            "left_ankle_pitch_joint": -0.27224,
            "left_ankle_roll_joint": 0.0,
            # right leg
            "right_hip_pitch_joint": -0.48891,
            "right_hip_roll_joint": -0.06213,
            "right_hip_yaw_joint": 0.33853,
            "right_knee_pitch_joint": 0.63204,
            "right_ankle_pitch_joint": -0.27224,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # Waist + Hips + Knees: R86-3 class motors (effort 150-180 Nm)
        "waist_legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "lumbar_yaw_joint",
                "lumbar_roll_joint",
                "lumbar_pitch_joint",
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            stiffness={
                "lumbar_yaw_joint": 120.0,
                "lumbar_roll_joint": 120.0,
                "lumbar_pitch_joint": 150.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_knee_pitch_joint": 150.0,
            },
            damping={
                "lumbar_yaw_joint": 4.0,
                "lumbar_roll_joint": 4.0,
                "lumbar_pitch_joint": 5.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_hip_roll_joint": 3.3,
                ".*_hip_yaw_joint": 3.3,
                ".*_knee_pitch_joint": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # Ankles: R86-2 class motors (effort 80 Nm)
        "ankles": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=13.61,
            stiffness=50.0,
            damping=2.5,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # Shoulders + Elbows: R52 class motors (effort 20 Nm)
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=13.61,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # Wrists: R52 class, weakest (effort 10 Nm)
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.5,
            stiffness=15.0,
            damping=1.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


X1_LINKS = [
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
