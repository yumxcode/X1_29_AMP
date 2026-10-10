import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.f1 import F1_CFG

from .x1_amp_env_cfg import X1AmpEnvCfg

# F1-AMP: X1 upper body + F1 lower body (orthogonal hip, perfect-mirror
# right leg). Inherits the ENTIRE X1 AMP reward/event/termination stack —
# F1 differs from X1 only in leg kinematics:
#   * robot asset: F1_CFG (own URDF + FK-verified standing pose)
#   * demo dataset: f1_lab_v67 (x1_lab_v67 leg-IK re-solved onto F1
#     kinematics by retarget_f1_from_x1lab.py; arms/lumbar/root verbatim,
#     ankle world poses tracked, 10-body key_body_pos recomputed with F1 FK)
# Everything else (rewards, commands, events, terminations, 50 Hz control)
# is byte-identical to the X1 v67 recipe.


@configclass
class F1AmpEnvCfg(X1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # F1 robot
        self.scene.robot = F1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # F1 demo dataset (v67 overground-dominant weights, X1 recipe)
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions",
            os.environ.get("F1_MOTION_DIR", "f1_lab_v67")
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 4.0, "36_11": 3.0, "0026_circle_walk": 3.0,
            "0000_treadmill_norm": 0.5, "0002_treadmill_slow": 0.5,
            "0005_normal_walk1": 0.5, "0007_normal_walk3": 0.5,
            "0008_normal_walk4": 0.5,
            "0003_treadmill_jog": 1.0, "0009_normal_jog1": 1.0,
            "36_01_mirror": 4.0, "36_11_mirror": 3.0,
            "0026_circle_walk_mirror": 3.0,
            "0000_treadmill_norm_mirror": 0.5,
            "0002_treadmill_slow_mirror": 0.5,
            "0005_normal_walk1_mirror": 0.5,
            "0007_normal_walk3_mirror": 0.5,
            "0008_normal_walk4_mirror": 0.5,
            "0003_treadmill_jog_mirror": 1.0,
            "0009_normal_jog1_mirror": 1.0,
        }

        if self.__class__.__name__ == "F1AmpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class F1AmpEnvCfg_PLAY(F1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
