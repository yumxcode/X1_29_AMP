import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.f1w import F1W_CFG

from .x1_amp_env_cfg import X1AmpEnvCfg

# F1W-AMP: weilai F1 (SYS-SPEC-0002-V1.1, 35.033 kg, 241.9/218.4 Nm ankles)
# running task. Inherits the ENTIRE X1 AMP stack (rewards/events/terminations,
# 50 Hz control, clock conditioning, disc obs) — differences from X1:
#   * robot asset: F1W_CFG (weilai URDF, legs renamed to X1 _link convention
#     so every body_names regex in the inherited stack matches verbatim;
#     kp/kd/armature from F1_29DOF_weilai/f1_kp_kd.json)
#   * demo dataset: f1w_lab_v1 (run library retargeted through
#     AMASS->GMR->v30/v31/root-anchor chain + retarget_f1w_from_x1lab.py)
# The run-band gait-clock constants are env-var driven (X1_CLOCK_* plus the
# T/D/H LO/HI clamp overrides added to gait_clock.py) and set by the
# launcher run_f1w_amp_train.py.


@configclass
class F1WAmpEnvCfg(X1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # F1W robot (link names X1-compatible after the Isaac-side rename)
        self.scene.robot = F1W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # F1W run demo dataset (weights set by F1W_MOTION_WEIGHTS json or
        # the default dict below — speed-band bucketing, style centroid in
        # the 1.4-1.8 m/s target band per F1W_RUN_SPEED_TARGET.md)
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions",
            os.environ.get("F1W_MOTION_DIR", "f1w_lab_v3")
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # v2 milestone library (f1w_lab_v2): GMR-direct construction,
            # time-warped to X1-side anchored speeds; 2 sources (mirror-free:
            # weilai joint space is not a structural mirror, pair_sign probe
            # 207-271mm; mirrors deferred to a root-space construction),
            # all R4/FK/F'/C' gates PASS (F1W_RUN_REPORT.md sec 8)
            "07_01": 1.0,
            "103_07": 1.0,
        }

        if self.__class__.__name__ == "F1WAmpEnvCfg":
            self.disable_zero_weight_rewards()
            # v2 milestone command band: demo coverage 0.81-0.89 m/s
            # (F1W_RUN_REPORT sec 8); F1W_CMD_LO/HI to widen later
            _lo = float(os.environ.get("F1W_CMD_LO", "0.3"))
            _hi = float(os.environ.get("F1W_CMD_HI", "1.0"))
            self.commands.base_velocity.ranges.lin_vel_x = (_lo, _hi)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


# Speed-band weights for the f1w_lab_v1 library (literal dict: the
# acceptance checker parses this with a regex).
# Bands (F1W_RUN_DATA_ASSESSMENT.md + F1W_RUN_SPEED_TARGET.md):
#   RUN_SPRINT 09_x/35_x/16_35/acc8/accM_C3 (rebuilt 1.3-2.6 m/s)  w 0.5
#   RUN_MID    accM_C4/C4b/C5, accF_C5 (0.85-1.45)                 w 1.2  <- centroid band
#   RUN_FAST2  02_03 (1.28)                                        w 0.8
#   JOG        f08_* (1.1-1.33, from x1_fast_v72k)                 w 1.0
#   WALK       v72 walk anchors (0.4-1.3)                          w 0.5
# Weighted time-centroid lands ~1.4-1.6 m/s (target band center).
_W = {}
for _n in ["run_09_01", "run_09_04", "run_09_05", "run_09_06", "run_09_07",
           "run_09_09", "run_09_11",
           "run_35_17", "run_35_18", "run_35_19", "run_35_20",
           "run_16_35", "run_accM_C3"]:
    _W[_n] = 0.5
for _n in ["run_accM_C4", "run_accM_C4b", "run_accM_C5", "run_accF_C5"]:
    _W[_n] = 1.2
_W["run_02_03"] = 0.8
for _n in ["f08_01", "f08_03", "f08_06", "f08_08"]:
    _W[_n] = 1.0
for _n in ["07_01", "38_02", "103_07", "0005", "35_02", "82_12"]:
    _W[_n] = 0.5
DEFAULT_WEIGHTS = dict(_W)
for _n, _v in list(_W.items()):
    _W[_n + "_mirror"] = _v


@configclass
class F1WAmpEnvCfg_PLAY(F1WAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # fixed-command rollout: F1W_PLAY_VEL (default 1.6 m/s = contract
        # run target); 1.0 for walk-evidence videos
        import os as _os
        _v = float(_os.environ.get("F1W_PLAY_VEL", "1.6"))
        self.commands.base_velocity.ranges.lin_vel_x = (_v, _v)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if self.__class__.__name__ == "F1WAmpEnvCfg_PLAY":
            self.disable_zero_weight_rewards()
@configclass
class F1WAmpEnvCfg_PLAY(F1WAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # fixed-command rollout: F1W_PLAY_VEL (default 1.6 m/s = contract
        # run target); 1.0 for walk-evidence videos
        import os as _os
        _v = float(_os.environ.get("F1W_PLAY_VEL", "1.6"))
        self.commands.base_velocity.ranges.lin_vel_x = (_v, _v)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if self.__class__.__name__ == "F1WAmpEnvCfg_PLAY":
            self.disable_zero_weight_rewards()
