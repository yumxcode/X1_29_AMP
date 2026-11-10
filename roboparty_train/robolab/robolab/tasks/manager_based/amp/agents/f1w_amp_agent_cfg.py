import os

from isaaclab.utils import configclass

from ..agents.x1_amp_agent_cfg import X1RslRlOnPolicyRunnerAmpCfg


# F1W-AMP runner: inherits the X1 AMP PPO/AMP hyperparameters verbatim.
# Only the experiment identity changes (separate logs / checkpoints).
# X1_CONTROL_HZ default '50' — native 50 Hz controller, X1 champion regime.


@configclass
class F1WRslRlOnPolicyRunnerAmpCfg(X1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "f1w_amp"
    wandb_project = "f1w_amp"
