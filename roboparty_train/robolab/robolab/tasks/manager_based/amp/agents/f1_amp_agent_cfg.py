import os

from isaaclab.utils import configclass

from ..agents.x1_amp_agent_cfg import X1RslRlOnPolicyRunnerAmpCfg


# F1-AMP runner: inherits the X1 AMP PPO/AMP hyperparameters verbatim.
# Only the experiment identity changes (separate logs / checkpoints).
# X1_CONTROL_HZ default '50' — the F1 goal is a native 50 Hz controller,
# matching the X1 50 Hz champion regime (v16-v59 lineage).


@configclass
class F1RslRlOnPolicyRunnerAmpCfg(X1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "f1_amp"
    wandb_project = "f1_amp"
