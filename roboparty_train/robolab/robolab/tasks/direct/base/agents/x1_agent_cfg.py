from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa: F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

from robolab.tasks.direct.base import (  # noqa: F401
    BaseAgentCfg,
)

# NOTE: Symmetry / mirror indices are DISABLED for X1.
# They must be computed after verifying the actual Isaac Lab joint order.
# To enable: run the diagnostic script (scripts/tools/print_x1_joint_order.py),
# then implement generate_x1_joint_mirror() based on the output.
#
# The RPO symmetry config hardcodes indices for 23 DOF with a specific joint order.
# X1 has 29 DOF and a different joint order, so the mirror mapping must be regenerated.


@configclass
class X1FlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "x1_flat"
        self.wandb_project: str = "x1_flat"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization = True
        self.critic_obs_normalization = True
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=None,  # TODO: enable after verifying joint order
            rnd_cfg=None,
        )
        self.clip_actions = 100.0


@configclass
class X1RoughAgentCfg(X1FlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "x1_rough"
        self.wandb_project: str = "x1_rough"
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=None,  # TODO: enable after verifying joint order
            rnd_cfg=None,
        )
