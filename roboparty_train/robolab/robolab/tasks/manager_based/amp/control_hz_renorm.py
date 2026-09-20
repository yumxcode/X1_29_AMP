"""X1_CONTROL_HZ time-constant re-normalization (v61 100 Hz arc).

Single shared implementation for X1AmpEnvCfg.__post_init__ AND the local
call-level dry-run (acceptance/dryrun_100hz_v61.py) — kept isaaclab-free
so the dry-run imports THE REAL function instead of a drifting replica
(the v61 first-launch miss: yaw_bias/arm_opposite_leg_coupling take alpha
as a FUNCTION DEFAULT, not a params entry, and the params-only walk left
their EMAs 2x too fast at 100 Hz).

Semantics: the 50 Hz-tuned recipe constants are defined PER CONTROL STEP.
At rate r = hz/50 they must be re-normalized to keep their TIME-domain
meaning:
  - EMA alphas (tau ~= dt/alpha)          -> alpha / r   (explicit params
    entry first; otherwise the reward func's `alpha` kwarg default is
    materialized into params, scaled)
  - action_delay_steps (steps -> 20 ms)   -> steps * r
  - action-rate/smoothness finite-diff
    penalties (per-step delta^2, steps/s) -> weight * r
  - disc key_body_vel_b EMA alpha         -> alpha / r
(style_reward_scale * r is handled in x1_amp_agent_cfg.py because
predict_style_reward multiplies by dt.)
"""
import inspect
import os

RATE_KEYS = ("action_rate_l2", "action_rate_l2_arms", "smoothness_1")


def apply_control_hz_renormalization(rewards, observations_disc, action_delay_steps,
                                     log=print):
    """Returns the re-scaled action_delay_steps (unchanged at 50 Hz).

    rewards: the env cfg .rewards namespace (terms are duck-typed:
        .weight, .params, optional .func)
    observations_disc: the disc ObservationGroup cfg (or None); its
        key_body_vel_b term's EMA alpha is rescaled like reward alphas.
    """
    hz = float(os.environ.get("X1_CONTROL_HZ", "50"))
    if abs(hz - 50.0) <= 1e-6:
        return action_delay_steps
    r = hz / 50.0
    scaled_alphas = []
    scaled_rates = []

    def _rescale_alpha(term, attr):
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            return False
        if "alpha" in params:
            params["alpha"] = float(params["alpha"]) / r
            return True
        if not params:
            return False  # nothing wired; a default-only call is unlikely
        func = getattr(term, "func", None)
        if func is None:
            return False
        try:
            sig = inspect.signature(func)
            p = sig.parameters.get("alpha")
        except (TypeError, ValueError):
            return False
        if (p is not None and p.default is not inspect.Parameter.empty
                and isinstance(p.default, (int, float))):
            params["alpha"] = float(p.default) / r
            return True
        return False

    for attr in dir(rewards):
        if attr.startswith("__"):
            continue
        term = getattr(rewards, attr)
        if callable(term) or term is None:
            continue
        if _rescale_alpha(term, attr):
            scaled_alphas.append(attr)
        if attr in RATE_KEYS and term.weight:
            term.weight = term.weight * r
            scaled_rates.append(attr)

    if action_delay_steps:
        action_delay_steps = int(round(action_delay_steps * r))

    kbv = getattr(observations_disc, "key_body_vel_b", None) \
        if observations_disc is not None else None
    if kbv is not None and isinstance(getattr(kbv, "params", None), dict):
        kbv.params["alpha"] = float(kbv.params.get("alpha", 0.2)) / r

    log(f"[CONTROL-HZ] {int(hz)} Hz: EMA alpha halved on "
        f"{sorted(scaled_alphas)} action-rate x{r:g} on {sorted(scaled_rates)} "
        f"delay_steps={action_delay_steps}")
    return action_delay_steps
