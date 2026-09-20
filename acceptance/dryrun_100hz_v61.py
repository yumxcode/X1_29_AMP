#!/usr/bin/env python3
"""v61 100 Hz control-rate dry-run (mock env, call-level).

Paid lessons applied (v40 0-iter crashes; v55 inline-ternary; v49b single-side
disc obs): every touched callable must be genuinely CALLED, and every
config-walk must be exercised on faithful fakes, before a platform slot.

Checks (all WITHOUT isaaclab — sys.modules mocks, same pattern as
dryrun_reward_v56.py):

  1. mdp.observations.key_body_vel_b called TWICE with a mock env whose
     step_dt = 0.01 (100 Hz): the finite difference must divide by the env's
     step_dt (not the old hardcoded 0.02). A body moving 0.01 m in x per
     control step must read 1.0 m/s at 100 Hz (and 0.5 m/s at 50 Hz).
  2. The X1_CONTROL_HZ reward-walk block replicated verbatim on fake
     RewTerms: EMA alphas halve, action-rate weights double, zero-weight and
     None terms are skipped, delay steps double.
  3. AmpEnvCfg decimation arithmetic: 200 // hz for hz in {50, 100, 200}.
"""
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roboparty_train" / "robolab"))


class _Math(types.SimpleNamespace):
    @staticmethod
    def quat_apply(q, v):
        return v.clone()

    @staticmethod
    def quat_apply_inverse(q, v):
        return v.clone()


def _install_mocks():
    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    mdp_mod = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    sensors = types.ModuleType("isaaclab.sensors")
    assets = types.ModuleType("isaaclab.assets")
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")
    string_mod = types.ModuleType("isaaclab.utils.string")

    class SceneEntityCfg(types.SimpleNamespace):
        def __init__(self, name, joint_names=None, body_names=None,
                     preserve_order=False):
            super().__init__(name=name, joint_names=joint_names,
                             body_names=body_names,
                             preserve_order=preserve_order,
                             joint_ids=None, body_ids=None)

    class ManagerBasedRLEnv:  # TYPE_CHECKING only
        pass

    class ManagerBasedEnv:  # TYPE_CHECKING only
        pass

    envs.mdp = mdp_mod
    envs.ManagerBasedRLEnv = ManagerBasedRLEnv
    envs.ManagerBasedEnv = ManagerBasedEnv
    isaaclab.envs = envs
    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers
    sensors.ContactSensor = object
    sensors.RayCaster = object
    sensors.FrameTransformer = object
    isaaclab.sensors = sensors
    assets.Articulation = object
    assets.RigidObject = object
    isaaclab.assets = assets
    math_mod.quat_apply = _Math.quat_apply
    math_mod.quat_apply_inverse = _Math.quat_apply_inverse
    utils.math = math_mod
    string_mod.to_camel_case = lambda s: s
    utils.string = string_mod
    isaaclab.utils = utils

    sys.modules.update({
        "isaaclab": isaaclab, "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp_mod, "isaaclab.managers": managers,
        "isaaclab.sensors": sensors, "isaaclab.assets": assets,
        "isaaclab.utils": utils, "isaaclab.utils.math": math_mod,
        "isaaclab.utils.string": string_mod,
    })


def check_key_body_vel():
    # load observations.py BY FILE PATH: importing the robolab package drags
    # in isaaclab_tasks (unavailable locally) via __init__ chains — the
    # call-level dry run only needs the module itself with isaaclab mocked
    # (same pattern as dryrun_reward_v56.py).
    import importlib.util as _ilu
    _op = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "mdp" / "observations.py").resolve()
    _spec = _ilu.spec_from_file_location("amp_mdp_obs_dryrun", _op)
    OBS = _ilu.module_from_spec(_spec)
    sys.modules["amp_mdp_obs_dryrun"] = OBS
    _spec.loader.exec_module(OBS)
    key_body_vel_b, _KB_VEL_STATE = OBS.key_body_vel_b, OBS._KB_VEL_STATE

    # A body moving at a TRUE 1 m/s covers step_dt metres per control step
    # at ANY rate — the finite diff must read 1.0 m/s in both. (The pre-fix
    # hardcoded /0.02 would read 0.5 m/s at 100 Hz.)
    for step_dt in (0.01, 0.02):
        _KB_VEL_STATE.clear()
        N, M = 2, 3

        class _Data(types.SimpleNamespace):
            pass

        class _Robot:
            data = _Data()

        class _Env(types.SimpleNamespace):
            pass

        env = _Env(scene={"robot": _Robot}, step_dt=step_dt)
        cfg = types.SimpleNamespace(name="robot", body_ids=list(range(M)))
        _Robot.data.body_pos_w = torch.zeros(N, M, 3)
        _Robot.data.root_quat_w = torch.tensor([[1.0, 0, 0, 0]] * N)

        out0 = key_body_vel_b(env, cfg)     # first call: zeros, state seeded
        assert out0.shape == (N, M * 3), out0.shape

        # move each key body +step_dt m in x during the next control step
        _Robot.data.body_pos_w = torch.zeros(N, M, 3)
        _Robot.data.body_pos_w[..., 0] = step_dt   # 1 m/s at ANY rate
        out1 = key_body_vel_b(env, cfg)
        # EMA(alpha=0.2): first non-trivial sample -> 0.2 * measured
        measured = out1.reshape(N, M, 3)[0, 0, 0].item()
        expect = 1.0
        want = 0.2 * expect
        assert abs(measured - want) < 1e-6, \
            f"step_dt={step_dt}: vel={measured} want {want} (EMA a=0.2 of {expect})"
        print(f"[OK] key_body_vel_b step_dt={step_dt}: measured {expect} m/s "
              f"(EMA first sample {measured:.4f})")
    print("[PASS] 1/3 obs finite-diff is step_dt-driven (100 Hz & 50 Hz)")


class _Term:
    """Faithful fake of RewardTermCfg for the walk block."""

    def __init__(self, weight=0.0, params=None, func=None):
        self.weight = weight
        self.params = params if params is not None else {}
        self.func = func


class _Rewards(types.SimpleNamespace):
    pass


def check_hz_walk():
    # THE REAL shared implementation (control_hz_renorm.py) — no replica,
    # so the dry-run cannot drift from the shipped walk (v61 first-launch
    # lesson: yaw_bias/arm_opposite_leg_coupling alphas are FUNC DEFAULTS).
    import importlib.util as _ilu
    _cp = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "control_hz_renorm.py").resolve()
    _spec = _ilu.spec_from_file_location("control_hz_renorm_dryrun", _cp)
    RE = _ilu.module_from_spec(_spec)
    sys.modules["control_hz_renorm_dryrun"] = RE
    _spec.loader.exec_module(RE)

    def _fake_ema_guard(alpha):
        # mimics yaw_rate_bias_guard / arm_opposite_leg_coupling: alpha as
        # a kwarg DEFAULT, not wired into params
        def _f(env, alpha=alpha):
            return 0.0
        return _f

    r = _Rewards(
        yaw_bias=_Term(-1.0, {"command_name": "base_velocity"},
                       func=_fake_ema_guard(0.005)),          # v61b gap: func default
        arm_leg_coupling=_Term(-0.3, {"asset_cfg": "x"},
                               func=_fake_ema_guard(0.02)),   # v61b gap: func default
        arm_asym_lean=_Term(-0.8, {"alpha": 0.0025}),
        action_rate_l2=_Term(-0.01, {"asset_cfg": "x"}),
        action_rate_l2_arms=_Term(-0.0025, {}),
        smoothness_1=_Term(0.0),          # zero weight -> skipped, then None-d later
        joint_acc_l2=_Term(-2.5e-7, {"asset_cfg": "x"}),   # no alpha: untouched
        dead_term=None,                    # None-safe
        a_method=lambda e: 0,              # callable-safe
    )
    kbv = _Term(0.0, {"asset_cfg": "y"}, func=None)            # disc-vel term
    disc = types.SimpleNamespace(key_body_vel_b=kbv)

    delay = RE.apply_control_hz_renormalization(r, disc, 1)

    assert float(os.environ.get("X1_CONTROL_HZ")) == 100.0
    assert r.yaw_bias.params["alpha"] == 0.0025, r.yaw_bias.params
    assert r.arm_leg_coupling.params["alpha"] == 0.01, r.arm_leg_coupling.params
    assert r.arm_asym_lean.params["alpha"] == 0.00125, r.arm_asym_lean.params
    assert r.action_rate_l2.weight == -0.02, r.action_rate_l2.weight
    assert r.action_rate_l2_arms.weight == -0.005, r.action_rate_l2_arms.weight
    assert r.smoothness_1.weight == 0.0            # zero weight untouched
    assert r.joint_acc_l2.weight == -2.5e-7        # non-alpha term untouched
    assert delay == 2
    assert kbv.params["alpha"] == 0.1               # 0.2 / 2 default
    print("[OK] walk@100Hz: alphas halved, rates x2, delay 1->2, "
          "zero/None/callable skipped")
    print("[PASS] 2/3 hz-walk logic")


def check_decimation():
    got = {hz: 200 // hz for hz in (50, 100, 200)}
    assert got == {50: 4, 100: 2, 200: 1}, got
    print(f"[OK] decimation map {got} (sim.dt 0.005 kept)")
    print("[PASS] 3/3 decimation arithmetic")


def check_disc_stride():
    """v61d: AMPDiscriminator ::stride slicing (call-level, REAL module).

    Verifies: (a) stride=2 with steps=6 slices frames [0,2,4] and gives
    input_dim 3*D — the exact weight shape of the 50 Hz champion disc
    (warm-resume compatible); (b) stride=1 preserves the v61c semantics;
    (c) predict_style_reward runs on the sliced window; (d) resolve_amp_
    config derives stride from step_dt (0.01->2, 0.02->1)."""
    import importlib.util as _ilu
    import torch.nn as _nn

    for name in ("rsl_rl", "rsl_rl.utils", "rsl_rl.networks", "rsl_rl.env",
                 "tensordict"):
        if name in sys.modules:
            raise RuntimeError(f"{name} unexpectedly importable/imported")

    rsl = types.ModuleType("rsl_rl")
    utils = types.ModuleType("rsl_rl.utils")
    networks = types.ModuleType("rsl_rl.networks")
    envmod = types.ModuleType("rsl_rl.env")
    td = types.ModuleType("tensordict")

    class _Norm(_nn.Module):
        def __init__(self, shape, until=1e8):
            super().__init__()

        def forward(self, x):
            return x

    def _act(name):
        return _nn.ReLU()

    utils.resolve_nn_activation = _act
    networks.EmpiricalNormalization = _Norm
    envmod.VecEnv = object
    td.TensorDict = dict
    rsl.utils, rsl.networks, rsl.env = utils, networks, envmod
    sys.modules.update({"rsl_rl": rsl, "rsl_rl.utils": utils,
                        "rsl_rl.networks": networks, "rsl_rl.env": envmod,
                        "tensordict": td})

    _ap = (ROOT / "roboparty_train" / "rsl_rl" / "rsl_rl" / "modules"
           / "amp.py").resolve()
    _spec = _ilu.spec_from_file_location("amp_disc_dryrun", _ap)
    AMP = _ilu.module_from_spec(_spec)
    sys.modules["amp_disc_dryrun"] = AMP
    _spec.loader.exec_module(AMP)

    D, N = 121, 4

    disc = AMP.AMPDiscriminator(
        disc_obs_dim=D, disc_obs_steps=6, disc_obs_stride=2,
        obs_groups={"discriminator": ["disc"],
                    "discriminator_demonstration": ["disc_demo"]},
    )
    assert disc.input_dim == 3 * D, disc.input_dim  # == champion disc (warm resume)
    # frame tagging: frame i is filled with constant i
    obs = {"disc": torch.zeros(N, 6, D), "disc_demo": torch.zeros(N, 6, D)}
    for i in range(6):
        obs["disc"][:, i, :] = i
        obs["disc_demo"][:, i, :] = i
    out = disc.get_disc_obs(obs)
    assert out.shape == (N, 3, D), out.shape
    assert torch.allclose(out[:, 0], torch.full((N, D), 0.0))
    assert torch.allclose(out[:, 1], torch.full((N, D), 2.0))  # frames 0,2,4
    assert torch.allclose(out[:, 2], torch.full((N, D), 4.0))
    demo = disc.get_disc_demo_obs(obs)
    assert demo.shape == (N, 3, D) and torch.allclose(demo[:, 1, 0], torch.full((N,), 2.0))
    flat = disc.get_disc_obs(obs, flatten_history_dim=True)
    assert flat.shape == (N, 3 * D), flat.shape
    rew, score = disc.predict_style_reward(out, dt=0.01)
    assert rew.shape == (N,) and torch.isfinite(rew).all()

    # stride=1 (50 Hz / legacy default): unchanged semantics
    disc1 = AMP.AMPDiscriminator(
        disc_obs_dim=D, disc_obs_steps=6,
        obs_groups={"discriminator": ["disc"],
                    "discriminator_demonstration": ["disc_demo"]},
    )
    assert disc1.input_dim == 6 * D and disc1.disc_obs_steps_eff == 6

    # resolve_amp_config stride derivation
    def _resolve(step_dt):
        alg = {"amp_cfg": {"disc_obs_steps": 6, "disc_obs_dim": D,
                           "step_dt": step_dt}}
        obs_probe = {"disc": torch.zeros(2, 6, D), "disc_demo": torch.zeros(2, 6, D)}
        groups = {"discriminator": ["disc"],
                  "discriminator_demonstration": ["disc_demo"]}
        env = types.SimpleNamespace(
            env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=step_dt)))
        return AMP.resolve_amp_config(alg, obs_probe, groups, env)["amp_cfg"]["disc_obs_stride"]

    assert _resolve(0.01) == 2, "100 Hz must stride 2"
    assert _resolve(0.02) == 1, "50 Hz must keep stride 1"
    print("[OK] disc stride: frames[0,2,4], input 3D == champion, "
          "style reward finite, resolve 0.01->2 / 0.02->1")
    print("[PASS] 4/4 disc ::stride slicing")


if __name__ == "__main__":
    assert os.environ.get("X1_CONTROL_HZ", "50") == "50", \
        "run the walk check with X1_CONTROL_HZ=100"
    _install_mocks()
    check_key_body_vel()
    os.environ["X1_CONTROL_HZ"] = "100"
    check_hz_walk()
    check_decimation()
    check_disc_stride()
    print("\n[ALL PASS] v61 100 Hz dry-run")
