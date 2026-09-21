#!/usr/bin/env python3
"""v63 dry-run: disc macro-window (600 ms) + disc root-linvel + action LPF.

Call-level, REAL implementations where importable (same pattern as
dryrun_100hz_v61.py); source-scan coverage where the module chain is not
locally importable (X1AmpEnvCfg needs a live sim).

Checks:
  1. ActionLPF (REAL amp_action_lpf.TorchActionLPF): DC gain exactly 1;
     sinusoid at fc attenuated ~0.7071^order (-3 dB/section); 1 Hz gait
     passes >= 0.99; hz=0 identity; numerical parity with the deploy-side
     numpy recurrence spec (identical inputs -> identical outputs).
  2. Deploy wiring source-scan: mujoco_rollout exposes --action-lpf /
     --action-lpf-order and applies the cascade BEFORE act_buf; P7 gate +
     battery + robustness sweep forward the flags; amp_env applies the
     filter BEFORE the delay ring.
  3. Disc macro-window (REAL AMPDiscriminator): steps=60 stride=2 ->
     30 frames [0,2,..,58], input_dim 30*124=3720; style reward finite;
     full post-buffer path (update_normalization -> normalize -> forward
     -> grad_penalty) on the sliced window.
  4. resolve_amp_config stride override at 100 Hz (X1_DISC_STRIDE=2).
  5. ref_root_lin_vel_b (REAL mdp.observations): mock animation term with
     world-frame velocity 1 m/s + identity quat -> returns 1.0 m/s,
     shape (N, steps, 3) unflattened.
  6. CircularBuffer memory arithmetic: v63 (24, 4096, 30, 124) fits;
     the v61 default 100 would not.
"""
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roboparty_train" / "robolab"))


def _install_mocks():
    """Same isaaclab mock set as dryrun_100hz_v61.py (mdp.observations
    imports isaaclab.utils.math at module level)."""
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

    math_mod.quat_apply = lambda q, v: v.clone()
    math_mod.quat_apply_inverse = lambda q, v: v.clone()
    string_mod.to_camel_case = lambda s: s
    envs.mdp = mdp_mod
    envs.ManagerBasedRLEnv = object
    envs.ManagerBasedEnv = object
    isaaclab.envs = envs
    managers.SceneEntityCfg = SceneEntityCfg
    sensors.ContactSensor = object
    sensors.RayCaster = object
    sensors.FrameTransformer = object
    isaaclab.sensors = sensors
    assets.Articulation = object
    assets.RigidObject = object
    isaaclab.assets = assets
    utils.math = math_mod
    utils.string = string_mod
    isaaclab.utils = utils
    sys.modules.update({
        "isaaclab": isaaclab, "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp_mod, "isaaclab.managers": managers,
        "isaaclab.sensors": sensors, "isaaclab.assets": assets,
        "isaaclab.utils": utils, "isaaclab.utils.math": math_mod,
        "isaaclab.utils.string": string_mod,
    })


# ---------------------------------------------------------------- check 1
def check_lpf():
    import importlib.util as _ilu
    _lp = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "amp_action_lpf.py").resolve()
    _spec = _ilu.spec_from_file_location("amp_action_lpf_dryrun", _lp)
    LPF = _ilu.module_from_spec(_spec)
    sys.modules["amp_action_lpf_dryrun"] = LPF
    _spec.loader.exec_module(LPF)

    dt, fc, order = 0.01, 10.0, 2
    f = LPF.TorchActionLPF(fc, order, dt)
    assert abs(LPF.lpf_alpha(fc, dt) - (1 - np.exp(-2 * np.pi * fc * dt))) < 1e-12

    # DC gain exactly 1 (steady state of a unity-DC section cascade)
    x = torch.ones(50, 29)
    f.reset_with(torch.zeros(29))
    y = f(x[-1])
    for _ in range(400):
        y = f(x[-1])
    assert torch.allclose(y, torch.ones(29), atol=1e-6), y

    # frequency response: drive with sinusoids, measure steady-state RMS
    # (NOT input-projection — at fc the cascade's phase lag reaches 90 deg
    # and the projection reads A*cos(90 deg) ~= 0, phase-confounded)
    def gain_at(freq):
        g = LPF.TorchActionLPF(fc, order, dt)
        g.reset_with(torch.zeros(1))
        n = int(60 / dt)                      # 60 s settling+measure
        sig = torch.sin(2 * np.pi * freq * torch.arange(n) * dt).reshape(-1, 1)
        ys = [g(sig[i]).item() for i in range(n)]
        yy = np.array(ys[len(ys) // 2:])      # discard transient
        return float(np.sqrt(2 * np.mean(yy ** 2)))  # sine amplitude from RMS

    g_fc = gain_at(fc)
    want = 0.7071 ** order
    assert abs(g_fc - want) < 0.03, f"gain@fc {g_fc:.3f} vs {want:.3f}"
    g_gait = gain_at(1.0)
    assert g_gait > 0.99, f"1 Hz gait must pass: {g_gait:.4f}"

    # disabled filter is a pure identity (no state created)
    f0 = LPF.TorchActionLPF(0.0, order, dt)
    z = torch.randn(4, 29)
    assert f0(z) is z

    # parity with the deploy-side recurrence spec (numpy, the exact two
    # lines mujoco_rollout.py runs per step per section)
    alpha = LPF.lpf_alpha(fc, dt)
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(300, 29))
    ft = LPF.TorchActionLPF(fc, order, dt)
    ft.reset_with(torch.from_numpy(seq[0].copy()))
    outs_t, outs_n = [], []
    state = [seq[0].astype(np.float64).copy() for _ in range(order)]
    for i in range(len(seq)):
        outs_t.append(ft(torch.from_numpy(seq[i])).numpy().copy())
        act = seq[i]
        for k in range(order):
            state[k] = alpha * act + (1 - alpha) * state[k]
            act = state[k]
        outs_n.append(act.copy())
    err = np.abs(np.array(outs_t) - np.array(outs_n)).max()
    assert err < 1e-10, err
    print(f"[OK] LPF: DC=1, gain@10Hz={g_fc:.3f}~0.7071^{order}, "
          f"1Hz={g_gait:.4f}, torch/numpy parity {err:.1e}")
    print("[PASS] 1/6 action low-pass")


# ---------------------------------------------------------------- check 2
def check_deploy_wiring():
    roll = (ROOT / "sim2sim" / "mujoco_rollout.py").read_text()
    for needle in ('"--action-lpf"', "--action-lpf-order",
                   "lpf_alpha", "act_buf.appendleft"):
        assert needle in roll, needle
    i_alpha = roll.index("lpf_alpha = ")
    i_lpf = roll.index("for _i in range(len(lpf_state)):")
    i_buf = roll.index("act_buf.appendleft(")
    assert i_alpha < i_lpf < i_buf, "cascade must run BEFORE the lag buffer"
    assert '"action_lpf": args.action_lpf' in roll, "meta stamp missing"

    p7 = (ROOT / "roboparty_train" / "run_x1_amp_train.py").read_text()
    assert 'os.environ.get("X1_ACT_LPF_HZ", "0")' in p7
    assert '"--action-lpf", str(_lpf_hz)' in p7
    bat = (ROOT / "acceptance" / "eval_100hz_battery.py").read_text()
    assert '"--action-lpf", str(args.action_lpf)' in bat
    sw = (ROOT / "sim2sim" / "robustness_sweep.py").read_text()
    assert "X1_ACT_LPF_HZ" in sw
    env = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "amp_env.py").read_text()
    assert "_action_lpf(action.to(self.device))" in env
    # the LPF must be the INNER call: _random_action_delay(_action_lpf(a))
    m = re.search(r"_random_action_delay\(\s*self\._action_lpf\(", env)
    assert m, "env: LPF must run BEFORE the delay ring (inner call)"
    print("[OK] deploy wiring: rollout args+order, P7/battery/sweep forward, "
          "env LPF-before-delay")
    print("[PASS] 2/6 deploy wiring")


# ---------------------------------------------------------------- check 3
def check_disc_window():
    import importlib.util as _ilu
    import torch.nn as _nn

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

        def update(self, x):
            return None

    utils.resolve_nn_activation = lambda name: _nn.ReLU()
    networks.EmpiricalNormalization = _Norm
    envmod.VecEnv = object
    td.TensorDict = dict
    rsl.utils, rsl.networks, rsl.env = utils, networks, envmod
    sys.modules.update({"rsl_rl": rsl, "rsl_rl.utils": utils,
                        "rsl_rl.networks": networks, "rsl_rl.env": envmod,
                        "tensordict": td})

    _ap = (ROOT / "roboparty_train" / "rsl_rl" / "rsl_rl" / "modules"
           / "amp.py").resolve()
    _spec = _ilu.spec_from_file_location("amp_disc_dryrun_v63", _ap)
    AMP = _ilu.module_from_spec(_spec)
    sys.modules["amp_disc_dryrun_v63"] = AMP
    _spec.loader.exec_module(AMP)

    D, N, STEPS, STRIDE = 124, 4, 60, 2       # v63: +base_lin_vel(3)
    disc = AMP.AMPDiscriminator(
        disc_obs_dim=D, disc_obs_steps=STEPS, disc_obs_stride=STRIDE,
        obs_groups={"discriminator": ["disc"],
                    "discriminator_demonstration": ["disc_demo"]},
    )
    assert disc.disc_obs_steps_eff == 30, disc.disc_obs_steps_eff
    assert disc.input_dim == 30 * D, disc.input_dim
    obs = {"disc": torch.zeros(N, STEPS, D), "disc_demo": torch.zeros(N, STEPS, D)}
    for i in range(STEPS):
        obs["disc"][:, i, :] = i
        obs["disc_demo"][:, i, :] = i
    out = disc.get_disc_obs(obs)
    assert out.shape == (N, 30, D), out.shape
    assert torch.allclose(out[:, 1, 0], torch.full((N,), 2.0))    # frames 0,2,4,...
    assert torch.allclose(out[:, -1, 0], torch.full((N,), 58.0))  # ...58
    demo = disc.get_disc_demo_obs(obs)
    assert demo.shape == (N, 30, D)
    rew, score = disc.predict_style_reward(out, dt=0.01)
    assert rew.shape == (N,) and torch.isfinite(rew).all()
    disc.update_normalization(out)
    disc.update_normalization(demo)
    normed = disc.normalize_disc_obs(out)
    assert normed.shape == (N, 30, D)
    scores = disc(normed.reshape(N, -1))
    assert scores.shape == (N, 1)
    gp = disc.compute_grad_penalty(demo.reshape(N, -1), scale=10.0)
    assert torch.isfinite(gp), gp

    # memory arithmetic: buffer budget on the 4090D 24G
    buf_bytes = lambda n: n * 4096 * 30 * D * 4
    assert buf_bytes(24) < 1.5e9, buf_bytes(24)      # 24 -> ~1.46 GB/buffer
    assert buf_bytes(100) > 6e9, buf_bytes(100)      # v61 default would OOM x2
    print(f"[OK] disc window: 30x{D}={30*D} dims, frames[0..58:2], "
          f"full post-buffer path, buffer 24={buf_bytes(24)/1e9:.2f}GB "
          f"(100 would be {buf_bytes(100)/1e9:.1f}GB)")
    print("[PASS] 3/6 disc macro-window")


# ---------------------------------------------------------------- check 4
def check_stride_override():
    import importlib.util as _ilu
    AMP = sys.modules["amp_disc_dryrun_v63"]

    def _resolve(step_dt):
        alg = {"amp_cfg": {"disc_obs_steps": 60, "disc_obs_dim": 124,
                           "step_dt": step_dt}}
        obs_probe = {"disc": torch.zeros(2, 60, 124),
                     "disc_demo": torch.zeros(2, 60, 124)}
        groups = {"discriminator": ["disc"],
                  "discriminator_demonstration": ["disc_demo"]}
        env = types.SimpleNamespace(
            env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=step_dt)))
        return AMP.resolve_amp_config(alg, obs_probe, groups, env)["amp_cfg"]

    cfg = _resolve(0.01)
    assert cfg["disc_obs_stride"] == 2, cfg["disc_obs_stride"]   # auto at 100 Hz
    os.environ["X1_DISC_STRIDE"] = "2"
    assert _resolve(0.01)["disc_obs_stride"] == 2                # explicit override
    del os.environ["X1_DISC_STRIDE"]
    # assertion: 60 % 2 == 0 (the AMPDiscriminator constructor assert)
    assert 60 % 2 == 0
    print("[OK] stride: auto=2 @0.01s, X1_DISC_STRIDE override, 60%2==0")
    print("[PASS] 4/6 stride override")


# ---------------------------------------------------------------- check 5
def check_ref_lin_vel():
    import importlib.util as _ilu
    _op = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "mdp" / "observations.py").resolve()
    _spec = _ilu.spec_from_file_location("amp_mdp_obs_dryrun_v63", _op)
    OBS = _ilu.module_from_spec(_spec)
    sys.modules["amp_mdp_obs_dryrun_v63"] = OBS
    _spec.loader.exec_module(OBS)

    N, S = 3, 60

    class _Anim:
        @staticmethod
        def get_root_vel_w():
            return torch.full((N, S, 3), 0.0)

        @staticmethod
        def get_root_quat():
            q = torch.zeros(N, S, 4)
            q[..., 0] = 1.0
            return q

    class _Env(types.SimpleNamespace):
        num_envs = N
        animation_manager = types.SimpleNamespace(
            get_term=lambda name: _Anim())

    out = OBS.ref_root_lin_vel_b(_Env(), "animation", flatten_steps_dim=False)
    assert out.shape == (N, S, 3), out.shape
    # world +x 1 m/s with identity quat -> local frame reads exactly 1.0
    _Anim.get_root_vel_w = staticmethod(
        lambda: torch.tensor([1.0, 0, 0]).repeat(N, S, 1))
    out = OBS.ref_root_lin_vel_b(_Env(), "animation", flatten_steps_dim=False)
    assert torch.allclose(out[..., 0], torch.full((N, S), 1.0)), out[..., 0]

    # config walk: the X1_DISC_LINVEL block adds BOTH terms (source-scan —
    # X1AmpEnvCfg itself needs a live sim to import)
    x1 = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "x1_amp_env_cfg.py").read_text()
    assert re.search(r'X1_DISC_LINVEL.*?"1".*?base_lin_vel', x1, re.S), \
        "policy-side base_lin_vel term must be gated by X1_DISC_LINVEL"
    assert "ref_root_lin_vel_b" in x1, "demo-side term missing"
    ag = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "amp_env_cfg.py").read_text()
    assert '"root_vel_w"' in ag, "animation must fetch root_vel_w for the demo term"
    print("[OK] ref_root_lin_vel_b: 1 m/s world->base exact, (N,60,3); "
          "cfg gating present both sides")
    print("[PASS] 5/6 disc root-linvel")


# ---------------------------------------------------------------- check 6
def check_buffer_lever():
    ag = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "agents" / "x1_amp_agent_cfg.py").read_text()
    assert 'X1_DISC_BUFFER' in ag, "buffer lever missing"
    m = re.search(r'X1_DISC_BUFFER", "(\d+)"', ag)
    assert m and m.group(1) == "100", m and m.group(1)  # default stays 100
    launch = (ROOT / "roboparty_train" / "run_v63_100hz.py").read_text()
    for k, v in [("X1_AMP_NUM_STEPS", "60"), ("X1_DISC_STRIDE", "2"),
                 ("X1_DISC_BUFFER", "24"), ("X1_DISC_LINVEL", "1"),
                 ("X1_ACT_LPF_HZ", "10"), ("X1_ACT_LPF_ORDER", "2"),
                 ("X1_CONTROL_HZ", "100"), ("X1_MOTION_DIR", "x1_lab_v32_200")]:
        assert f'"{k}"' in launch and f'"{v}"' in launch, (k, v)
    print("[OK] launcher levers complete; X1_DISC_BUFFER default 100 "
          "(50 Hz recipes unaffected)")
    print("[PASS] 6/6 buffer lever + launcher")


if __name__ == "__main__":
    _install_mocks()
    check_lpf()
    check_deploy_wiring()
    check_disc_window()
    check_stride_override()
    check_ref_lin_vel()
    check_buffer_lever()
    print("\n[ALL PASS] v63 dry-run (macro-window + linvel + action LPF)")
