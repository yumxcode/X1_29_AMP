#!/usr/bin/env python3
"""v56 reward CALL-LEVEL dry-run (mock env, correct tensor shapes).

The #1 paid lesson of this repo (v40 three 0-iter crashes; v55 r1): every
new reward callable must be genuinely CALLED once — with a mock env and
correctly-shaped tensors — before it earns a platform slot. py_compile and
AST checks cannot see runtime shape/indexing bugs (the v55 inline-ternary
bug was syntactically valid).

This runs WITHOUT isaaclab: isaaclab modules are injected as sys.modules
mocks, then robolab...amp.mdp.rewards is imported and
knee_extension_stance is called across the calibration states:

  crouch 36 deg stance walk  -> ~0.113/foot (GOAL §4.1 calibration)
  extended 15 deg stance     -> 1.0/foot (saturated)
  jog cmd 2.5 m/s           -> 0 (walk gate)
  swing (no contact)        -> 0
  hyperextended -5 deg      -> 1.0/foot (kernel saturates, no pressure)
  mixed feet L stance R air -> single-foot contribution

Also exercises the OTHER terms touched by v56 wiring for import sanity:
arm_swing_amplitude_prior, arm_amp_phase_prior (weight 0 but must import),
stance_sole_flat_walk (unchanged reference).
"""
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roboparty_train" / "robolab"))


# ---------------------------------------------------------------- mocks
class _Math(types.SimpleNamespace):
    @staticmethod
    def quat_apply(q, v):
        # identity-rotation approximation is fine for the dry run (values
        # are not asserted on quat-dependent terms)
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

    class SceneEntityCfg(types.SimpleNamespace):
        def __init__(self, name, joint_names=None, body_names=None,
                     preserve_order=False):
            super().__init__(name=name, joint_names=joint_names,
                             body_names=body_names,
                             preserve_order=preserve_order,
                             joint_ids=None, body_ids=None)

    class ManagerBasedRLEnv:  # TYPE_CHECKING only, never instantiated
        pass

    envs.mdp = mdp_mod
    envs.ManagerBasedRLEnv = ManagerBasedRLEnv
    isaaclab.envs = envs
    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers
    sensors.ContactSensor = object
    sensors.RayCaster = object
    isaaclab.sensors = sensors
    assets.Articulation = object
    assets.RigidObject = object
    isaaclab.assets = assets
    math_mod.quat_apply = _Math.quat_apply
    utils.math = math_mod
    isaaclab.utils = utils

    sys.modules.update({
        "isaaclab": isaaclab, "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp_mod, "isaaclab.managers": managers,
        "isaaclab.sensors": sensors, "isaaclab.assets": assets,
        "isaaclab.utils": utils, "isaaclab.utils.math": math_mod,
    })


_install_mocks()

# load rewards.py BY FILE PATH: importing the robolab package drags in
# isaaclab_tasks/toml (unavailable locally) via __init__ chains — the
# call-level dry run only needs the module itself with isaaclab mocked.
import importlib.util as _ilu  # noqa: E402
_rp = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
       / "manager_based" / "amp" / "mdp" / "rewards.py").resolve()
_spec = _ilu.spec_from_file_location("amp_mdp_rewards_dryrun", _rp)
R = _ilu.module_from_spec(_spec)
sys.modules["amp_mdp_rewards_dryrun"] = R
_spec.loader.exec_module(R)


class _CmdMgr:
    def __init__(self, cmd, n):
        row = torch.tensor(cmd, dtype=torch.float32)
        self._c = row.unsqueeze(0).expand(n, 3).contiguous()

    def get_command(self, name):
        return self._c


def make_env(n=4, knee_rad=0.63, contact=(True, True), cmd=(1.0, 0.0, 0.0)):
    """Mock env matching IsaacLab shapes.

    - contact_forces.net_forces_w_history: (N, H, B, 3); H=2 frames;
      B=6 bodies with body_ids=[2, 3] as the two ankle_roll bodies.
      contact force 5 N (>1 threshold) when the foot is in stance.
    - robot.joint_pos: (N, 29); knee joints at ids 11 (L) and 23 (R).
    """
    ankle_ids = [2, 3]
    knee_ids = [11, 23]
    nf = torch.zeros(n, 2, 6, 3)
    for f, on in enumerate(contact):
        nf[:, :, ankle_ids[f], 2] = 5.0 if on else 0.0

    jp = torch.zeros(n, 29)
    for k in knee_ids:
        jp[:, k] = knee_rad

    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(net_forces_w_history=nf))
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(joint_pos=jp, default_joint_pos=torch.zeros(n, 29),
                                   joint_vel=torch.zeros(n, 29), joint_acc=torch.zeros(n, 29)))
    scene = {"contact_forces": sensor, "robot": asset}
    env = types.SimpleNamespace(
        scene=scene, num_envs=n, device="cpu", step_dt=0.02,
        command_manager=_CmdMgr(cmd, n),
        action_manager=types.SimpleNamespace(
            action=torch.zeros(n, 29), prev_action=torch.zeros(n, 29)),
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
    )
    return env, ankle_ids, knee_ids


def run_case(label, expect, **kw):
    env, ankle_ids, knee_ids = make_env(**kw)
    sensor_cfg = types.SimpleNamespace(name="contact_forces", body_ids=ankle_ids)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=knee_ids)
    out = R.knee_extension_stance(
        env, sensor_cfg, asset_cfg,
        command_name="base_velocity", max_cmd_speed=1.5,
        target=0.26, sigma=0.17)
    assert out.shape == (env.num_envs,), f"{label}: bad shape {tuple(out.shape)}"
    val = float(out[0])
    ok = abs(val - expect) < 0.02 * max(1.0, abs(expect)) + 1e-3
    print(f"  {label:42s} r={val:+.4f}  expect~{expect:+.4f}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    import math
    print("[dryrun] knee_extension_stance call-level cases:")
    r_crouch = math.exp(-(0.63 - 0.26) / 0.17)   # 36 deg crouch per foot
    r_15deg = 1.0                                 # at target -> saturated
    oks = [
        run_case("crouch 36deg, double stance, walk 1.0", 2 * r_crouch,
                 knee_rad=0.63, contact=(True, True), cmd=(1.0, 0.0, 0.0)),
        run_case("crouch, L stance only", r_crouch,
                 knee_rad=0.63, contact=(True, False), cmd=(1.0, 0.0, 0.0)),
        run_case("extended 15deg, double stance", 2.0,
                 knee_rad=0.26, contact=(True, True), cmd=(1.0, 0.0, 0.0)),
        run_case("hyperextended -5deg (saturate)", 2.0,
                 knee_rad=-0.09, contact=(True, True), cmd=(1.0, 0.0, 0.0)),
        run_case("jog cmd 2.5 -> walk gate OFF", 0.0,
                 knee_rad=0.26, contact=(True, True), cmd=(2.5, 0.0, 0.0)),
        run_case("cmd 1.5 boundary -> OFF (strict <)", 0.0,
                 knee_rad=0.26, contact=(True, True), cmd=(1.5, 0.0, 0.0)),
        run_case("swing both feet -> 0", 0.0,
                 knee_rad=0.26, contact=(False, False), cmd=(1.0, 0.0, 0.0)),
        run_case("25deg (mid-gradient)", 2 * math.exp(-(0.4363 - 0.26) / 0.17),
                 knee_rad=0.4363, contact=(True, True), cmd=(0.5, 0.0, 0.0)),
    ]
    # walk05 gate check: 0.5 m/s is INSIDE the walk regime
    oks.append(run_case("walk 0.5 crouch", 2 * r_crouch,
                        knee_rad=0.63, contact=(True, True), cmd=(0.5, 0.0, 0.0)))
    # import sanity for every other term the v56 cfg touches
    env, ankle_ids, _ = make_env()
    sc = types.SimpleNamespace(name="contact_forces", body_ids=ankle_ids)
    ac_all = types.SimpleNamespace(name="robot", joint_ids=list(range(29)))
    ac_pair = types.SimpleNamespace(name="robot", joint_ids=[17, 18])
    ac4 = types.SimpleNamespace(name="robot", joint_ids=[17, 18, 3, 9])
    ac_body = types.SimpleNamespace(name="robot", body_ids=[2, 3])
    quat = torch.zeros(env.num_envs, 6, 4); quat[:, :, 0] = 1.0
    env.scene["robot"].data.body_quat_w = quat
    env.scene["robot"].data.body_pos_w = torch.zeros(env.num_envs, 6, 3)
    env.scene["robot"].data.root_quat_w = torch.zeros(env.num_envs, 4); env.scene["robot"].data.root_quat_w[:, 0] = 1
    env.scene["robot"].data.root_lin_vel_w = torch.zeros(env.num_envs, 3)
    env.scene["robot"].data.root_pos_w = torch.zeros(env.num_envs, 3)
    env.scene["robot"].data.root_lin_vel_b = torch.zeros(env.num_envs, 3)
    env.scene["robot"].data.root_ang_vel_b = torch.zeros(env.num_envs, 3)
    env.scene["robot"].data.projected_gravity_b = torch.zeros(env.num_envs, 3)
    env.scene["robot"].data.projected_gravity_b[:, 2] = -1.0
    print("[dryrun] sibling-term call sanity (import + shape):")
    for name, fn, kwargs in [
        ("arm_swing_amplitude_prior", R.arm_swing_amplitude_prior, dict(asset_cfg=ac_pair)),
        ("arm_amp_phase_prior", R.arm_amp_phase_prior, dict(asset_cfg=ac4)),
        ("stance_sole_flat_walk", R.stance_sole_flat_walk, dict(sensor_cfg=sc, asset_cfg=ac_body)),
    ]:
        try:
            out = fn(env, **kwargs)
            assert out.shape == (env.num_envs,)
            print(f"  {name:32s} OK shape={tuple(out.shape)}")
            oks.append(True)
        except Exception as e:
            print(f"  {name:32s} FAIL: {type(e).__name__}: {e}")
            oks.append(False)
    print(f"[dryrun] {'ALL PASS' if all(oks) else 'FAILURES PRESENT'} ({sum(oks)}/{len(oks)})")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
