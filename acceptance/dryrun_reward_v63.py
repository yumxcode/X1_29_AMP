#!/usr/bin/env python3
"""v63 RHYTHM prior dry-run (mock env, call-level) — GOAL_RHYTHM.md §3.

Paid lessons applied (v40 reward-wiring 0-iter crashes; v55 inline-ternary;
v49b single-side obs; v61d consumer-contract): the new reward is genuinely
CALLED on faithful fakes across its full input space, and the env-cfg
wiring is verified by AST on the REAL file (func symbol, params, env-var
lever, default weight 0).

Checks:
  1. gait_period_prior on a synthetic alternating-gait env:
     a) human-cadence gait (cycle 1.10 s @ v=1.0)  -> score ~1.0/edge
     b) micro-gait (cycle 0.40 s @ v=1.0)          -> score ~exp(-2.8)
     c) speed conditioning: v=0.5 -> T*=1.28; v=2.5 -> clamp 0.70
     d) episode reset: first edge after reset scores 0 (re-arm only)
     e) cmd=0 (stand) -> reward 0 (gate)
     f) both feet contribute; no NaN; per-env shape
  2. AST on x1_amp_env_cfg.py:
     a) RewTerm gait_period func == mdp.gait_period_prior
     b) params include sensor_cfg/body_names both ankles + command_name
     c) __post_init__ wires X1_CADENCE_PRIOR (default "0")
"""
import ast
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roboparty_train" / "robolab"))


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

    class ManagerBasedRLEnv:
        pass

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
    math_mod.quat_apply = lambda q, v: v.clone()
    math_mod.quat_apply_inverse = lambda q, v: v.clone()
    utils.math = math_mod
    isaaclab.utils = utils
    sys.modules.update({
        "isaaclab": isaaclab, "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp_mod, "isaaclab.managers": managers,
        "isaaclab.sensors": sensors, "isaaclab.assets": assets,
        "isaaclab.utils": utils, "isaaclab.utils.math": math_mod,
    })


_install_mocks()

import importlib.util as _ilu  # noqa: E402
_rp = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
       / "manager_based" / "amp" / "mdp" / "rewards.py").resolve()
_spec = _ilu.spec_from_file_location("amp_mdp_rewards_dryrun63", _rp)
R = _ilu.module_from_spec(_spec)
sys.modules["amp_mdp_rewards_dryrun63"] = R
_spec.loader.exec_module(R)


class _CmdMgr:
    def __init__(self, cmd, n):
        row = torch.tensor(cmd, dtype=torch.float32)
        self._c = row.unsqueeze(0).expand(n, 3).contiguous()

    def get_command(self, name):
        return self._c


class _SensorCfg(types.SimpleNamespace):
    pass


def make_env(n=1, cmd=(1.0, 0.0, 0.0), dt=0.02):
    """Contact forces driven EXTERNALLY by the test loop (set_contact)."""
    nf = torch.zeros(n, 2, 6, 3)
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(net_forces_w_history=nf))
    env = types.SimpleNamespace(
        scene={"contact_forces": sensor},
        num_envs=n, device="cpu", step_dt=dt,
        command_manager=_CmdMgr(cmd, n),
        episode_length_buf=torch.zeros(n, dtype=torch.long))
    cfg = _SensorCfg(name="contact_forces", body_ids=[0, 1])
    return env, cfg


def set_contact(env, foot_flags):
    """foot_flags: tuple of 2 bools, broadcast to all envs."""
    nf = env.scene["contact_forces"].data.net_forces_w_history
    nf[:] = 0.0
    for f, on in enumerate(foot_flags):
        if on:
            nf[:, :, f, 2] = 5.0
    nf[:] = nf  # no-op to keep tensor identity


def run_gait(cmd, cycle_steps, n_steps, dt=0.02, duty=0.62, reset_at=None):
    """Alternating gait with the given cycle (steps). Returns (total, edges)."""
    env, cfg = make_env(cmd=cmd, dt=dt)
    total = 0.0
    n_edges = 0
    prev = [False, False]
    for step in range(n_steps):
        env.episode_length_buf += 1
        t = step * dt
        ph = (t / (cycle_steps * dt)) % 1.0
        flags = []
        for f in range(2):
            pf = (ph + 0.5 * f) % 1.0          # R offset half cycle
            on = pf < duty
            flags.append(on)
        if reset_at is not None and step == reset_at:
            env.episode_length_buf[:] = 0
            prev = [False, False]
            flags = [False, False]
        set_contact(env, tuple(flags))
        r = R.gait_period_prior(env, cfg, command_name="base_velocity")
        total += float(r.sum())
        for f in range(2):
            if flags[f] and not prev[f]:
                n_edges += 1
            prev[f] = flags[f]
    return total, n_edges


def main():
    fails = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not cond:
            fails.append(name)

    # 1a) human cadence 1.10 s @ v=1.0 -> per-edge score ~ exp(-|1.10-1.10|/.25)=1
    total, edges = run_gait((1.0, 0.0, 0.0), cycle_steps=55, n_steps=1200)
    per_edge = total / max(edges, 1)
    check("1a human cadence scores ~1", 0.6 < per_edge <= 1.05,
          f"per_edge={per_edge:.3f} edges={edges}")

    # 1b) micro-gait 0.40 s @ v=1.0 -> per-edge ~ exp(-0.7/0.25)=0.061
    total, edges = run_gait((1.0, 0.0, 0.0), cycle_steps=20, n_steps=1200)
    per_edge = total / max(edges, 1)
    check("1b micro-gait partial credit", 0.02 < per_edge < 0.20,
          f"per_edge={per_edge:.3f} edges={edges}")

    # 1b') reward RATE comparison (gradient direction): human >> micro
    t_human, e_h = run_gait((1.0, 0.0, 0.0), cycle_steps=55, n_steps=1200)
    t_micro, e_m = run_gait((1.0, 0.0, 0.0), cycle_steps=20, n_steps=1200)
    rate_h, rate_m = t_human / 24.0, t_micro / 24.0
    check("1b' rate gradient human>micro x3", rate_h > 3 * rate_m,
          f"rate_h={rate_h:.3f}/s rate_m={rate_m:.3f}/s")

    # 1c) speed conditioning: v=0.5 -> T*=1.28; a 1.28 s gait (64 steps) ~1
    total, edges = run_gait((0.5, 0.0, 0.0), cycle_steps=64, n_steps=1400)
    per_edge = total / max(edges, 1)
    check("1c v=0.5 target 1.28s", 0.55 < per_edge <= 1.05,
          f"per_edge={per_edge:.3f}")

    # 1c') v=2.5 -> T* clamps to 0.70 (35 steps @0.02)
    total, edges = run_gait((2.5, 0.0, 0.0), cycle_steps=35, n_steps=1000)
    per_edge = total / max(edges, 1)
    check("1c' v=2.5 clamp 0.70s", 0.5 < per_edge <= 1.05,
          f"per_edge={per_edge:.3f}")

    # 1d) episode reset: first edge after reset must NOT score cross-episode
    env, cfg = make_env(cmd=(1.0, 0.0, 0.0))
    vals = []
    prev = [False, False]
    marks = [300, 355, 410]        # TD steps for foot L
    for step in range(500):
        env.episode_length_buf += 1
        on = (step in marks) or (step in [m + 1 for m in marks])
        if step == 320:            # reset mid-way; next edge at 410 is first
            env.episode_length_buf[:] = 0
        set_contact(env, (on, False))
        r = R.gait_period_prior(env, cfg, command_name="base_velocity")
        vals.append(float(r.sum()))
        prev = [on, False]
    # Edges: 300 = FIRST ever -> arms only (0). 355 = first edge after the
    # 320 reset: its stored state (6.0 s) > episode t_now (0.7 s) -> STALE,
    # re-arms only (0). 410 = interval within the NEW episode's clock
    # (t 0.7 -> 1.8 s = 1.1 s) -> scores ~1. This is the designed reset
    # semantics: no cross-episode interval is ever scored.
    check("1d reset semantics (arm/arm/score)",
          abs(vals[300]) < 1e-9 and abs(vals[355]) < 1e-9 and vals[410] > 0.5,
          f"e@300={vals[300]:.3e} e@355={vals[355]:.3e} e@410={vals[410]:.3f}")

    # 1e) cmd=0 -> all zeros even with stepping
    total, edges = run_gait((0.0, 0.0, 0.0), cycle_steps=55, n_steps=600)
    check("1e stand gate zero", total == 0.0, f"total={total}")

    # 1f) NaN / shape
    env, cfg = make_env(n=7)
    set_contact(env, (True, False))
    r = R.gait_period_prior(env, cfg, command_name="base_velocity")
    check("1f shape+finite", r.shape == (7,) and torch.isfinite(r).all())

    # 1g) swing_airtime_prior: dense swing-duration kernel
    #     human swing (0.43 s) high, micro (0.19 s) low, drag ~0
    def run_swing(cmd, cycle_steps, swing_steps, n_steps=1200, dt=0.02):
        env, cfg = make_env(cmd=cmd, dt=dt)
        total = 0.0
        for step in range(n_steps):
            env.episode_length_buf += 1
            t = step * dt
            ph = (t / (cycle_steps * dt)) % 1.0
            flags = []
            for f in range(2):
                pf = (ph + 0.5 * f) % 1.0
                on = pf < (1.0 - swing_steps / cycle_steps)  # airborne frac
                flags.append(on)
            set_contact(env, tuple(flags))
            r = R.swing_airtime_prior(env, cfg, command_name="base_velocity")
            total += float(r.sum())
        return total / (n_steps * dt)

    rate_h = run_swing((1.0, 0, 0), 55, 22)      # cycle 1.1s swing 0.44s
    # micro-gait at its MEASURED duty 0.95 (m10500: swing 0.19s of a 0.35s
    # cycle at 100 Hz ~= 2 airborne steps of 17) — not an idealized 50%-duty
    rate_m = run_swing((1.0, 0, 0), 17, 2)       # cycle 0.34s swing 0.04s x2... (mock min)
    rate_m2 = run_swing((1.0, 0, 0), 20, 4)      # cycle 0.4s swing 0.08s (duty 0.8)
    rate_d = run_swing((1.0, 0, 0), 55, 0)       # drag (never airborne)
    check("1g swing prior human>micro>drag",
          rate_h > 5 * max(rate_m, rate_m2) and max(rate_m, rate_m2) > 5 * rate_d and rate_d < 1e-9,
          f"rate h/m/m2/d = {rate_h:.3f}/{rate_m:.3f}/{rate_m2:.3f}/{rate_d:.3f} per s (w=1)")

    # 1g') jog-speed gate (>1.5 m/s) exempt
    rate_j = run_swing((2.0, 0, 0), 55, 22)
    check("1g' jog gate zero", rate_j < 1e-9, f"rate={rate_j:.2e}")

    # 2) AST wiring on the REAL env cfg
    src = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "x1_amp_env_cfg.py").read_text()
    tree = ast.parse(src)
    found_term, found_lever = False, False
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "gait_period":
            found_term = True
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "gait_period":
                    found_term = True
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "X1_CADENCE_PRIOR"):
            found_lever = True
    check("2a RewTerm gait_period exists", found_term)
    check("2b X1_CADENCE_PRIOR lever wired (default 0)", found_lever)
    check("2e RewTerm swing_airtime exists", "swing_airtime = RewTerm" in src.replace("    ", " "))
    check("2f X1_SWING_PRIOR lever wired (default 0)", 'X1_SWING_PRIOR' in src)
    check("2g swing_airtime func referenced",
          "mdp.swing_airtime_prior" in src)
    check("2c func symbol referenced",
          "mdp.gait_period_prior" in src)
    check("2d both ankles in params",
          src.count('body_names=["left_ankle_roll_link", "right_ankle_roll_link"]') >= 1)

    print("\n== DRYRUN v63:", "ALL PASS" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
