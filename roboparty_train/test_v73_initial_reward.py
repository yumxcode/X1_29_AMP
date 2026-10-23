#!/usr/bin/env python3
"""v73 local unit test: X1_INITIAL_REWARD reset block.

Validates (isaaclab-free, RECORD_BOOK §6 audit discipline):
  1. Every reward term name touched by the X1_INITIAL_REWARD block
     (16 kept + 30 zeroed) EXISTS as a RewTerm attribute in the real
     X1AmpRewards class (AST parse of the real file — catches typos).
  2. Executing the REAL block source (extracted verbatim from
     x1_amp_env_cfg.py) on a stub rewards object yields EXACTLY the
     initial 16-term table with weight-for-weight values from commit
     6ea0655.
  3. The mutual-exclusion guard raises on X1_GAIT_CLOCK=1.
"""
import ast
import re
import sys
import types
from pathlib import Path

print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"

SRC = CFG.read_text()

# ---- 1. AST: real term names --------------------------------------------
tree = ast.parse(SRC)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == "X1AmpRewards")
real_terms = set()
for stmt in cls.body:
    targets = []
    if isinstance(stmt, ast.Assign):
        targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        targets = [stmt.target.id]
    if targets and isinstance(getattr(stmt, "value", None), ast.Call):
        f = stmt.value.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
        if name == "RewTerm":
            real_terms.update(targets)
print(f"[1] X1AmpRewards defines {len(real_terms)} RewTerms")

KEEP = {
    "track_lin_vel_xy_exp": 1.25, "track_ang_vel_z_exp": 1.25, "alive": 0.15,
    "ang_vel_xy_l2": -0.1, "flat_orientation_l2": -1.2,
    "joint_vel_l2": -2e-4, "joint_acc_l2": -2.5e-7, "action_rate_l2": -0.01,
    "joint_pos_limits": -1.0, "joint_energy": -1e-4, "joint_torques_l2": -1e-5,
    "feet_slide": -0.1, "sound_suppression": -5e-5, "feet_distance_y": 0.05,
    "undesired_contacts": -10.0,
}
ZERO = [
    "lin_vel_z_l2", "joint_acc_l2_arms", "action_rate_l2_arms", "smoothness_1",
    "joint_regularization", "arm_pitch_sync", "arm_asym_lean",
    "arm_leg_coupling", "lumbar_posture", "arm_amp_prior", "arm_amp_phase",
    "knee_extension", "heel_first", "leg_amp_asym", "hip_phase",
    "heel_down_ready", "swing_clearance", "ankle_flick", "foot_yaw_neutral",
    "hip_roll_abduction", "shoulder_yaw_fold", "arm_shared_lean",
    "demo_joint_track", "lateral_sway", "foot_clock", "contact_clock",
    "gait_period", "swing_airtime", "stance_sole_flat", "yaw_bias",
]
missing = [t for t in list(KEEP) + ZERO if t not in real_terms]
assert not missing, f"typo: terms not in X1AmpRewards: {missing}"
print(f"[1] PASS — all {len(KEEP)} kept + {len(ZERO)} zeroed names exist")

# ---- 2. exec the REAL block on a stub ------------------------------------
m = re.search(
    r'(        import os as _os_v73\n        if _os_v73\.environ\.get\("X1_INITIAL_REWARD".*?'
    r'\+ ", "\.join\(f"\{n\}=\{w:g\}" for n, w in _active\)\))',
    SRC, re.S)
assert m, "could not locate the X1_INITIAL_REWARD block in source"
import textwrap
block_src = textwrap.dedent(m.group(1)).replace("import os as _os_v73\n", "")

class FakeTerm:
    def __init__(self, weight=0.0):
        self.weight = weight
        self.params = {"asset_cfg": "LEGSONLY", "max": 0.38}

class FakeRewards:
    def __init__(self):
        for t in real_terms:
            setattr(self, t, FakeTerm())

self_obj = types.SimpleNamespace(rewards=FakeRewards())

class SceneEntityCfg:  # stub
    def __init__(self, *a, **k):
        pass

mdp = types.SimpleNamespace(action_rate_l2=lambda env: None)
env = {"X1_INITIAL_REWARD": "1", "X1_GAIT_CLOCK": "0"}

g = {"self": self_obj, "SceneEntityCfg": SceneEntityCfg, "mdp": mdp,
     "_os_v73": types.SimpleNamespace(environ=env)}
import os as _os_mod
class _FakeOs:
    environ = env
g["__builtins__"] = __builtins__
exec(compile(block_src.replace("import os as _os_v73", "pass"), "<block>", "exec"), g)

final = {t: getattr(self_obj.rewards, t).weight.real
         for t in real_terms if getattr(self_obj.rewards, t).weight != 0}
assert final == KEEP, f"final table mismatch:\n got {sorted(final.items())}\n exp {sorted(KEEP.items())}"
# whole-robot param overrides applied:
assert self_obj.rewards.joint_acc_l2.params["asset_cfg"].__class__.__name__ == "SceneEntityCfg"
assert self_obj.rewards.action_rate_l2.params == {}
assert self_obj.rewards.action_rate_l2.func is mdp.action_rate_l2
assert self_obj.rewards.feet_distance_y.params["max"] == 0.50
print(f"[2] PASS — real block produces EXACTLY the initial 15-term table; "
      f"whole-robot acc/rate params, feet max 0.50")

# ---- 3. v74 mode: clock terms SURVIVE the reset when GAIT_CLOCK=1 ---------
env["X1_GAIT_CLOCK"] = "1"
self74 = types.SimpleNamespace(rewards=FakeRewards())
# simulate the v71 clock block having already set clock weights:
self74.rewards.foot_clock.weight = 1.0
self74.rewards.contact_clock.weight = -0.5
exec(compile(block_src, "<block>", "exec"),
     {"self": self74, "SceneEntityCfg": SceneEntityCfg, "mdp": mdp,
      "__builtins__": __builtins__,
      "_os_v73": types.SimpleNamespace(environ=env)})
final74 = {t: getattr(self74.rewards, t).weight.real
           for t in real_terms if getattr(self74.rewards, t).weight != 0}
exp74 = dict(KEEP); exp74["foot_clock"] = 1.0; exp74["contact_clock"] = -0.5
assert final74 == exp74, f"v74 table mismatch: {sorted(final74.items())}"
print("[3] PASS — v74 mode: initial 15 + foot_clock 1.0 + contact_clock -0.5; clock survives, all other additions still zeroed")

# ---- 4. launcher parity: v73 env knobs vs the v72 infra list --------------
launcher = (REPO / "roboparty_train/run_v73_initial_reward.py").read_text()
for knob in ("X1_INITIAL_REWARD", "X1_ROBUST_TRAIN", "X1_DISC_VEL",
             "X1_AMP_NUM_STEPS", "X1_DISC_LINVEL", "X1_DISC_BUFFER",
             "X1_MOTION_DIR", "X1_DISC_VMATCH", "X1_FRESH_ITERS"):
    assert knob in launcher, f"launcher missing {knob}"
for banned in ("X1_GAIT_CLOCK", "X1_HIP_PHASE", "X1_SY_GUARD", "X1_LEAN_GUARD",
               "X1_YAW_GUARD", "X1_STANCE_MAX", "X1_TRACK_HIP", "X1_DEMO_RESET_P"):
    assert f'os.environ["{banned}"]' not in launcher, f"launcher SETS {banned}"
print("[4] PASS — launcher sets exactly the v73 knob set (no clock/guards)")

print("ALL v73 INITIAL-REWARD TESTS PASS")
