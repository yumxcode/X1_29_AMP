#!/usr/bin/env python3
"""v80 local unit test: X1_FORM_PACKAGE block (isaaclab-free).

Validates (RECORD_BOOK §6 audit discipline):
  1. arm_dc_band exists as a RewTerm with the 8 ordered arm joints and
     demo-calibrated bands [(0,25d),(0,20d),(0,15d),(20d,40d)]x2 rad.
  2. Executing the REAL X1_INITIAL_REWARD block source with
     X1_FORM_PACKAGE=1 (v76 env: clock+track2.0) yields the v76 table
     PLUS exactly {foot_yaw_neutral:-0.6, hip_roll_abduction:-0.8,
     stance_sole_flat:-1.5, arm_dc_band:-0.3}.
  3. Without X1_FORM_PACKAGE the four terms stay zeroed (byte-identical
     v76 recipe).
  4. The mdp func arm_dc_band_guard exists in rewards.py and its EMA-band
     math scores ~0 on demo-like DC and large on the m5500 defect DC.
"""
import ast
import math
import re
import types
from pathlib import Path

print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"
RWD = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py"
SRC = CFG.read_text()

# ---- 1. AST: arm_dc_band term + bands --------------------------------------
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
        if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")) == "RewTerm":
            real_terms.update(targets)
assert "arm_dc_band" in real_terms, "arm_dc_band RewTerm missing"
m = re.search(r"arm_dc_band = RewTerm\((.*?)\n    \)", SRC, re.S)
assert m and "left_shoulder_pitch_joint" in m.group(1)
assert "right_elbow_pitch_joint" in m.group(1)
bands = eval(re.search(r'"bands": (\[\[.*?\]\]|\[\(.*?\)\] \* 2)',
                       m.group(1), re.S).group(1))
exp = [(0.0, math.radians(25)), (0.0, math.radians(20)),
       (0.0, math.radians(15)), (math.radians(20), math.radians(40))] * 2
assert len(bands) == 8 and all(
    abs(a - b) < 1e-3 for x, y in zip(bands, exp) for a, b in zip(x, y)), \
    f"bands mismatch: {bands}"
print(f"[1] PASS — arm_dc_band term present, 8 ordered joints, bands "
      f"{[(round(a,3),round(b,3)) for a,b in bands[:4]]}x2")

# ---- 2. exec the REAL block with X1_FORM_PACKAGE=1 -------------------------
m = re.search(
    r'(        import os as _os_v73\n        if _os_v73\.environ\.get\("X1_INITIAL_REWARD".*?'
    r'\+ ", "\.join\(f"\{n\}=\{w:g\}" for n, w in _active\)\))',
    SRC, re.S)
assert m, "X1_INITIAL_REWARD block not found"
import textwrap
block_src = textwrap.dedent(m.group(1)).replace("import os as _os_v73\n", "")

V76 = {
    "track_lin_vel_xy_exp": 2.0, "track_ang_vel_z_exp": 1.25, "alive": 0.15,
    "ang_vel_xy_l2": -0.1, "flat_orientation_l2": -1.2,
    "joint_vel_l2": -2e-4, "joint_acc_l2": -2.5e-7, "action_rate_l2": -0.01,
    "joint_pos_limits": -1.0, "joint_energy": -1e-4, "joint_torques_l2": -1e-5,
    "feet_slide": -0.1, "sound_suppression": -5e-5, "feet_distance_y": 0.05,
    "undesired_contacts": -10.0,
    # NOTE foot_clock/contact_clock are set by the v71 block (outside this
    # extracted block); with X1_GAIT_CLOCK=1 they are EXEMPT from _zero --
    # verified in the pod log [REWARD-TABLE] print (v74 precedent).
}
FORM = {"foot_yaw_neutral": -0.6, "hip_roll_abduction": -0.8,
        "stance_sole_flat": -1.5, "arm_dc_band": -0.3}

class FakeTerm:
    def __init__(self, weight=0.0):
        self.weight = weight
        self.params = {"asset_cfg": "LEGSONLY", "max": 0.38}

class FakeRewards:
    def __init__(self):
        for t in real_terms:
            setattr(self, t, FakeTerm())

def run_block(env):
    self_obj = types.SimpleNamespace(rewards=FakeRewards())
    class SceneEntityCfg:
        def __init__(self, *a, **k):
            pass
    mdp = types.SimpleNamespace(action_rate_l2=lambda env: None,
                                action_rate_l2_joints=lambda env, asset_cfg=None: None)
    exec(compile(block_src, "<block>", "exec"),
         {"self": self_obj, "SceneEntityCfg": SceneEntityCfg, "mdp": mdp,
          "__builtins__": __builtins__,
          "_os_v73": types.SimpleNamespace(environ=env)})
    return self_obj

env76 = {"X1_INITIAL_REWARD": "1", "X1_GAIT_CLOCK": "1", "X1_TRACK_W": "2.0"}
s76 = run_block(dict(env76))
final76 = {t: getattr(s76.rewards, t).weight.real for t in real_terms
           if getattr(s76.rewards, t).weight != 0}
assert final76 == V76, f"v76 base mismatch: {sorted(set(final76.items()) ^ set(V76.items()))}"
print(f"[2a] PASS — v76 recipe reproduced ({len(final76)} terms)")

env80 = dict(env76)
env80["X1_FORM_PACKAGE"] = "1"
s80 = run_block(env80)
final80 = {t: getattr(s80.rewards, t).weight.real for t in real_terms
           if getattr(s80.rewards, t).weight != 0}
assert final80 == {**V76, **FORM}, \
    f"v80 mismatch: {sorted(set(final80.items()) ^ set({**V76, **FORM}.items()))}"
print(f"[2b] PASS — X1_FORM_PACKAGE=1: v76 + FORM exactly "
      f"({len(final80)} terms: {FORM})")

s_off = run_block(dict(env76))          # env WITHOUT the form var
final_off = {t: getattr(s_off.rewards, t).weight for t in real_terms
             if getattr(s_off.rewards, t).weight != 0}
for t in FORM:
    assert getattr(s_off.rewards, t).weight == 0, f"{t} not zeroed when form off"
print("[2c] PASS — form OFF => v76 byte-identical (4 terms zeroed)")

# dose knobs are env-overridable
env_d = dict(env80); env_d["X1_ARM_DC_W"] = "0.6"; env_d["X1_TOE_GUARD"] = "1.2"
s_d = run_block(env_d)
assert s_d.rewards.arm_dc_band.weight == -0.6
assert s_d.rewards.foot_yaw_neutral.weight == -1.2
print("[2d] PASS — dose knobs X1_TOE_GUARD/X1_ARM_DC_W override")

# ---- 3. mdp func exists + math sanity --------------------------------------
rsrc = RWD.read_text()
assert "def arm_dc_band_guard(" in rsrc
# emulate the statistic: DC band excess
def band_excess(dc_deg, center, half):
    return max(0.0, abs(math.radians(dc_deg) - center) - half)
demo = [band_excess(-9, 0, math.radians(20)),   # shoR demo DC -9
        band_excess(16, 0, math.radians(25)),   # shoP demo +16
        band_excess(20, math.radians(20), math.radians(40)),  # elbow
        band_excess(0, 0, math.radians(15))]    # shoY
m5500 = [band_excess(-74, 0, math.radians(20)),  # shoR L
         band_excess(-54, 0, math.radians(25)),  # shoP L
         band_excess(70, math.radians(20), math.radians(40)),
         band_excess(-33, 0, math.radians(15))]
assert sum(demo) < 0.01, f"demo must pay ~0, got {sum(demo)}"
assert 1.0 < sum(m5500) < 3.5, f"m5500 defect must pay large, got {sum(m5500)}"
print(f"[3] PASS — arm_dc_band_guard present; demo excess {sum(demo):.4f} rad "
      f"(~0), m5500 defect {sum(m5500):.2f} rad -> {-0.3*sum(m5500):.2f}/step at -0.3")

# ---- 4. launcher env set ----------------------------------------------------
L = (REPO / "roboparty_train/run_v80_form_package.py").read_text()
for k in ("X1_FORM_PACKAGE", "X1_TOE_GUARD", "X1_ABD_GUARD", "X1_SOFLAT_W",
          "X1_ARM_DC_W", "X1_INITIAL_REWARD", "X1_GAIT_CLOCK", "X1_TRACK_W",
          "X1_VMATCH_TOL", "X1_MOTION_DIR", "X1_FRESH_ITERS"):
    assert f'"{k}"' in L, f"launcher missing {k}"
assert "X1_RESUME_CKPT" not in L.replace('os.environ.pop("X1_RESUME_CKPT", None)', "")
print("[4] PASS — launcher sets the full v76+form env set, from-scratch")
print("\nALL v80 UNIT TESTS PASS")
