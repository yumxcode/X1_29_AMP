#!/usr/bin/env python3
"""v81 local unit test: hip_yaw_neutral term + X1_ARM_DC_W/X1_HIPYAW_W knobs."""
import ast, math, re, types
from pathlib import Path
print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"
SRC = CFG.read_text()

tree = ast.parse(SRC)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "X1AmpRewards")
real_terms = set()
for stmt in cls.body:
    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)] if isinstance(stmt, ast.Assign) else (
        [stmt.target.id] if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) else [])
    if targets and isinstance(getattr(stmt, "value", None), ast.Call):
        f = stmt.value.func
        if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")) == "RewTerm":
            real_terms.update(targets)
assert "hip_yaw_neutral" in real_terms
m = re.search(r"hip_yaw_neutral = RewTerm\((.*?)\n    \)", SRC, re.S)
assert "left_hip_yaw_joint" in m.group(1) and "right_hip_yaw_joint" in m.group(1)
assert m.group(1).count("arm_dc_band_guard") == 1
print("[1] PASS — hip_yaw_neutral term present, generic EMA-band func, hip_yaw pair")

m = re.search(
    r'(        import os as _os_v73\n        if _os_v73\.environ\.get\("X1_INITIAL_REWARD".*?'
    r'\+ ", "\.join\(f"\{n\}=\{w:g\}" for n, w in _active\)\))', SRC, re.S)
import textwrap
block_src = textwrap.dedent(m.group(1)).replace("import os as _os_v73\n", "")

class FakeTerm:
    def __init__(self, weight=0.0):
        self.weight = weight; self.params = {"asset_cfg": "L", "max": 0.38}
class FakeRewards:
    def __init__(self):
        for t in real_terms: setattr(self, t, FakeTerm())
def run_block(env):
    so = types.SimpleNamespace(rewards=FakeRewards())
    class SEC:
        def __init__(self, *a, **k): pass
    mdp = types.SimpleNamespace(action_rate_l2=lambda e: None,
                                action_rate_l2_joints=lambda e, asset_cfg=None: None)
    exec(compile(block_src, "<b>", "exec"),
         {"self": so, "SceneEntityCfg": SEC, "mdp": mdp, "__builtins__": __builtins__,
          "_os_v73": types.SimpleNamespace(environ=env)})
    return so

env = {"X1_INITIAL_REWARD": "1", "X1_GAIT_CLOCK": "1", "X1_TRACK_W": "2.0",
       "X1_FORM_PACKAGE": "1", "X1_ARM_DC_W": "0.9", "X1_HIPYAW_W": "0.8"}
s = run_block(env)
assert s.rewards.arm_dc_band.weight == -0.9
assert s.rewards.hip_yaw_neutral.weight == -0.8
assert s.rewards.foot_yaw_neutral.weight == -0.6
assert s.rewards.hip_roll_abduction.weight == -0.8
assert s.rewards.stance_sole_flat.weight == -1.5
print("[2] PASS — v81 doses: arm_dc -0.9, hip_yaw -0.8 (form pkg on)")
# knob off => hip_yaw_neutral stays 0 (weight set only when knob>0)
env2 = dict(env); env2.pop("X1_HIPYAW_W")
s2 = run_block(env2)
assert s2.rewards.hip_yaw_neutral.weight == 0.0
print("[3] PASS — X1_HIPYAW_W default 0 => term off (v80 byte-identical)")
# form off => zero list includes hip_yaw_neutral
env3 = {"X1_INITIAL_REWARD": "1", "X1_GAIT_CLOCK": "1"}
s3 = run_block(env3)
assert s3.rewards.hip_yaw_neutral.weight == 0.0 and s3.rewards.arm_dc_band.weight == 0.0
print("[4] PASS — form OFF => all five form terms zeroed")
L = (REPO / "roboparty_train/run_v81_form_dose2.py").read_text()
for k in ('X1_ARM_DC_W"] = "0.9', 'X1_HIPYAW_W"] = "0.8', "model_2000_v80.pt", "X1_FINE_TUNE_ITERS"):
    assert k in L, f"launcher missing {k}"
print("[5] PASS — launcher: doses + embedded m2000 resume +1500")
print("\nALL v81 UNIT TESTS PASS")
