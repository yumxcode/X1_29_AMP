#!/usr/bin/env python3
"""v82 local unit test: band-tightening knobs (isaaclab-free)."""
import ast, math, re, types
from pathlib import Path
print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"
SRC = CFG.read_text()
tree = ast.parse(SRC)
assert "X1_SHOR_BAND_DEG" in SRC and "X1_ELBOW_HALF_DEG" in SRC and "X1_HIPYAW_BAND_DEG" in SRC
print("[1] PASS — band env knobs present in cfg source")

m = re.search(
    r'(        import os as _os_v73\n        if _os_v73\.environ\.get\("X1_INITIAL_REWARD".*?'
    r'\+ ", "\.join\(f"\{n\}=\{w:g\}" for n, w in _active\)\))', SRC, re.S)
import textwrap
block_src = textwrap.dedent(m.group(1)).replace("import os as _os_v73\n", "")

tree2 = ast.parse(SRC)
cls = next(n for n in tree2.body if isinstance(n, ast.ClassDef) and n.name == "X1AmpRewards")
real_terms = set()
for stmt in cls.body:
    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)] if isinstance(stmt, ast.Assign) else []
    if targets and isinstance(getattr(stmt, "value", None), ast.Call):
        f = stmt.value.func
        if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")) == "RewTerm":
            real_terms.update(targets)

class FakeTerm:
    def __init__(self, weight=0.0):
        self.weight = weight; self.params = {"asset_cfg": "L", "max": 0.38,
                                             "bands": "sentinel"}
class FakeRewards:
    def __init__(self):
        for t in real_terms: setattr(self, t, FakeTerm())
def run_block(env):
    so = types.SimpleNamespace(rewards=FakeRewards())
    class SEC:
        def __init__(self, *a, **k): pass
    mdp = types.SimpleNamespace(action_rate_l2=lambda e: None,
                                action_rate_l2_joints=lambda e, asset_cfg=None: None)
    g = {"self": so, "SceneEntityCfg": SEC, "mdp": mdp, "__builtins__": __builtins__,
         "_os_v73": types.SimpleNamespace(environ=env), "math": math}
    exec(compile(block_src, "<b>", "exec"), g)
    return so

base = {"X1_INITIAL_REWARD": "1", "X1_GAIT_CLOCK": "1", "X1_TRACK_W": "2.0",
        "X1_FORM_PACKAGE": "1", "X1_ARM_DC_W": "0.9", "X1_HIPYAW_W": "0.8"}
s = run_block(dict(base))
shor_bands = s.rewards.arm_dc_band.params["bands"]
assert abs(shor_bands[1][1] - math.radians(20)) < 1e-9  # default
assert abs(shor_bands[3][1] - math.radians(40)) < 1e-9
hy = s.rewards.hip_yaw_neutral.params["bands"]
assert abs(hy[0][1] - math.radians(8)) < 1e-9
print("[2] PASS — defaults byte-identical (shoR 20 / elbow 40 / hipyaw 8)")

v82 = dict(base); v82.update({"X1_SHOR_BAND_DEG": "12", "X1_ELBOW_HALF_DEG": "32", "X1_HIPYAW_BAND_DEG": "5"})
s2 = run_block(v82)
b2 = s2.rewards.arm_dc_band.params["bands"]
assert abs(b2[1][1] - math.radians(12)) < 1e-9
assert abs(b2[3][1] - math.radians(32)) < 1e-9
assert abs(b2[3][0] - math.radians(20)) < 1e-9
h2 = s2.rewards.hip_yaw_neutral.params["bands"]
assert abs(h2[0][1] - math.radians(5)) < 1e-9
assert s2.rewards.arm_dc_band.weight == -0.9 and s2.rewards.hip_yaw_neutral.weight == -0.8
print("[3] PASS — v82 bands: shoR 12 / elbow 20+-32 / hip_yaw 5; doses unchanged")

L = (REPO / "roboparty_train/run_v82_band_tighten.py").read_text()
for k in ('X1_SHOR_BAND_DEG"] = "12', 'X1_ELBOW_HALF_DEG"] = "32', 'X1_HIPYAW_BAND_DEG"] = "5', "model_2500_v81.pt", 'X1_FINE_TUNE_ITERS"] = "600'):
    assert k in L, f"launcher missing {k}"
print("[4] PASS — launcher sets v82 bands, base m2500, +600 short")
print("\nALL v82 UNIT TESTS PASS")
