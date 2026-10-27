#!/usr/bin/env python3
"""v83 local unit test: stance package knobs (isaaclab-free)."""
import ast, math, re, types
from pathlib import Path
print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"
SRC = CFG.read_text()

# [1] knobs + reward func present
assert "X1_STANCE_HINGE_W" in SRC and "X1_STANCE_HINGE_MAX" in SRC
RWD = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/rewards.py"
RSRC = RWD.read_text()
assert "def stance_separation_hinge" in RSRC
assert "stance_separation_hinge = RewTerm" in SRC
# the hinge term must be zeroed when FORM package off (byte-identity)
m0 = re.search(r'_zero \+= \["foot_yaw_neutral", "hip_roll_abduction",\n'
               r'\s*"stance_sole_flat", "arm_dc_band", "hip_yaw_neutral",\n'
               r'\s*"stance_separation_hinge"\]', SRC)
assert m0, "stance_separation_hinge missing from the FORM-off zero list"
print("[1] PASS — hinge func + RewTerm + FORM-off zero-list present")

# [2] the stance knobs are applied INSIDE the FORM package block, AFTER the
# unconditional feet_distance_y.params["max"] = 0.50 reset (ordering trap:
# the v69 block at ~line 1197 sets max from X1_STANCE_MAX too, but the
# initial-reward reset later overwrites it with 0.50 -- v83 re-applies)
reset_pos = SRC.find('self.rewards.feet_distance_y.params["max"] = 0.50')
knob_pos = SRC.find('environ.get("X1_STANCE_MAX", "0")', reset_pos)
hinge_pos = SRC.find('environ.get("X1_STANCE_HINGE_W", "0")', reset_pos)
assert 0 < reset_pos < knob_pos < hinge_pos, "stance knobs must follow the 0.50 reset"
print("[2] PASS — knobs applied after the initial-reward 0.50 reset (ordering)")

# [3] block execution: defaults byte-identical, v83 knobs bite
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
        self.weight = weight
        self.params = {"asset_cfg": "L", "max": 0.50, "bands": "sentinel"}
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
assert s.rewards.feet_distance_y.params["max"] == 0.50
assert s.rewards.stance_separation_hinge.weight == 0.0
print("[3] PASS — v81 defaults byte-identical (max 0.50, hinge 0)")

v83 = dict(base); v83.update({"X1_STANCE_MAX": "0.30", "X1_STANCE_HINGE_W": "3.0",
                              "X1_STANCE_HINGE_MAX": "0.32"})
s2 = run_block(v83)
assert s2.rewards.feet_distance_y.params["max"] == 0.30
assert s2.rewards.stance_separation_hinge.weight == -3.0
assert abs(s2.rewards.stance_separation_hinge.params["max"] - 0.32) < 1e-12
# non-FORM path (old recipes): hinge must stay 0 even with env var set
v83b = dict(v83); v83b["X1_FORM_PACKAGE"] = "0"
s3 = run_block(v83b)
assert s3.rewards.stance_separation_hinge.weight == 0.0
assert s3.rewards.feet_distance_y.params["max"] == 0.50
print("[4] PASS — v83 knobs bite inside FORM; FORM-off stays byte-identical")

# [5] hinge dose math on the audited defect: EMA sep 0.447 -> excess 0.127
excess = 0.447 - 0.32
assert abs(excess - 0.127) < 0.001
pay = 3.0 * excess
assert 0.35 < pay < 0.42, pay   # 0.38/step at the defect, 0 inside band
# reward function source sanity: linear hinge, EMA key, clamp min 0
mfn = re.search(r"def stance_separation_hinge\(.*?\n    return \(dc - max\)\.clamp\(min=0\.0\)",
                RSRC, re.S)
assert mfn and '_ema(env, "stancesep", distance, alpha)' in mfn.group(0)
assert "torch.exp" not in mfn.group(0).split('"""')[2] if '"""' in mfn.group(0) else True
print(f"[5] PASS — dose math: defect pays {pay:.2f}/step at w=3.0; linear hinge on 1 s EMA")

# [6] launcher contract
L = (REPO / "roboparty_train/run_v83_stance_package.py").read_text()
for k in ('X1_STANCE_MAX"] = "0.30', 'X1_STANCE_HINGE_W"] = "3.0',
          'X1_STANCE_HINGE_MAX"] = "0.32', "model_2500_v81.pt",
          'X1_FINE_TUNE_ITERS"] = "1500', 'X1_ARM_DC_W"] = "0.9',
          'X1_HIPYAW_W"] = "0.8', 'X1_TRACK_W"] = "2.0', 'X1_VMATCH_TOL"] = "0.15'):
    assert k in L, f"launcher missing {k}"
print("[6] PASS — launcher: stance knobs + v81 recipe + embedded m2500 +1500")
print("\nALL v83 UNIT TESTS PASS")
