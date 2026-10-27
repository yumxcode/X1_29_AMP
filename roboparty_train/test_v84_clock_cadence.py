#!/usr/bin/env python3
"""v84 local unit test: clock cadence-command coupling (isaaclab-free)."""
import importlib.util, math, os, sys
from pathlib import Path
print = __import__("functools").partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
GC = REPO / "roboparty_train/robolab/robolab/tasks/manager_based/amp/mdp/gait_clock.py"
MR = REPO / "sim2sim/mujoco_rollout.py"

def load_clock(env):
    for k in ("X1_CLOCK_T_A", "X1_CLOCK_T_B"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("gc_test", GC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def _t(v):
    import torch
    return torch.tensor([v], dtype=torch.float32)

# [1] default byte-identity (v71 schedule)
m = load_clock({})
for v, want in [(0.0, 1.46), (0.5, 1.28), (1.0, 1.10), (1.25, 1.01), (2.0, 0.74)]:
    got = m.t_star(_t(v)).item()
    assert abs(got - want) < 1e-6, (v, got, want)
print("[1] PASS — defaults byte-identical (T*(0/0.5/1.0/1.25/2.0) = v71)")

# [2] v84 dose: A=1.61 B=0.66 -> T*(1.0)=0.95 (1.053 Hz), T*(0.5)=1.28 unchanged
m84 = load_clock({"X1_CLOCK_T_A": "1.61", "X1_CLOCK_T_B": "0.66"})
t10 = m84.t_star(_t(1.0)).item()
t05 = m84.t_star(_t(0.5)).item()
t00 = m84.t_star(_t(0.0)).item()
assert abs(t10 - 0.95) < 1e-6, t10
assert abs(t05 - 1.28) < 1e-6, t05
assert abs(t00 - 1.61) < 1e-6 and t00 <= 1.75
cad = 1.0 / t10
speed_pred = 1.86 * cad * 0.50      # empirical ratio from m3999/soup_ed75
assert 0.95 < speed_pred < 1.01, speed_pred
print(f"[2] PASS — v84 schedule: T*(1.0)={t10:.2f}s = {cad:.3f} Hz/foot "
      f"(predicted speed ~{speed_pred:.2f} m/s at 0.50 m step); T*(0.5) unchanged")

# [3] R1/R2 feasibility at the new cadence (duty 0.61 at v=1.0)
d = m84.duty_star(_t(1.0)).item()
swing = (1 - d) * t10
assert cad <= 1.20, cad              # R1 gate
assert swing >= 0.28, swing          # R2 gate
print(f"[3] PASS — gates feasible: cadence {cad:.2f} <= 1.20 Hz, swing {swing:.2f} s >= 0.28")

# [4] sim2sim twin mirrors the knobs (source scan: no hardcoded 1.46 left in
# the rollout clock block; env-var read present)
src = MR.read_text()
assert 'X1_CLOCK_T_A' in src and 'X1_CLOCK_T_B' in src
import re as _re
clock_block = _re.search(r"GAIT_CLOCK_OBS:.*?_T = min\(max\(_tA - _tB \* _v, 0\.70\), 1\.75\)", src, _re.S)
assert clock_block, "twin clock block not updated"
assert "1.46 - 0.36 * _v" not in src
print("[4] PASS — sim2sim twin reads the same env knobs (hardcoded schedule removed)")

# [5] launcher contract
L = (REPO / "roboparty_train/run_v84_clock_cadence.py").read_text()
for k in ('X1_CLOCK_T_A"] = "1.61"', 'X1_CLOCK_T_B"] = "0.66"',
          "model_4500_v83d.pt", 'X1_TRACK_W"] = "2.25"',
          'X1_STANCE_HINGE_W"] = "3.0"', 'X1_FINE_TUNE_ITERS"] = "1500"'):
    assert k in L, f"launcher missing {k}"
print("[5] PASS — launcher: clock knobs + d4500 recipe + embedded base +1500")
print("\nALL v84 UNIT TESTS PASS")

def _t(v):
    import torch
    return torch.tensor([v], dtype=torch.float32)
