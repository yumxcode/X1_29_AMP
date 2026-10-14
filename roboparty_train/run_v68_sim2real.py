#!/usr/bin/env python3
"""v68 launcher: soup59c recipe + sim2real-identified dynamics (X1, 50 Hz).

用户指令（contract rev2/rev3）: 训练对象 X1（非 F1）；按 soup59c 训练策略；
基座 = sim2real 上实际部署的 soup59c 模型；控制频率 50 Hz。

What changed vs the champion's own training (v59c cell):
  * assets/robots/x1.py actuators = x1_armature_pd.md §1 (identified
    armature 0.012-0.273 per joint; kp 65-100 / kd 8 on the heavy chain)
  * EventCfg: armature rand (0.8,1.3), damping rand (0.8,1.0)  [md §2]
  * sim2sim mujoco_rollout.py PD/ARMATURE tables = same values  [md §3]

Motivation (local A/B, acceptance/v68_eval/baseline_soup59c_*):
  soup59c policy  legacy dyn   11.5 s, 10.82 m @1.0 m/s (champion behavior)
  soup59c policy  v68 dyn      FALL 1.66 s @1.0 / 1.72 s @0.5 — the
  identified dynamics BREAK the deployed policy; retraining is required.

Recipe = the v59c native cell unchanged (robust 3 / form guards / disc vel /
yaw guard 1.0) + fine-tune from checkpoints/model_soup59c.pt (iter 9500,
full resumable ckpt incl. discriminator state).

Iteration budget: 1200 (v59c's +600 was a same-dynamics recipe tweak; this
run re-adapts to a -46% kp / 20x armature plant). save_interval 500 ->
ladder m10000 / m10500 / m10699 for local battery + optional soup merge
(the exact v59c -> soup59c delivery workflow). 50 Hz control (default;
X1_CONTROL_HZ set explicitly for self-documentation).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"           # v59c native dose
os.environ["X1_FINE_TUNE_ITERS"] = "1200"
os.environ.setdefault("X1_CONTROL_HZ", "50") # contract: 50 Hz (== default)

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_soup59c.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_soup59c*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("soup59c base checkpoint not found "
                                "(roboparty_train/checkpoints/model_soup59c.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
