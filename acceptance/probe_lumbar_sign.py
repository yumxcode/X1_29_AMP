import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sim2sim'))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q  # noqa: E402

model, _ = build_model(ROOT / 'gmr_x1_assets/x1.xml')
data = mujoco.MjData(model)
BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in DEFAULT_Q}

for val, tag in [(0.0, 'zero'), (np.radians(15), '+15deg'), (np.radians(-15), '-15deg')]:
    data.qpos[:] = 0
    data.qpos[0:3] = [0, 0, 0.8]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr['lumbar_pitch_joint']] = val
    mujoco.mj_forward(model, data)
    S = (data.xpos[BID('left_shoulder_pitch_link')] + data.xpos[BID('right_shoulder_pitch_link')]) / 2
    v = S - data.xpos[BID('base_link')]
    tilt = np.degrees(np.arctan2(v[0], v[2]))
    print(f"lumbar_pitch {tag:8s}: base->shoulder-mid xz tilt = {tilt:+6.1f} deg "
          f"({'chest FORWARD' if tilt > 0 else 'chest BACKWARD'})")
