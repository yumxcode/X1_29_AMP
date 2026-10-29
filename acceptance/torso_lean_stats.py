"""Quantify torso / waist backward-lean (腰后仰) in a mujoco rollout npz.

Metrics:
  base_pitch_deg   : pelvis pitch from base_quat (world XY-projection; + = nose up = backward lean of pelvis top)
  lumbar_pitch_deg : lumbar_pitch_joint angle (+ = chest FORWARD, per probe_lumbar_sign)
  chest_tilt_deg   : base->shoulder-mid tilt via FK in the rollout's own model
Sign convention: chest BACKWARD (后仰) = negative lumbar / we report lean_angle where + = backward.
"""
import sys
import json
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sim2sim'))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q  # noqa: E402


def quat_to_euler_zyx(q):  # wxyz -> (roll_x, pitch_y, yaw_z)
    w, x, y, z = q
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return pitch


def torso_stats(npz_path, settle=0.5):
    model, _ = build_model(ROOT / 'gmr_x1_assets/x1.xml')
    data = mujoco.MjData(model)
    BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    base_id = BID('base_link')
    sh_id = (BID('left_shoulder_pitch_link') + BID('right_shoulder_pitch_link')) / 2

    d = np.load(npz_path, allow_pickle=True)
    q, bq, t = d['q'], d['base_quat'], d['t']
    jadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in DEFAULT_Q}
    # q columns are hinge order; map by meta hinge_names
    meta = json.loads(str(d['meta'].item()))
    names = meta['hinge_names']
    li = names.index('lumbar_pitch_joint')

    lum = np.degrees(q[:, li])
    base_pitch = np.degrees(np.array([quat_to_euler_zyx(bqq) for bqq in bq]))

    # chest tilt via FK per frame
    tilt = np.zeros(len(q))
    for i in range(len(q)):
        data.qpos[:] = 0
        data.qpos[0:3] = d['base_pos'][i]
        data.qpos[3:7] = bq[i]
        for j, n in enumerate(names):
            data.qpos[jadr[n]] = q[i, j]
        mujoco.mj_forward(model, data)
        v = data.xpos[BID('left_shoulder_pitch_link')] + data.xpos[BID('right_shoulder_pitch_link')]
        v = v / 2 - data.xpos[base_id]
        tilt[i] = np.degrees(np.arctan2(v[0], v[2]))  # + = chest forward

    # skip settle frames
    n0 = int(settle / 0.02)
    sl = slice(n0, None)
    return dict(
        file=str(npz_path),
        lumbar_pitch_mean=float(np.mean(lum[sl])), lumbar_pitch_p50=float(np.median(lum[sl])),
        lumbar_pitch_p10=float(np.percentile(lum[sl], 10)), lumbar_pitch_p90=float(np.percentile(lum[sl], 90)),
        base_pitch_mean=float(np.mean(base_pitch[sl])), base_pitch_p50=float(np.median(base_pitch[sl])),
        base_pitch_p10=float(np.percentile(base_pitch[sl], 10)), base_pitch_p90=float(np.percentile(base_pitch[sl], 90)),
        chest_tilt_mean=float(np.mean(tilt[sl])), chest_tilt_p50=float(np.median(tilt[sl])),
        chest_tilt_p10=float(np.percentile(tilt[sl], 10)), chest_tilt_p90=float(np.percentile(tilt[sl], 90)),
    )


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(torso_stats(p), indent=1))
